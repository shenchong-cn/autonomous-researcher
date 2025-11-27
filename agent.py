import os
import sys
import threading
import json
from typing import Optional, List

# 使用TUN模式（全局VPN）连接Modal服务器
print("[DEBUG] 使用TUN模式全局VPN连接Modal", file=sys.stderr)

# 清除所有代理设置，让TUN模式处理所有流量
proxy_vars = [
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    "GRPC_PROXY", "GRPC_TRACE", "GRPC_VERBOSITY"
]

for var in proxy_vars:
    os.environ.pop(var, None)

print("[DEBUG] 已清除所有代理设置，使用TUN模式", file=sys.stderr)

from google import genai
from google.genai import types

import anthropic

from logger import print_panel, print_status, log_step, logger

import modal
from modal.stream_type import StreamType

# Cache a single sandbox per run so the agent can keep state across tool calls.
_shared_sandbox: Optional[modal.Sandbox] = None
_shared_gpu: Optional[str] = None  # Track which GPU the sandbox was created with
_selected_gpu: Optional[str] = None  # User-selected GPU for this run


def emit_event(event_type: str, data: dict) -> None:
    """Emit a structured event for the frontend."""
    # Only emit structured events when explicitly enabled (e.g. from the web API).
    # This keeps the CLI output clean while still allowing rich UIs to subscribe.
    if not os.environ.get("AI_RESEARCHER_ENABLE_EVENTS"):
        return

    import json
    payload = {
        "type": event_type,
        "timestamp": 0,
        "data": data,
    }
    print(f"::EVENT::{json.dumps(payload)}")
    sys.stdout.flush()


def _build_generation_config(
    *,
    tools: Optional[list] = None,
    system_instruction: Optional[str] = None,
    disable_autofc: bool = False,
) -> types.GenerateContentConfig:
    """
    Build a GenerateContentConfig that:

    - Enables Gemini "thinking mode" with visible thought summaries.
    - Sets thinking_level=HIGH (recommended for Gemini 3 Pro).
    - Optionally disables automatic function calling so we can control
      when tools run and show thoughts before actions.
    """
    thinking_config = types.ThinkingConfig(
        thinking_level=types.ThinkingLevel.HIGH,
        include_thoughts=True,
    )

    config_kwargs = {
        "tools": tools,
        "system_instruction": system_instruction,
        "thinking_config": thinking_config,
    }

    if disable_autofc:
        # Turn off automatic Python function calling so we get function_call
        # parts back and can execute tools manually in our loop.
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True
        )

    return types.GenerateContentConfig(**config_kwargs)


def _get_shared_sandbox(gpu: Optional[str]) -> modal.Sandbox:
    """Create (once) and return a persistent sandbox for this run."""
    global _shared_sandbox, _shared_gpu
    if _shared_sandbox is not None:
        # Reuse only if GPU selection matches
        if gpu == _shared_gpu:
            return _shared_sandbox
        _close_shared_sandbox()

    log_step("EXECUTION", "Initializing shared Sandbox...")

    # Define a robust image with common dependencies (built once).
    image = (
        modal.Image.debian_slim()
        .pip_install("numpy", "pandas", "torch", "scikit-learn", "matplotlib")
    )

    # Create a Modal App to associate with the Sandbox
    log_step("EXECUTION", "Looking up Modal App 'agent-sandbox-app'...")
    app = modal.App.lookup("agent-sandbox-app", create_if_missing=True)
    log_step("EXECUTION", "Modal App found/created.")

    # Keep the sandbox alive by running an inert loop; subcommands run via sandbox.exec.
    gpu_msg = f"gpu={gpu}" if gpu else "cpu-only"
    log_step("EXECUTION", f"Creating persistent Sandbox (keep-alive loop, {gpu_msg})...")
    _shared_sandbox = modal.Sandbox.create(
        "bash",
        "-lc",
        "while true; do sleep 3600; done",
        app=app,
        image=image,
        timeout=7200,
        gpu=gpu,
    )
    _shared_gpu = gpu
    log_step("EXECUTION", "Persistent Sandbox ready.")
    return _shared_sandbox


def _close_shared_sandbox():
    """Terminate the shared sandbox if it exists."""
    global _shared_sandbox
    if _shared_sandbox is not None:
        try:
            _shared_sandbox.terminate()
            log_step("EXECUTION", "Persistent Sandbox terminated.")
        except Exception as e:
            log_step("WARNING", f"Failed to terminate sandbox cleanly: {e}")
        _shared_sandbox = None


def execute_in_sandbox(code: str):
    """
    Executes Python code inside a persistent Modal Sandbox using sandbox.exec.

    Behavior:
    - Starts a long-lived `python -u -` process in the sandbox.
    - Streams both STDOUT and STDERR to your local CLI *as they are produced*,
      similar to running a long training job in Colab.
    - Captures full STDOUT/STDERR buffers and returns them as a string so the
      agent can inspect logs after the run finishes.
    """
    try:
        sandbox = _get_shared_sandbox(_selected_gpu)

        log_step("EXECUTION", "Launching python exec inside Sandbox...")
        print_panel(code, "Sandbox Code", "code")

        # Use PIPE on both streams so we can capture and stream them ourselves.
        proc = sandbox.exec(
            "python",
            "-u",
            "-",
            stdout=StreamType.PIPE,
            stderr=StreamType.PIPE,
        )

        # Send the code into the sandboxed Python process.
        proc.stdin.write(code.encode("utf-8"))
        proc.stdin.write_eof()
        proc.stdin.drain()  # Flush buffered stdin

        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []

        log_step("EXECUTION", "Streaming stdout/stderr from Sandbox...")

        def _drain_stream(reader, buffer: List[str], is_stderr: bool):
            """Continuously read from a StreamReader and mirror to local stdout/stderr."""
            try:
                for chunk in reader:
                    # Modal returns text lines (with trailing newline preserved).
                    buffer.append(chunk)
                    if is_stderr:
                        print(chunk, end="", file=sys.stderr, flush=True)
                    else:
                        print(chunk, end="", flush=True)

                    # Also emit a structured streaming event for the web UI so it can
                    # render progress bars and logs as they happen, without waiting
                    # for the entire sandbox run to complete.
                    try:
                        emit_event(
                            "AGENT_STREAM",
                            {
                                "stream": "stderr" if is_stderr else "stdout",
                                "chunk": chunk,
                            },
                        )
                    except Exception as e:
                        # Structured events are best-effort only; don't break execution.
                        log_step("WARNING", f"Failed to emit AGENT_STREAM event: {e}")
            except Exception as e:
                # Don't crash the whole tool if streaming fails; just log.
                stream_name = "stderr" if is_stderr else "stdout"
                log_step("WARNING", f"Error while streaming {stream_name}: {e}")

        # Read stdout and stderr concurrently so training logs / progress bars
        # appear in real time regardless of which stream they use.
        stdout_thread = threading.Thread(
            target=_drain_stream, args=(proc.stdout, stdout_chunks, False), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_drain_stream, args=(proc.stderr, stderr_chunks, True), daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to finish.
        log_step("EXECUTION", "Waiting for process exit...")
        exit_code = proc.wait()

        # Make sure we've drained any remaining output.
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)

        log_step("EXECUTION", f"Process exited with code {exit_code}")

        stdout_str = "".join(stdout_chunks)
        stderr_str = "".join(stderr_chunks)

        return f"Exit Code: {exit_code}\nSTDOUT:\n{stdout_str}\nSTDERR:\n{stderr_str}"

    except Exception as e:
        log_step("ERROR", f"Sandbox Execution Failed: {str(e)}")
        return f"Sandbox Execution Failed: {str(e)}"


def _build_claude_tool_definition() -> dict:
    """Build the tool definition for Claude's format."""
    return {
        "name": "execute_in_sandbox",
        "description": (
            "Executes Python code inside a persistent Modal Sandbox. "
            "The sandbox has numpy, pandas, torch, scikit-learn, and matplotlib installed. "
            "Returns the exit code, stdout, and stderr from the execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute in the sandbox."
                }
            },
            "required": ["code"]
        }
    }


def _build_system_prompt(gpu_hint: str) -> str:
    """System-level instructions for the Gemini agent."""
    return f"""You are an autonomous research scientist.
Your job is to rigorously verify the user's hypothesis using experiments
run in a Python sandbox.

Tool:
- `execute_in_sandbox(code: str)`: Runs a Python script in a persistent Modal Sandbox.
  - Preinstalled: numpy, pandas, torch, scikit-learn, matplotlib.
  - Compute: Sandbox GPU request for this run: {gpu_hint}.
  - The code runs as a normal Python script; no need to import `modal`.

Working loop:
1. **Think before acting.** Plan your next step in natural language.
   We will show these thoughts in the CLI, so keep them understandable.
2. **Act with tools.** When you need computation, call `execute_in_sandbox`
   with a complete, self-contained script.
3. **Observe and update.** Interpret tool results and decide what to do next.
4. **Finish clearly.** When you have confidently verified or falsified
   the hypothesis, write a short natural-language conclusion and then a
   final line that contains only `[DONE]`.
"""


def run_experiment_loop(hypothesis: str, test_mode: bool = False, model: str = "gemini-3-pro-preview"):
    """Main agent loop using Gemini 3 Pro or Claude Opus 4.5 with thinking + manual tool calling."""
    gpu_hint = _selected_gpu or "CPU"

    print_panel(f"Hypothesis: {hypothesis}", "Starting Experiment", "bold green")
    log_step("START", f"Hypothesis: {hypothesis}")
    print_status(f"Sandbox GPU request: {gpu_hint}", "info")
    print_status(f"Model: {model}", "info")

    if test_mode:
        print_status("TEST MODE ENABLED: Using mock data and skipping LLM calls.", "bold yellow")
        import time
        
        # Mock Agent Loop
        
        # Step 1: Thinking
        thought = (
            "I need to verify this hypothesis using a Python script.\n"
            "I will create a synthetic dataset and run a simple regression model.\n"
            "Then I will analyze the coefficients to check the relationship."
        )
        print_panel(thought, "Agent Thinking", "thought")
        log_step("THOUGHT", thought)
        emit_event("AGENT_THOUGHT", {"thought": thought})
        time.sleep(1.5)
        
        # Step 2: Tool Call
        code = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "print('Generating synthetic data...')\n"
            "data = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100)})\n"
            "print('Data shape:', data.shape)\n"
            "print('Correlation:', data.corr().iloc[0,1])"
        )
        fn_name = "execute_in_sandbox"
        fn_args = {"code": code}
        
        print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
        log_step("TOOL_CALL", f"{fn_name}({fn_args})")
        emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})
        time.sleep(1)
        
        # Step 3: Tool Result
        result = (
            "Exit Code: 0\n"
            "STDOUT:\n"
            "Generating synthetic data...\n"
            "Data shape: (100, 2)\n"
            "Correlation: 0.042\n"
            "STDERR:\n"
        )
        print_panel(result, "Tool Result", "result")
        log_step("TOOL_RESULT", "Executed")
        emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})
        time.sleep(1.5)
        
        # Step 4: Analysis
        message = (
            "The correlation is very low, which suggests no strong linear relationship.\n"
            "However, since this is mock data, I will conclude based on the hypothesis."
        )
        print_panel(message, "Agent Message", "info")
        log_step("MODEL", message)
        time.sleep(1)
        
        # Step 5: Final Report
        print_status("Generating Final Report...", "bold green")
        final_report = (
            "## Experiment Report\n\n"
            "We tested the hypothesis: " + hypothesis + "\n\n"
            "### Methodology\n"
            "We ran a simulation using synthetic data.\n\n"
            "### Conclusion\n"
            "The hypothesis was tested in a mock environment.\n"
            "[DONE]"
        )
        print_panel(final_report, "Final Report", "bold green")
        return

    # Branch based on model selection
    if model == "claude-opus-4-5":
        _run_claude_experiment_loop(hypothesis, gpu_hint)
    else:
        _run_gemini_experiment_loop(hypothesis, gpu_hint)


def _run_claude_experiment_loop(hypothesis: str, gpu_hint: str):
    """Run the experiment loop using Claude Opus 4.5 with extended thinking."""
    print_status("Claude extended thinking enabled", "info")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system_prompt = _build_system_prompt(gpu_hint)
    tool_def = _build_claude_tool_definition()

    # Initial conversation with hypothesis
    messages = [
        {"role": "user", "content": f"Hypothesis: {hypothesis}"}
    ]

    max_steps = 10

    for step in range(1, max_steps + 1):
        print_status(f"Step {step}...", "dim")

        try:
            # Use streaming for Claude with extended thinking enabled
            # We need to track thinking blocks with their signatures for proper history
            thinking_blocks = []  # List of {"thinking": str, "signature": str}
            text_content = []
            tool_use_blocks = []

            with client.messages.stream(
                model="claude-opus-4-5-20251101",
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000
                },
                system=system_prompt,
                tools=[tool_def],
                messages=messages,
            ) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            if hasattr(event, 'content_block'):
                                block = event.content_block
                                if hasattr(block, 'type'):
                                    if block.type == 'thinking':
                                        thinking_blocks.append({"thinking": "", "signature": None})
                                    elif block.type == 'text':
                                        text_content.append("")
                                    elif block.type == 'tool_use':
                                        tool_use_blocks.append({
                                            "id": block.id,
                                            "name": block.name,
                                            "input": ""
                                        })
                        elif event.type == 'content_block_delta':
                            if hasattr(event, 'delta'):
                                delta = event.delta
                                if hasattr(delta, 'type'):
                                    if delta.type == 'thinking_delta' and hasattr(delta, 'thinking'):
                                        if thinking_blocks:
                                            thinking_blocks[-1]["thinking"] += delta.thinking
                                            emit_event("AGENT_THOUGHT_STREAM", {"chunk": delta.thinking})
                                    elif delta.type == 'text_delta' and hasattr(delta, 'text'):
                                        if text_content:
                                            text_content[-1] += delta.text
                                    elif delta.type == 'input_json_delta' and hasattr(delta, 'partial_json'):
                                        if tool_use_blocks:
                                            tool_use_blocks[-1]["input"] += delta.partial_json
                                    elif delta.type == 'signature_delta' and hasattr(delta, 'signature'):
                                        # Capture signature for thinking blocks
                                        if thinking_blocks:
                                            if thinking_blocks[-1]["signature"] is None:
                                                thinking_blocks[-1]["signature"] = ""
                                            thinking_blocks[-1]["signature"] += delta.signature

        except Exception as e:
            print_status(f"API Error: {e}", "error")
            logger.error(f"API Error: {e}")
            break

        # Process thinking content
        thinking_texts = [tb["thinking"] for tb in thinking_blocks if tb["thinking"]]
        if thinking_texts:
            joined_thinking = "\n\n".join(thinking_texts)
            if joined_thinking:
                print_panel(joined_thinking, "Agent Thinking", "thought")
                log_step("THOUGHT", joined_thinking)

        # Process text content
        if text_content:
            joined_text = "\n\n".join(t for t in text_content if t)
            if joined_text:
                print_panel(joined_text, "Agent Message", "info")
                log_step("MODEL", joined_text)

        # Check for completion
        combined_text = "\n".join(thinking_texts + text_content)
        if "[DONE]" in combined_text:
            print_status("Agent signaled completion.", "success")
            break

        # Build assistant message for history - include signature for thinking blocks
        assistant_content = []
        for tb in thinking_blocks:
            if tb["thinking"]:
                thinking_block = {"type": "thinking", "thinking": tb["thinking"]}
                if tb["signature"]:
                    thinking_block["signature"] = tb["signature"]
                assistant_content.append(thinking_block)
        for t in text_content:
            if t:
                assistant_content.append({"type": "text", "text": t})

        # Process tool calls
        if not tool_use_blocks:
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            print_status(
                "No tool calls in this step; assuming experiment is complete.", "info"
            )
            break

        # Execute tool calls
        tool_results = []
        for tool_block in tool_use_blocks:
            fn_name = tool_block["name"]
            try:
                fn_args = json.loads(tool_block["input"]) if tool_block["input"] else {}
            except json.JSONDecodeError:
                fn_args = {}

            print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
            log_step("TOOL_CALL", f"{fn_name}({fn_args})")
            emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})

            # Add tool_use to assistant content
            assistant_content.append({
                "type": "tool_use",
                "id": tool_block["id"],
                "name": fn_name,
                "input": fn_args
            })

            if fn_name == "execute_in_sandbox":
                result = execute_in_sandbox(**fn_args)
            else:
                result = (
                    f"Unsupported tool '{fn_name}'. "
                    "Only 'execute_in_sandbox' is available."
                )

            # Truncate long outputs
            if isinstance(result, str) and len(result) > 20000:
                result = (
                    result[:10000]
                    + "\n...[TRUNCATED]...\n"
                    + result[-10000:]
                )

            print_panel(result, "Tool Result", "result")
            log_step("TOOL_RESULT", "Executed")
            emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block["id"],
                "content": result
            })

        # Add assistant message and tool results to history
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    # Final report generation
    try:
        print_status("Generating Final Report...", "bold green")
        messages.append({
            "role": "user",
            "content": (
                "Generate a concise, information-dense report that explains "
                "how you tested the hypothesis, what you observed, and your "
                "final conclusion."
            )
        })

        final_thinking = []
        final_text = []

        with client.messages.stream(
            model="claude-opus-4-5-20251101",
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            },
            system=system_prompt,
            messages=messages,
        ) as stream:
            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'content_block_start':
                        if hasattr(event, 'content_block'):
                            block = event.content_block
                            if hasattr(block, 'type'):
                                if block.type == 'thinking':
                                    final_thinking.append("")
                                elif block.type == 'text':
                                    final_text.append("")
                    elif event.type == 'content_block_delta':
                        if hasattr(event, 'delta'):
                            delta = event.delta
                            if hasattr(delta, 'type'):
                                if delta.type == 'thinking_delta' and hasattr(delta, 'thinking'):
                                    if final_thinking:
                                        final_thinking[-1] += delta.thinking
                                        emit_event("AGENT_THOUGHT_STREAM", {"chunk": delta.thinking})
                                elif delta.type == 'text_delta' and hasattr(delta, 'text'):
                                    if final_text:
                                        final_text[-1] += delta.text

        final_report = "\n\n".join(t for t in final_text if t)
        print_panel(final_report, "Final Report", "bold green")
    finally:
        _close_shared_sandbox()


def _run_gemini_experiment_loop(hypothesis: str, gpu_hint: str):
    """Run the experiment loop using Gemini 3 Pro with thinking mode."""
    print_status("Gemini thinking: HIGH (thought summaries visible)", "info")

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    # Expose the sandbox executor as a tool.
    tools = [execute_in_sandbox]
    system_prompt = _build_system_prompt(gpu_hint)

    # Initial conversation: just the hypothesis as a user message.
    history: List[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"Hypothesis: {hypothesis}")],
        )
    ]

    max_steps = 10

    for step in range(1, max_steps + 1):
        print_status(f"Step {step}...", "dim")

        try:
            # Stream the model's response so we can surface thinking and tool calls in real time.
            response_stream = client.models.generate_content_stream(
                model="gemini-3-pro-preview",
                contents=history,
                config=_build_generation_config(
                    tools=tools,
                    system_instruction=system_prompt,
                    disable_autofc=True,  # manual tool loop
                ),
            )
        except Exception as e:
            print_status(f"API Error: {e}", "error")
            logger.error(f"API Error: {e}")
            break

        # Accumulate full response for history and logic
        accumulated_parts = []

        # Track chunks
        for chunk in response_stream:
             if not chunk.candidates:
                 continue
             
             candidate = chunk.candidates[0]
             if not candidate.content or not candidate.content.parts:
                 continue
             
             for part in candidate.content.parts:
                 # 1. Streaming thoughts
                 if getattr(part, "thought", False) and part.text:
                     emit_event("AGENT_THOUGHT_STREAM", {"chunk": part.text})
                 
                 # Add to accumulator
                 accumulated_parts.append(part)

        # Reconstruct the full Content object (merge logic similar to orchestrator)
        merged_parts = []
        current_text_part = None
        current_thought_part = None
        
        for part in accumulated_parts:
            # Handle Function Calls
            if part.function_call:
                if current_text_part:
                    merged_parts.append(current_text_part)
                    current_text_part = None
                if current_thought_part:
                    merged_parts.append(current_thought_part)
                    current_thought_part = None
                merged_parts.append(part)
                continue
                
            # Handle Thoughts
            if getattr(part, "thought", False):
                if current_text_part:
                    merged_parts.append(current_text_part)
                    current_text_part = None
                
                if current_thought_part:
                    current_thought_part.text += part.text
                else:
                    current_thought_part = part
                continue

            # Handle Text
            if part.text:
                if current_thought_part:
                    merged_parts.append(current_thought_part)
                    current_thought_part = None
                    
                if current_text_part:
                    current_text_part.text += part.text
                else:
                    current_text_part = part
                continue
        
        if current_text_part:
            merged_parts.append(current_text_part)
        if current_thought_part:
            merged_parts.append(current_thought_part)

        if not merged_parts:
            print_status("Empty content from model.", "warning")
            break
            
        model_content = types.Content(role="model", parts=merged_parts)

        # IMPORTANT: append the full model message (including thought signatures
        # and function call parts) so the SDK can preserve reasoning state.
        history.append(model_content)

        thoughts: List[str] = []
        messages: List[str] = []
        function_calls = []

        for part in model_content.parts:
            # Thought summaries from thinking mode.
            if getattr(part, "thought", False) and part.text:
                thoughts.append(part.text)

            # Function/tool call parts.
            if part.function_call:
                function_calls.append(part.function_call)

            # Regular assistant text (exclude thought parts so we don't double-print).
            if part.text and not getattr(part, "thought", False):
                messages.append(part.text)

        # 1. Show reasoning before any action.
        if thoughts:
            joined_thoughts = "\n\n".join(thoughts)
            print_panel(joined_thoughts, "Agent Thinking", "thought")
            log_step("THOUGHT", joined_thoughts)

        # 2. Show natural-language messages (plans, explanations, etc.).
        if messages:
            joined_messages = "\n\n".join(messages)
            print_panel(joined_messages, "Agent Message", "info")
            log_step("MODEL", joined_messages)

        combined_text = "\n".join(thoughts + messages)
        if "[DONE]" in combined_text:
            print_status("Agent signaled completion.", "success")
            break

        # If the model didn't call any tools this turn, assume we're done.
        if not function_calls:
            print_status(
                "No tool calls in this step; assuming experiment is complete.", "info"
            )
            break

        # 3. Execute requested tools (currently just execute_in_sandbox).
        for fn_call in function_calls:
            fn_name = fn_call.name
            fn_args = dict(fn_call.args or {})

            print_panel(f"{fn_name}({fn_args})", "Tool Call", "code")
            log_step("TOOL_CALL", f"{fn_name}({fn_args})")
            emit_event("AGENT_TOOL", {"tool": fn_name, "args": fn_args})

            if fn_name == "execute_in_sandbox":
                result = execute_in_sandbox(**fn_args)
            else:
                result = (
                    f"Unsupported tool '{fn_name}'. "
                    "Only 'execute_in_sandbox' is available."
                )

            # Truncate long outputs to keep console readable.
            if isinstance(result, str) and len(result) > 20000:
                result = (
                    result[:10000]
                    + "\n...[TRUNCATED]...\n"
                    + result[-10000:]
                )

            print_panel(result, "Tool Result", "result")
            log_step("TOOL_RESULT", "Executed")
            emit_event("AGENT_TOOL_RESULT", {"tool": fn_name, "result": result})

            # Feed the tool response back as a TOOL message with a functionResponse part.
            history.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"result": result},
                        )
                    ],
                )
            )

    # Final report generation.
    try:
        print_status("Generating Final Report...", "bold green")
        history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Generate a concise, information-dense report that explains "
                            "how you tested the hypothesis, what you observed, and your "
                            "final conclusion."
                        )
                    )
                ],
            )
        )

        final_response_stream = client.models.generate_content_stream(
            model="gemini-3-pro-preview",
            contents=history,
            # Still use thinking so the model can reason about its own trace,
            # but tools are not needed here.
            config=_build_generation_config(
                tools=None,
                system_instruction=system_prompt,
                disable_autofc=True,
            ),
        )

        final_parts = []
        for chunk in final_response_stream:
            if chunk.candidates and chunk.candidates[0].content:
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False) and part.text:
                        emit_event("AGENT_THOUGHT_STREAM", {"chunk": part.text})
                    final_parts.append(part)

        # Basic merge for final text extraction
        final_text = ""
        for part in final_parts:
            if part.text and not getattr(part, "thought", False):
                final_text += part.text
        
        print_panel(final_text, "Final Report", "bold green")
    finally:
        _close_shared_sandbox()

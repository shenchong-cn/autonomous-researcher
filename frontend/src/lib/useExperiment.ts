import { useState, useRef } from "react";
import { streamExperiment, LogEvent, summarizeAgent, ChartSpec } from "./api";

export type StepType = "thought" | "code" | "result" | "text";

export interface ExperimentStep {
    id: string;
    type: StepType;
    content: string;
    metadata?: Record<string, any>;
    timestamp: number;
}

export interface AgentInsight {
    id: string;
    summary: string;
    chart?: ChartSpec | null;
    timestamp: number;
}

export interface AgentState {
    id: string;
    status: "idle" | "running" | "completed" | "failed";
    hypothesis?: string;
    gpu?: string;
    logs: string[];
    exitCode?: number;
    steps: ExperimentStep[];
    insights: AgentInsight[];
}

export type TimelineItem =
    | { type: "thought"; content: string; timestamp: number }
    | { type: "agents"; agentIds: string[]; timestamp: number }
    | { type: "paper"; content: string; charts?: ChartSpec[]; timestamp: number };

export interface OrchestratorState {
    status: "idle" | "planning" | "running" | "completed";
    thoughts: string[];
    plan: string[];
    timeline: TimelineItem[];
}

export function useExperiment() {
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState<LogEvent[]>([]);
    const [agents, setAgents] = useState<Record<string, AgentState>>({});
    const [orchestrator, setOrchestrator] = useState<OrchestratorState>({
        status: "idle",
        thoughts: [],
        plan: [],
        timeline: [],
    });
    const [error, setError] = useState<string | null>(null);

    // Keep track of the latest agents state to update it functionally
    const agentsRef = useRef<Record<string, AgentState>>({});
    const summaryTimersRef = useRef<Record<string, NodeJS.Timeout | null>>({});
    const summaryInflightRef = useRef<Record<string, boolean>>({});

    const updateAgent = (id: string, update: Partial<AgentState>) => {
        setAgents((prev) => {
            const current = prev[id] || { id, status: "idle", logs: [], steps: [], insights: [] };
            const next = {
                ...prev,
                [id]: { ...current, ...update },
            };
            agentsRef.current = next;
            return next;
        });
    };

    const addAgentStep = (id: string, step: Omit<ExperimentStep, "id" | "timestamp">) => {
        setAgents((prev) => {
            const current = prev[id] || { id, status: "idle", logs: [], steps: [], insights: [] };
            const newStep: ExperimentStep = {
                ...step,
                id: Math.random().toString(36).substring(7),
                timestamp: Date.now(),
            };

            const next = {
                ...prev,
                [id]: {
                    ...current,
                    steps: [...current.steps, newStep],
                },
            };
            agentsRef.current = next;
            return next;
        });
    };

    const appendToLatestAgentStep = (id: string, type: StepType, chunk: string) => {
        setAgents((prev) => {
            const current = prev[id];
            if (!current) return prev;

            const steps = [...current.steps];

            const newStep: ExperimentStep = {
                id: Math.random().toString(36).substring(7),
                type,
                content: chunk,
                timestamp: Date.now(),
            };

            // If there are no steps yet, create the first one so we can stream into it.
            if (steps.length === 0) {
                const next = {
                    ...prev,
                    [id]: {
                        ...current,
                        steps: [newStep],
                    },
                };
                agentsRef.current = next;
                return next;
            }

            const lastStep = steps[steps.length - 1];

            // If the last step matches the type, append to it.
            if (lastStep.type === type) {
                steps[steps.length - 1] = {
                    ...lastStep,
                    content: lastStep.content + chunk,
                };
            } else {
                // Fallback: create new step if types mismatch
                steps.push(newStep);
            }

            const next = {
                ...prev,
                [id]: { ...current, steps },
            };
            agentsRef.current = next;
            return next;
        });
    };

    const addAgentInsight = (id: string, insight: Omit<AgentInsight, "id" | "timestamp"> & { id?: string; timestamp?: number }) => {
        setAgents((prev) => {
            const current = prev[id];
            if (!current) return prev;

            const nextInsight: AgentInsight = {
                id: insight.id || Math.random().toString(36).substring(7),
                timestamp: insight.timestamp || Date.now(),
                summary: insight.summary,
                chart: insight.chart,
            };

            const next = {
                ...prev,
                [id]: {
                    ...current,
                    insights: [...(current.insights || []), nextInsight],
                },
            };

            agentsRef.current = next;
            return next;
        });
    };

    const appendToLatestOrchestratorStep = (type: "thought" | "text", chunk: string) => {
        setOrchestrator((prev) => {
            const timeline = [...prev.timeline];
            const lastItem = timeline[timeline.length - 1];
            
            // Check if we can append to the last item
            if (lastItem && lastItem.type === type) {
                timeline[timeline.length - 1] = {
                    ...lastItem,
                    content: lastItem.content + chunk
                };
                
                // Also update thoughts array if it's a thought
                let thoughts = prev.thoughts;
                if (type === "thought") {
                    thoughts = [...prev.thoughts];
                    if (thoughts.length > 0) {
                        thoughts[thoughts.length - 1] = thoughts[thoughts.length - 1] + chunk;
                    } else {
                        thoughts.push(chunk);
                    }
                }
                
                return { ...prev, timeline, thoughts };
            } else {
                // Create new item
                const newItem: TimelineItem = {
                    type: type as any, // 'text' isn't in TimelineItem type explicitly? let's check
                    content: chunk,
                    timestamp: Date.now()
                };
                
                // TimelineItem is: thought | agents | paper. 
                // If 'text' is meant to be something else, we might need to adjust.
                // But 'thought' is definitely supported.
                if (type !== "thought") {
                    // For now orchestrator only really supports 'thought' and 'paper' and 'agents' in timeline
                    // If we have general text messages, maybe treat as thoughts or ignore?
                    // Orchestrator messages are usually shown as 'info' panels in CLI.
                    // In frontend timeline, we map 'thought' to the Orchestrator block.
                    // Let's assume 'thought' for now.
                    return prev;
                }

                return {
                    ...prev,
                    timeline: [...timeline, newItem],
                    thoughts: [...prev.thoughts, chunk]
                };
            }
        });
    };

    const runAgentSummary = async (agentId: string) => {
        // Clear pending timer marker
        summaryTimersRef.current[agentId] = null;

        if (summaryInflightRef.current[agentId]) return;

        const agent = agentsRef.current[agentId];
        if (!agent || agent.steps.length === 0) return;

        summaryInflightRef.current[agentId] = true;

        try {
            const recentSteps = agent.steps.slice(-5).map((step) => ({
                type: step.type,
                content: step.content.slice(-2000),
            }));

            const resp = await summarizeAgent({
                agent_id: agentId,
                history: recentSteps,
            });

            addAgentInsight(agentId, {
                summary: resp.summary,
                chart: resp.chart,
            });
        } catch (err) {
            console.warn("Failed to summarize agent", agentId, err);
        } finally {
            summaryInflightRef.current[agentId] = false;
        }
    };

    const scheduleAgentSummary = (agentId: string) => {
        if (!agentId) return;

        // debounce to wait for the end of a thought stream
        const timers = summaryTimersRef.current;
        if (timers[agentId]) {
            clearTimeout(timers[agentId]!);
        }

        timers[agentId] = setTimeout(() => runAgentSummary(agentId), 900);
    };

    const startExperiment = async (
        mode: "single" | "orchestrator",
        config: {
            task: string;
            gpu?: string;
            model?: string;
            num_agents?: number;
            max_rounds?: number;
            max_parallel?: number;
            test_mode?: boolean;
        }
    ) => {
        // Reset any pending sidebar summary timers between runs
        Object.values(summaryTimersRef.current).forEach((timer) => timer && clearTimeout(timer));
        summaryTimersRef.current = {};
        summaryInflightRef.current = {};

        setIsRunning(true);
        setError(null);
        setAgents({});
        setOrchestrator({ thoughts: [], plan: [], timeline: [], status: "running" });

        try {
            const endpoint = mode === "single"
                ? "/api/experiments/single/stream"
                : "/api/experiments/orchestrator/stream";

            const payload = mode === "single"
                ? { task: config.task, gpu: config.gpu, model: config.model, test_mode: config.test_mode }
                : {
                    task: config.task,
                    gpu: config.gpu,
                    model: config.model,
                    num_agents: config.num_agents || 3,
                    max_rounds: config.max_rounds || 3,
                    max_parallel: config.max_parallel || 2,
                    test_mode: config.test_mode
                };

            await streamExperiment(
                endpoint,
                payload,
                (event) => {
                    setLogs((prev) => [...prev, event]);

                    if (event.type === "line" && event.plain) {
                        // Check for ::EVENT:: marker
                        const eventIndex = event.plain.indexOf("::EVENT::");
                        if (eventIndex !== -1) {
                            try {
                                const jsonStr = event.plain.substring(eventIndex + "::EVENT::".length);
                                const payload = JSON.parse(jsonStr);

                                // Try to extract Agent ID from prefix if present: "[Agent 1] ::EVENT::..."
                                let inferredAgentId: string | undefined;
                                const prefix = event.plain.substring(0, eventIndex);
                                const match = prefix.match(/\[Agent (\d+)\]/);
                                if (match) {
                                    inferredAgentId = match[1];
                                }

                                handleStructuredEvent(payload, inferredAgentId);
                            } catch (e) {
                                console.warn("Failed to parse structured event:", e);
                            }
                        }
                    }
                },
                (err) => {
                    setError(err.message);
                    setIsRunning(false);
                },
                () => {
                    setIsRunning(false);
                }
            );
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to start experiment");
            setIsRunning(false);
        }
    };

    const handleStructuredEvent = (event: any, inferredAgentId?: string) => {
        const { type, data } = event;

        switch (type) {
            case "AGENT_START":
                updateAgent(data.agent_id, {
                    status: "running",
                    hypothesis: data.hypothesis,
                    gpu: data.gpu,
                });

                // Add agent to the current 'agents' timeline item if it exists, or create new one
                setOrchestrator((prev) => {
                    const lastItem = prev.timeline[prev.timeline.length - 1];
                    if (lastItem && lastItem.type === "agents") {
                        // Check if agent is already in the list to avoid dupes
                        if (lastItem.agentIds.includes(data.agent_id)) {
                            return prev;
                        }
                        // Update the last item in place (immutably)
                        const newTimeline = [...prev.timeline];
                        newTimeline[newTimeline.length - 1] = {
                            ...lastItem,
                            agentIds: [...lastItem.agentIds, data.agent_id]
                        };
                        return { ...prev, timeline: newTimeline };
                    } else {
                        // Create new agents group
                        return {
                            ...prev,
                            timeline: [
                                ...prev.timeline,
                                { type: "agents", agentIds: [data.agent_id], timestamp: Date.now() }
                            ]
                        };
                    }
                });
                break;

            case "AGENT_THOUGHT":
                if (inferredAgentId) {
                    addAgentStep(inferredAgentId, {
                        type: "thought",
                        content: data.thought,
                    });
                    scheduleAgentSummary(inferredAgentId);
                }
                break;

            case "AGENT_THOUGHT_STREAM":
                if (inferredAgentId && typeof data?.chunk === "string") {
                    appendToLatestAgentStep(inferredAgentId, "thought", data.chunk);
                    scheduleAgentSummary(inferredAgentId);
                }
                break;

            case "AGENT_TOOL":
                if (inferredAgentId) {
                    addAgentStep(inferredAgentId, {
                        type: "code",
                        content: `${data.tool}(${JSON.stringify(data.args, null, 2)})`,
                        metadata: { tool: data.tool, args: data.args },
                    });
                }
                break;

            case "AGENT_TOOL_RESULT":
                if (inferredAgentId) {
                    // Update the latest step if it's a result block (from streaming),
                    // otherwise create a new one.
                    setAgents((prev) => {
                        const current = prev[inferredAgentId];
                        if (!current) return prev;

                        const steps = [...current.steps];
                        const lastStep = steps[steps.length - 1];

                        if (lastStep && lastStep.type === "result") {
                            // Update existing result block with the final full content
                            steps[steps.length - 1] = {
                                ...lastStep,
                                content: data.result,
                                metadata: { ...lastStep.metadata, tool: data.tool }
                            };
                        } else {
                            // Create new result block
                            steps.push({
                                id: Math.random().toString(36).substring(7),
                                type: "result",
                                content: data.result,
                                metadata: { tool: data.tool },
                                timestamp: Date.now(),
                            });
                        }

                        return {
                            ...prev,
                            [inferredAgentId]: { ...current, steps }
                        };
                    });

                    scheduleAgentSummary(inferredAgentId);
                }
                break;

            case "AGENT_STREAM":
                if (inferredAgentId && typeof data?.chunk === "string") {
                    // Stream incremental sandbox output into the latest result cell.
                    // NotebookCell already handles carriage returns (\r) to render
                    // tqdm-style progress bars cleanly.
                    appendToLatestAgentStep(inferredAgentId, "result", data.chunk);
                }
                break;

            case "AGENT_COMPLETE":
                updateAgent(data.agent_id, {
                    status: "completed",
                    exitCode: data.exit_code,
                });
                scheduleAgentSummary(data.agent_id);
                break;

            case "ORCH_THOUGHT":
                setOrchestrator((prev) => ({
                    ...prev,
                    thoughts: [...prev.thoughts, data.thought],
                    timeline: [
                        ...prev.timeline,
                        { type: "thought", content: data.thought, timestamp: Date.now() }
                    ]
                }));
                break;

            case "ORCH_THOUGHT_STREAM":
                if (typeof data?.chunk === "string") {
                    appendToLatestOrchestratorStep("thought", data.chunk);
                }
                break;

            case "ORCH_PAPER":
                // Capture any charts that agents have produced so we can surface them alongside the paper.
                const charts: ChartSpec[] = Object.values(agentsRef.current)
                    .flatMap((agent) => (agent.insights || []).map((insight) => insight.chart))
                    .filter(Boolean) as ChartSpec[];

                // Deduplicate loosely by title + type + first series name + length of labels.
                const seen = new Set<string>();
                const uniqueCharts: ChartSpec[] = [];
                for (const chart of charts) {
                    const key = [
                        chart.title || "untitled",
                        chart.type,
                        chart.series?.[0]?.name || "series",
                        chart.labels?.length || 0,
                        chart.series?.[0]?.values?.length || 0,
                    ].join("|");
                    if (seen.has(key)) continue;
                    seen.add(key);
                    uniqueCharts.push(chart);
                }

                setOrchestrator((prev) => ({
                    ...prev,
                    timeline: [
                        ...prev.timeline,
                        {
                            type: "paper",
                            content: data.content,
                            charts: uniqueCharts.slice(0, 6), // keep it concise
                            timestamp: Date.now(),
                        },
                    ],
                }));
                break;

            case "ORCH_TOOL":
                // We could also track orchestrator steps if we wanted a notebook for it
                break;
        }
    };

    // We need a way to parse the agent ID from the line if it exists.
    // The orchestrator prefixes: `[Agent {id}] `

    const clearError = () => setError(null);

    return {
        isRunning,
        logs,
        agents,
        orchestrator,
        error,
        startExperiment,
        clearError,
    };
}

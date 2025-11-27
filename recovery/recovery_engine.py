"""
Recovery engine for orchestrator restart and hang detection.
"""
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional

# Add project root to Python path for standalone execution
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from recovery.models import (
    OrchestratorState,
    OrchestratorStatus,
    DEFAULT_CONFIG
)
from recovery.state_manager import StateManager
from recovery.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class RecoveryEngine:
    """Handles orchestrator recovery and hang detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DEFAULT_CONFIG
        self.state_manager = StateManager(config)
        self.checkpoint_manager = CheckpointManager(config)
        self._running_processes: Dict[str, subprocess.Popen] = {}

    def detect_hang(self, experiment_id: str) -> bool:
        """
        Detect if an experiment is hung.

        Args:
            experiment_id: ID of the experiment to check

        Returns:
            True if hung, False otherwise
        """
        try:
            state = self.state_manager.load_state(experiment_id)
            if not state:
                return False

            now = datetime.now(timezone.utc)
            hang_timeout = self.config["recovery"]["hang_timeout_seconds"]
            experiment_timeout = self.config["recovery"]["experiment_timeout_seconds"]

            # Check if overall experiment is hung
            time_since_update = (now - state.updated_at).total_seconds()
            if time_since_update > hang_timeout:
                logger.warning(f"Experiment {experiment_id} appears hung (inactive for {time_since_update}s)")
                return True

            # Check if any individual experiment is hung
            for active_exp_id in state.active_experiments:
                # Try to get start time of active experiment from metadata
                exp_start_time = state.metadata.get(f"{active_exp_id}_start_time")
                if exp_start_time:
                    start_time = datetime.fromisoformat(exp_start_time)
                    time_since_start = (now - start_time).total_seconds()
                    if time_since_start > experiment_timeout:
                        logger.warning(f"Active experiment {active_exp_id} appears hung (running for {time_since_start}s)")
                        return True

            return False

        except Exception as e:
            logger.error(f"Error detecting hang for {experiment_id}: {e}")
            return False

    def graceful_restart(self, experiment_id: str, checkpoint_id: Optional[str] = None) -> bool:
        """
        Gracefully restart an experiment from a checkpoint.

        Args:
            experiment_id: ID of the experiment to restart
            checkpoint_id: Optional specific checkpoint to use

        Returns:
            True if restart initiated successfully, False otherwise
        """
        try:
            # Load state to restore
            if checkpoint_id:
                state = self.checkpoint_manager.restore_from_checkpoint(experiment_id, checkpoint_id)
            else:
                state = self.state_manager.load_state(experiment_id)

            if not state:
                logger.error(f"No state found for experiment {experiment_id}")
                return False

            # Terminate any running processes
            self._terminate_running_processes(experiment_id)

            # Update status to running and force Claude API
            state.model = "claude-opus-4-5"
            state.status = OrchestratorStatus.RUNNING
            state.updated_at = datetime.now(timezone.utc)
            state.metadata["recovery_attempts"] = state.metadata.get("recovery_attempts", 0) + 1
            state.metadata["last_recovery_time"] = datetime.now(timezone.utc).isoformat()

            # Save updated state
            if not self.state_manager.save_state(state):
                logger.error("Failed to save updated state before restart")
                return False

            # Start new orchestrator process
            success = self._start_orchestrator_process(state)

            if success:
                logger.info(f"Successfully restarted experiment {experiment_id}")
            else:
                logger.error(f"Failed to restart experiment {experiment_id}")

            return success

        except Exception as e:
            logger.error(f"Error during graceful restart of {experiment_id}: {e}")
            return False

    def resume_execution(self, experiment_id: str, checkpoint_id: Optional[str] = None) -> bool:
        """
        Resume execution of a paused/hung experiment.

        Args:
            experiment_id: ID of the experiment to resume
            checkpoint_id: Optional specific checkpoint to resume from

        Returns:
            True if resume initiated successfully, False otherwise
        """
        try:
            state = self.state_manager.load_state(experiment_id)
            if not state:
                logger.error(f"No state found for experiment {experiment_id}")
                return False

            # Check if we can resume
            if not self.state_manager.can_resume(experiment_id):
                logger.error(f"Experiment {experiment_id} cannot be resumed")
                return False

            # Restore from checkpoint if specified
            if checkpoint_id:
                restored_state = self.checkpoint_manager.restore_from_checkpoint(experiment_id, checkpoint_id)
                if restored_state:
                    restored_state.status = OrchestratorStatus.RUNNING
                    restored_state.updated_at = datetime.now(timezone.utc)
                    if not self.state_manager.save_state(restored_state):
                        logger.error("Failed to save restored state")
                        return False
                    state = restored_state

            # Update state - force use Claude API
            state.model = "claude-opus-4-5"
            state.status = OrchestratorStatus.RUNNING
            state.updated_at = datetime.now(timezone.utc)

            # Save state
            if not self.state_manager.save_state(state):
                logger.error("Failed to save state for resume")
                return False

            # Start orchestrator process
            return self._start_orchestrator_process(state)

        except Exception as e:
            logger.error(f"Error resuming execution of {experiment_id}: {e}")
            return False

    def auto_recovery_check(self) -> List[str]:
        """
        Perform automatic recovery check on all experiments.

        Returns:
            List of experiment IDs that were auto-recovered
        """
        recovered = []
        resumable_experiments = self.state_manager.get_resumable_experiments()

        for exp_summary in resumable_experiments:
            experiment_id = exp_summary["experiment_id"]

            # Check if experiment is hung
            if self.detect_hang(experiment_id):
                # Check recovery attempts limit
                state = self.state_manager.load_state(experiment_id)
                if state:
                    max_attempts = self.config["recovery"]["max_recovery_attempts"]
                    attempts = state.metadata.get("recovery_attempts", 0)

                    if attempts < max_attempts:
                        logger.info(f"Attempting auto-recovery for experiment {experiment_id}")
                        if self.graceful_restart(experiment_id):
                            recovered.append(experiment_id)
                    else:
                        logger.warning(f"Experiment {experiment_id} exceeded max recovery attempts")

        return recovered

    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Status dictionary or None if not found
        """
        state = self.state_manager.load_state(experiment_id)
        if not state:
            return None

        now = datetime.now(timezone.utc)
        time_since_update = (now - state.updated_at).total_seconds()
        is_hung = self.detect_hang(experiment_id)

        return {
            "experiment_id": state.experiment_id,
            "status": state.status.value,
            "current_step": state.current_step,
            "max_steps": state.max_steps,
            "progress": {
                "current_step": state.current_step,
                "max_steps": state.max_steps,
                "completed_experiments": len(state.completed_experiments),
                "total_experiments": state.current_step * 3 if state.config else 0,
                "percentage": (state.current_step / state.max_steps * 100) if state.max_steps > 0 else 0,
            },
            "can_resume": self.state_manager.can_resume(experiment_id),
            "last_activity": state.updated_at.isoformat(),
            "time_since_update_seconds": time_since_update,
            "is_hung": is_hung,
            "active_experiments": state.active_experiments,
            "completed_experiments": len(state.completed_experiments),
            "available_checkpoints": [cp.to_dict() for cp in state.checkpoints],
            "recovery_attempts": state.metadata.get("recovery_attempts", 0),
            "last_recovery_time": state.metadata.get("last_recovery_time"),
        }

    def _start_orchestrator_process(self, state: OrchestratorState) -> bool:
        """
        Start a new orchestrator process with the given state.

        Args:
            state: State to resume from

        Returns:
            True if process started successfully, False otherwise
        """
        try:
            # Build command to restart orchestrator
            # Force use Claude API to avoid geographic restrictions
            cmd = [
                "python", "main.py",
                state.research_task,
                "--mode", "orchestrator",
                "--model", "claude-opus-4-5",
                "--resume-from", state.experiment_id
            ]

            if state.config:
                cmd.extend([
                    "--num-agents", str(state.config.num_agents),
                    "--max-rounds", str(state.config.max_rounds),
                    "--max-parallel", str(state.config.max_parallel)
                ])
                if state.config.gpu:
                    cmd.extend(["--gpu", state.config.gpu])
                if state.config.test_mode:
                    cmd.append("--test-mode")

            print(f"[DEBUG] Starting orchestrator with command: {' '.join(cmd)}", file=sys.stderr)
            print(f"[DEBUG] Working directory: {Path.cwd()}", file=sys.stderr)

            # Start process
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=Path.cwd()
            )

            # Start output reader threads
            def _reader(stream, is_err: bool) -> None:
                """Read from a subprocess stream and output to console."""
                prefix = f"[Resumed {state.experiment_id}] "
                target = sys.stderr if is_err else sys.stdout
                try:
                    for line in stream:
                        if line:  # Only process non-empty lines
                            target.write(f"{prefix}{line}")
                            target.flush()
                except Exception as e:
                    logger.error(f"Error reading {'stderr' if is_err else 'stdout'}: {e}")

            # Start threads to read stdout and stderr
            import threading
            t_out = threading.Thread(
                target=_reader, args=(proc.stdout, False), daemon=True
            )
            t_err = threading.Thread(
                target=_reader, args=(proc.stderr, True), daemon=True
            )

            t_out.start()
            t_err.start()

            self._running_processes[state.experiment_id] = proc

            print(f"[DEBUG] Started orchestrator process for {state.experiment_id} (PID: {proc.pid})", file=sys.stderr)
            logger.info(f"Started orchestrator process for {state.experiment_id} (PID: {proc.pid})")
            return True

        except Exception as e:
            print(f"[DEBUG] Failed to start orchestrator process: {e}", file=sys.stderr)
            logger.error(f"Failed to start orchestrator process: {e}")
            return False

    def _terminate_running_processes(self, experiment_id: str) -> None:
        """
        Terminate any running processes for the experiment.

        Args:
            experiment_id: ID of the experiment
        """
        if experiment_id in self._running_processes:
            proc = self._running_processes[experiment_id]
            try:
                if proc.poll() is None:  # Process is still running
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    logger.info(f"Terminated process for experiment {experiment_id}")
            except Exception as e:
                logger.error(f"Error terminating process for {experiment_id}: {e}")
            finally:
                del self._running_processes[experiment_id]

    def cleanup_on_shutdown(self) -> None:
        """Clean up all running processes on shutdown."""
        for experiment_id in list(self._running_processes.keys()):
            self._terminate_running_processes(experiment_id)

    def validate_state(self, state: OrchestratorState) -> bool:
        """
        Validate the integrity of a state object.

        Args:
            state: State to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if not state.experiment_id or not state.research_task:
                logger.error("Missing required fields in state")
                return False

            # Check step consistency
            if state.current_step < 0 or state.max_steps <= 0:
                logger.error("Invalid step values")
                return False

            if state.current_step > state.max_steps:
                logger.error("Current step exceeds max steps")
                return False

            # Check experiment results consistency
            for exp_result in state.completed_experiments:
                if not exp_result.experiment_id or not exp_result.hypothesis:
                    logger.error("Invalid experiment result")
                    return False

            # Check checkpoints consistency
            for checkpoint in state.checkpoints:
                if not checkpoint.id or checkpoint.step < 0:
                    logger.error("Invalid checkpoint")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating state: {e}")
            return False
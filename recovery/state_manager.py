"""
State management for orchestrator recovery.
"""
import json
import shutil
import threading
import sys
from datetime import datetime, timezone
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional

# Add project root to Python path for standalone execution
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from recovery.models import (
    OrchestratorState,
    OrchestratorStatus,
    ExperimentResult,
    CheckpointInfo,
    get_experiment_dir,
    ensure_state_dirs,
    DEFAULT_CONFIG
)

logger = logging.getLogger(__name__)


class StateManager:
    """Manages orchestrator state persistence and recovery."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DEFAULT_CONFIG
        self._lock = threading.RLock()
        ensure_state_dirs()

    def save_state(self, state: OrchestratorState) -> bool:
        """
        Save orchestrator state to persistent storage.

        Args:
            state: The orchestrator state to save

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                # Update timestamp
                state.updated_at = datetime.now(timezone.utc)

                # Create experiment directory
                exp_dir = get_experiment_dir(state.experiment_id)
                exp_dir.mkdir(parents=True, exist_ok=True)

                # Save main state file
                state_file = exp_dir / "state.json"
                with open(state_file, 'w', encoding='utf-8') as f:
                    f.write(state.to_json())

                # Create backup if enabled
                if self.config["persistence"]["backup_enabled"]:
                    self._create_backup(state_file, state.experiment_id)

                logger.info(f"State saved for experiment {state.experiment_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def load_state(self, experiment_id: str) -> Optional[OrchestratorState]:
        """
        Load orchestrator state from persistent storage.

        Args:
            experiment_id: ID of the experiment to load

        Returns:
            Loaded state or None if not found
        """
        try:
            with self._lock:
                state_file = get_experiment_dir(experiment_id) / "state.json"

                if not state_file.exists():
                    logger.info(f"No state file found for experiment {experiment_id}")
                    return None

                with open(state_file, 'r', encoding='utf-8') as f:
                    state_json = f.read()

                state = OrchestratorState.from_json(state_json)
                logger.info(f"State loaded for experiment {experiment_id}")
                return state

        except Exception as e:
            logger.error(f"Failed to load state for {experiment_id}: {e}")
            # Try to restore from backup
            return self._restore_from_backup(experiment_id)

    def can_resume(self, experiment_id: str) -> bool:
        """
        Check if an experiment can be resumed.

        Args:
            experiment_id: ID of the experiment to check

        Returns:
            True if resumable, False otherwise
        """
        state = self.load_state(experiment_id)
        if not state:
            return False

        # Can resume if not completed and has valid state
        return (state.status in [OrchestratorStatus.RUNNING, OrchestratorStatus.PAUSED, OrchestratorStatus.HUNG]
                and len(state.completed_experiments) > 0)

    def get_resumable_experiments(self) -> List[Dict[str, Any]]:
        """
        Get list of experiments that can be resumed.

        Returns:
            List of resumable experiment summaries
        """
        resumable = []
        state_dir = Path(self.config["persistence"]["storage_path"]) / "experiments"

        if not state_dir.exists():
            return resumable

        for exp_dir in state_dir.iterdir():
            if exp_dir.is_dir():
                state = self.load_state(exp_dir.name)
                if state and self.can_resume(exp_dir.name):
                    resumable.append({
                        "experiment_id": state.experiment_id,
                        "research_task": state.research_task,
                        "model": state.model,
                        "created_at": state.created_at.isoformat(),
                        "updated_at": state.updated_at.isoformat(),
                        "current_step": state.current_step,
                        "max_steps": state.max_steps,
                        "status": state.status.value,
                        "completed_experiments": [exp.to_dict() for exp in state.completed_experiments],
                        "available_checkpoints": [cp.to_dict() for cp in state.checkpoints],
                        "config": state.config.to_dict() if state.config else None,
                    })

        return sorted(resumable, key=lambda x: x["updated_at"], reverse=True)

    def update_experiment_result(self, experiment_id: str, result: ExperimentResult) -> bool:
        """
        Update or add an experiment result.

        Args:
            experiment_id: ID of the orchestrator experiment
            result: Result of a single agent experiment

        Returns:
            True if successful, False otherwise
        """
        state = self.load_state(experiment_id)
        if not state:
            logger.error(f"No state found for experiment {experiment_id}")
            return False

        # Remove existing result for this experiment if any
        state.completed_experiments = [
            exp for exp in state.completed_experiments
            if exp.experiment_id != result.experiment_id
        ]

        # Add the new result
        state.completed_experiments.append(result)

        # Remove from active if present
        if result.experiment_id in state.active_experiments:
            state.active_experiments.remove(result.experiment_id)

        return self.save_state(state)

    def add_active_experiment(self, experiment_id: str, agent_experiment_id: str) -> bool:
        """
        Add an experiment to the active list.

        Args:
            experiment_id: ID of the orchestrator experiment
            agent_experiment_id: ID of the agent experiment

        Returns:
            True if successful, False otherwise
        """
        state = self.load_state(experiment_id)
        if not state:
            logger.error(f"No state found for experiment {experiment_id}")
            return False

        if agent_experiment_id not in state.active_experiments:
            state.active_experiments.append(agent_experiment_id)

        return self.save_state(state)

    def update_step(self, experiment_id: str, current_step: int, max_steps: int) -> bool:
        """
        Update the current step of an experiment.

        Args:
            experiment_id: ID of the orchestrator experiment
            current_step: Current step number
            max_steps: Maximum number of steps

        Returns:
            True if successful, False otherwise
        """
        state = self.load_state(experiment_id)
        if not state:
            logger.error(f"No state found for experiment {experiment_id}")
            return False

        state.current_step = current_step
        state.max_steps = max_steps

        return self.save_state(state)

    def update_status(self, experiment_id: str, status: OrchestratorStatus) -> bool:
        """
        Update the status of an experiment.

        Args:
            experiment_id: ID of the orchestrator experiment
            status: New status

        Returns:
            True if successful, False otherwise
        """
        state = self.load_state(experiment_id)
        if not state:
            logger.error(f"No state found for experiment {experiment_id}")
            return False

        state.status = status

        return self.save_state(state)

    def delete_experiment_state(self, experiment_id: str) -> bool:
        """
        Delete all state data for an experiment.

        Args:
            experiment_id: ID of the experiment to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                exp_dir = get_experiment_dir(experiment_id)
                if exp_dir.exists():
                    shutil.rmtree(exp_dir)
                    logger.info(f"Deleted state for experiment {experiment_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete state for {experiment_id}: {e}")
            return False

    def _create_backup(self, state_file: Path, experiment_id: str) -> None:
        """Create a backup of the state file."""
        try:
            backup_dir = Path(self.config["persistence"]["storage_path"]) / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"{experiment_id}_{timestamp}.json"

            shutil.copy2(state_file, backup_file)
            logger.debug(f"Created backup: {backup_file}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    def _restore_from_backup(self, experiment_id: str) -> Optional[OrchestratorState]:
        """Try to restore state from backup."""
        try:
            backup_dir = Path(self.config["persistence"]["storage_path"]) / "backups"
            backup_pattern = f"{experiment_id}_*.json"

            backup_files = list(backup_dir.glob(backup_pattern))
            if not backup_files:
                return None

            # Get the most recent backup
            latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)

            with open(latest_backup, 'r', encoding='utf-8') as f:
                state_json = f.read()

            state = OrchestratorState.from_json(state_json)
            logger.info(f"Restored state from backup for {experiment_id}")
            return state

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return None

    def cleanup_old_states(self, max_age_days: int = 7) -> int:
        """
        Clean up old state files.

        Args:
            max_age_days: Maximum age in days before cleanup

        Returns:
            Number of experiments cleaned up
        """
        try:
            cleaned_count = 0
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)

            state_dir = Path(self.config["persistence"]["storage_path"]) / "experiments"
            if not state_dir.exists():
                return 0

            for exp_dir in state_dir.iterdir():
                if exp_dir.is_dir():
                    state_file = exp_dir / "state.json"
                    if state_file.exists():
                        if state_file.stat().st_mtime < cutoff_time:
                            shutil.rmtree(exp_dir)
                            cleaned_count += 1
                            logger.info(f"Cleaned up old state: {exp_dir.name}")

            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup old states: {e}")
            return 0
"""
Checkpoint management for orchestrator recovery.
"""
import json
import shutil
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
    CheckpointInfo,
    get_experiment_dir,
    ensure_state_dirs,
    DEFAULT_CONFIG
)

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint creation and restoration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DEFAULT_CONFIG
        ensure_state_dirs()

    def create_checkpoint(self, state: OrchestratorState, description: str = "") -> str:
        """
        Create a checkpoint of the current state.

        Args:
            state: Current orchestrator state
            description: Optional description for the checkpoint

        Returns:
            Checkpoint ID if successful, None otherwise
        """
        try:
            # Generate checkpoint ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_id = f"step_{state.current_step}_{timestamp}"

            # Create checkpoint info
            checkpoint_info = CheckpointInfo(
                id=checkpoint_id,
                description=description or f"Checkpoint at step {state.current_step}",
                created_at=datetime.now(timezone.utc),
                step=state.current_step
            )

            # Add to state's checkpoints list
            state.checkpoints.append(checkpoint_info)

            # Create checkpoint directory
            exp_dir = get_experiment_dir(state.experiment_id)
            checkpoints_dir = exp_dir / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)

            checkpoint_file = checkpoints_dir / f"{checkpoint_id}.json"

            # Save checkpoint
            checkpoint_data = {
                "checkpoint_info": checkpoint_info.to_dict(),
                "state": state.to_dict()
            }

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            # Cleanup old checkpoints if needed
            max_checkpoints = self.config["checkpoints"]["max_checkpoints_per_experiment"]
            self._cleanup_old_checkpoints(state.experiment_id, max_checkpoints)

            logger.info(f"Checkpoint created: {checkpoint_id}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return None

    def restore_from_checkpoint(self, experiment_id: str, checkpoint_id: str) -> Optional[OrchestratorState]:
        """
        Restore state from a specific checkpoint.

        Args:
            experiment_id: ID of the experiment
            checkpoint_id: ID of the checkpoint to restore from

        Returns:
            Restored state or None if not found
        """
        try:
            checkpoint_file = (
                get_experiment_dir(experiment_id) / "checkpoints" / f"{checkpoint_id}.json"
            )

            if not checkpoint_file.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_id}")
                return None

            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            state = OrchestratorState.from_dict(checkpoint_data["state"])
            logger.info(f"State restored from checkpoint: {checkpoint_id}")
            return state

        except Exception as e:
            logger.error(f"Failed to restore from checkpoint {checkpoint_id}: {e}")
            return None

    def get_available_checkpoints(self, experiment_id: str) -> List[CheckpointInfo]:
        """
        Get list of available checkpoints for an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            List of checkpoint information
        """
        checkpoints = []
        checkpoints_dir = get_experiment_dir(experiment_id) / "checkpoints"

        if not checkpoints_dir.exists():
            return checkpoints

        try:
            for checkpoint_file in checkpoints_dir.glob("*.json"):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)

                checkpoint_info = CheckpointInfo.from_dict(checkpoint_data["checkpoint_info"])
                checkpoints.append(checkpoint_info)

        except Exception as e:
            logger.error(f"Failed to load checkpoints for {experiment_id}: {e}")

        return sorted(checkpoints, key=lambda cp: cp.created_at, reverse=True)

    def delete_checkpoint(self, experiment_id: str, checkpoint_id: str) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            experiment_id: ID of the experiment
            checkpoint_id: ID of the checkpoint to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint_file = (
                get_experiment_dir(experiment_id) / "checkpoints" / f"{checkpoint_id}.json"
            )

            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"Checkpoint deleted: {checkpoint_id}")
                return True
            else:
                logger.warning(f"Checkpoint file not found: {checkpoint_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    def auto_save_checkpoint(self, state: OrchestratorState) -> Optional[str]:
        """
        Automatically save checkpoint if conditions are met.

        Args:
            state: Current orchestrator state

        Returns:
            Checkpoint ID if saved, None otherwise
        """
        interval = self.config["checkpoints"]["auto_save_interval_steps"]

        # Save if at the right interval
        if state.current_step % interval == 0:
            description = f"Auto-save at step {state.current_step}"
            return self.create_checkpoint(state, description)

        # Save before final report if enabled
        if (self.config["checkpoints"]["save_before_final_report"] and
            state.current_step == state.max_steps - 1):
            description = "Auto-save before final report"
            return self.create_checkpoint(state, description)

        return None

    def get_latest_checkpoint(self, experiment_id: str) -> Optional[CheckpointInfo]:
        """
        Get the most recent checkpoint for an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Latest checkpoint info or None if no checkpoints
        """
        checkpoints = self.get_available_checkpoints(experiment_id)
        return checkpoints[0] if checkpoints else None

    def _cleanup_old_checkpoints(self, experiment_id: str, max_checkpoints: int) -> None:
        """
        Clean up old checkpoints, keeping only the most recent ones.

        Args:
            experiment_id: ID of the experiment
            max_checkpoints: Maximum number of checkpoints to keep
        """
        try:
            checkpoints = self.get_available_checkpoints(experiment_id)

            if len(checkpoints) <= max_checkpoints:
                return

            # Remove oldest checkpoints
            checkpoints_to_remove = checkpoints[max_checkpoints:]
            for checkpoint in checkpoints_to_remove:
                self.delete_checkpoint(experiment_id, checkpoint.id)

            logger.info(f"Cleaned up {len(checkpoints_to_remove)} old checkpoints")

        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")

    def export_checkpoints(self, experiment_id: str, export_path: Path) -> bool:
        """
        Export all checkpoints for an experiment to a zip file.

        Args:
            experiment_id: ID of the experiment
            export_path: Path to export file

        Returns:
            True if successful, False otherwise
        """
        try:
            import zipfile

            checkpoints_dir = get_experiment_dir(experiment_id) / "checkpoints"
            if not checkpoints_dir.exists():
                logger.warning(f"No checkpoints found for experiment {experiment_id}")
                return False

            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for checkpoint_file in checkpoints_dir.glob("*.json"):
                    zipf.write(checkpoint_file, checkpoint_file.name)

            logger.info(f"Checkpoints exported to: {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export checkpoints: {e}")
            return False

    def import_checkpoints(self, experiment_id: str, import_path: Path) -> bool:
        """
        Import checkpoints from a zip file.

        Args:
            experiment_id: ID of the experiment
            import_path: Path to import file

        Returns:
            True if successful, False otherwise
        """
        try:
            import zipfile

            checkpoints_dir = get_experiment_dir(experiment_id) / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(import_path, 'r') as zipf:
                zipf.extractall(checkpoints_dir)

            logger.info(f"Checkpoints imported from: {import_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to import checkpoints: {e}")
            return False
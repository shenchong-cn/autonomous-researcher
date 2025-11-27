"""
Data models for orchestrator state management and recovery.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class OrchestratorStatus(Enum):
    """Orchestrator execution status."""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    HUNG = "hung"


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_id: str
    hypothesis: str
    status: str  # COMPLETED, FAILED, RUNNING
    exit_code: int
    stdout: str
    stderr: str
    started_at: datetime
    completed_at: Optional[datetime]
    gpu: Optional[str]
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "status": self.status,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "gpu": self.gpu,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            hypothesis=data["hypothesis"],
            status=data["status"],
            exit_code=data["exit_code"],
            stdout=data["stdout"],
            stderr=data["stderr"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None,
            gpu=data["gpu"],
            duration_seconds=data["duration_seconds"],
        )


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator execution."""
    num_agents: int
    max_rounds: int
    max_parallel: int
    gpu: Optional[str]
    model: str
    test_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_agents": self.num_agents,
            "max_rounds": self.max_rounds,
            "max_parallel": self.max_parallel,
            "gpu": self.gpu,
            "model": self.model,
            "test_mode": self.test_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorConfig":
        return cls(**data)


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    id: str
    description: str
    created_at: datetime
    step: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointInfo":
        return cls(
            id=data["id"],
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            step=data["step"],
        )


@dataclass
class OrchestratorState:
    """Complete orchestrator state for recovery."""
    # Basic information
    experiment_id: str
    research_task: str
    model: str
    created_at: datetime
    updated_at: datetime

    # Execution status
    current_step: int
    max_steps: int
    status: OrchestratorStatus

    # Experiment results
    completed_experiments: List[ExperimentResult] = field(default_factory=list)
    active_experiments: List[str] = field(default_factory=list)

    # Message history
    message_history: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration
    config: Optional[OrchestratorConfig] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Checkpoints
    checkpoints: List[CheckpointInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "research_task": self.research_task,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "status": self.status.value,
            "completed_experiments": [exp.to_dict() for exp in self.completed_experiments],
            "active_experiments": self.active_experiments,
            "message_history": self.message_history,
            "config": self.config.to_dict() if self.config else None,
            "metadata": self.metadata,
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorState":
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            research_task=data["research_task"],
            model=data["model"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            current_step=data["current_step"],
            max_steps=data["max_steps"],
            status=OrchestratorStatus(data["status"]),
            completed_experiments=[ExperimentResult.from_dict(exp) for exp in data["completed_experiments"]],
            active_experiments=data["active_experiments"],
            message_history=data["message_history"],
            config=OrchestratorConfig.from_dict(data["config"]) if data["config"] else None,
            metadata=data["metadata"],
            checkpoints=[CheckpointInfo.from_dict(cp) for cp in data["checkpoints"]],
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "OrchestratorState":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Configuration constants
DEFAULT_STATE_DIR = Path(".orchestrator_state")
DEFAULT_CONFIG = {
    "recovery": {
        "enabled": True,
        "hang_timeout_seconds": 600,
        "experiment_timeout_seconds": 1800,
        "max_recovery_attempts": 3,
        "auto_cleanup_days": 7
    },
    "persistence": {
        "storage_path": ".orchestrator_state",
        "backup_enabled": True,
        "compression_enabled": False,
        "max_state_file_size_mb": 100
    },
    "checkpoints": {
        "auto_save_interval_steps": 1,
        "max_checkpoints_per_experiment": 10,
        "save_before_final_report": True
    }
}


def get_state_dir() -> Path:
    """Get the state directory path."""
    state_path = os.environ.get("ORCHESTRATOR_STATE_PATH", DEFAULT_STATE_DIR)
    return Path(state_path)


def get_experiment_dir(experiment_id: str) -> Path:
    """Get the directory for a specific experiment."""
    return get_state_dir() / "experiments" / experiment_id


def ensure_state_dirs() -> None:
    """Ensure all necessary state directories exist."""
    state_dir = get_state_dir()
    experiments_dir = state_dir / "experiments"
    checkpoints_dir = state_dir / "checkpoints"

    for directory in [state_dir, experiments_dir, checkpoints_dir]:
        directory.mkdir(parents=True, exist_ok=True)
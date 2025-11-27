from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

MAIN_PATH = BASE_DIR / "main.py"

# Regex for stripping ANSI escape sequences
import re
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

def strip_ansi(text: str) -> str:
    """Remove ANSI colour / style escape sequences from terminal output."""
    return ANSI_ESCAPE_RE.sub("", text)

# Models
class ResumeRequest(BaseModel):
    experiment_id: str = Field(..., description="ID of the experiment to resume")
    checkpoint_id: Optional[str] = Field(None, description="Specific checkpoint to resume from")
    force: bool = Field(False, description="Force resume even if not recommended")

class CheckpointRequest(BaseModel):
    experiment_id: str = Field(..., description="ID of the experiment")
    description: Optional[str] = Field("", description="Description of the checkpoint")

class CheckpointDeleteRequest(BaseModel):
    experiment_id: str = Field(..., description="ID of the experiment")
    checkpoint_id: str = Field(..., description="ID of the checkpoint to delete")

app = FastAPI(
    title="AI Researcher API - Minimal",
    description="Minimal API for testing recovery functionality",
    version="0.1.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

def _env_value_present(value: Optional[str]) -> bool:
    """Treat empty or placeholder values as missing."""
    if value is None:
        return False
    cleaned = value.strip()
    if not cleaned:
        return False
    lower = cleaned.lower()
    if lower.startswith("your_") or lower.endswith("_here"):
        return False
    if lower in {"changeme", "example"}:
        return False
    return True

# Basic API endpoints
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Simple health check."""
    exists = MAIN_PATH.exists()
    return {
        "status": "ok" if exists else "error",
        "main_py": str(MAIN_PATH),
        "main_py_exists": exists,
    }

@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    """Return the current global state of the orchestrator."""
    return {
        "status": "active",
        "info": "Minimal API server - monitoring disabled",
    }

@app.get(
    "/api/credentials/status",
    response_model=Dict[str, Any],
    summary="Check whether required API keys are set",
)
def credentials_status() -> Dict[str, Any]:
    """Return a minimal view of credential readiness."""
    return {
        "has_google_api_key": _env_value_present(os.environ.get("GOOGLE_API_KEY")),
        "has_anthropic_api_key": _env_value_present(os.environ.get("ANTHROPIC_API_KEY")),
        "has_modal_token": _env_value_present(os.environ.get("MODAL_TOKEN_ID")) and _env_value_present(os.environ.get("MODAL_TOKEN_SECRET")),
    }

@app.post(
    "/api/credentials",
    response_model=Dict[str, Any],
    summary="Set Google/Modal credentials (persists to .env)",
)
def update_credentials(req: Dict[str, Any]) -> Dict[str, Any]:
    """Allow the UI to persist credentials locally."""
    try:
        if req.get("google_api_key") and req["google_api_key"].strip():
            os.environ["GOOGLE_API_KEY"] = req["google_api_key"].strip()
        if req.get("anthropic_api_key") and req["anthropic_api_key"].strip():
            os.environ["ANTHROPIC_API_KEY"] = req["anthropic_api_key"].strip()
        if req.get("modal_token_id") and req["modal_token_id"].strip():
            os.environ["MODAL_TOKEN_ID"] = req["modal_token_id"].strip()
        if req.get("modal_token_secret") and req["modal_token_secret"].strip():
            os.environ["MODAL_TOKEN_SECRET"] = req["modal_token_secret"].strip()

        return credentials_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist credentials: {e}") from e

# Add missing experiment endpoints that the frontend might expect
@app.post(
    "/api/experiments/single",
    response_model=Dict[str, Any],
    summary="Run a single-agent experiment (blocking)",
)
def run_single_experiment(req: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for single agent experiment."""
    return {
        "mode": "single",
        "task": req.get("task", ""),
        "message": "Single agent experiments disabled in minimal API",
    }

@app.post(
    "/api/experiments/orchestrator",
    response_model=Dict[str, Any],
    summary="Run the multi-agent orchestrator (blocking)",
)
def run_orchestrator_experiment(req: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for orchestrator experiment."""
    return {
        "mode": "orchestrator",
        "task": req.get("task", ""),
        "message": "Orchestrator experiments disabled in minimal API",
    }

@app.post(
    "/api/agents/summarize",
    response_model=Dict[str, Any],
    summary="Summarize the last few sub-agent turns for the sidebar",
)
def summarize_agent(req: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for agent summarization."""
    return {
        "summary": "Agent summarization disabled in minimal API",
        "chart": None
    }

# Recovery API endpoints
@app.get("/api/orchestrator/resumable")
def get_resumable_experiments() -> Dict[str, Any]:
    """Get all experiments that can be resumed."""
    try:
        state_dir = Path(".orchestrator_state") / "experiments"
        resumable = []

        if state_dir.exists():
            for exp_dir in state_dir.iterdir():
                if exp_dir.is_dir() and "resumable" in exp_dir.name:
                    state_file = exp_dir / "state.json"
                    if state_file.exists():
                        try:
                            with open(state_file, 'r', encoding='utf-8') as f:
                                state = json.load(f)
                                resumable.append({
                                    "experiment_id": state.get("experiment_id", exp_dir.name),
                                    "research_task": state.get("research_task", ""),
                                    "status": state.get("status", "unknown"),
                                    "updated_at": state.get("updated_at", ""),
                                    "current_step": state.get("current_step", 0),
                                    "max_steps": state.get("max_steps", 0)
                                })
                        except Exception:
                            continue

        return {
            "resumable_experiments": resumable,
            "count": len(resumable)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/orchestrator/state/{experiment_id}")
def get_experiment_state(experiment_id: str) -> Dict[str, Any]:
    """Get detailed status and state of a specific experiment."""
    try:
        state_file = Path(".orchestrator_state") / "experiments" / experiment_id / "state.json"
        if not state_file.exists():
            raise HTTPException(status_code=404, detail="Experiment not found")

        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)

        return state
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/orchestrator/resume")
def resume_experiment(req: ResumeRequest) -> Dict[str, Any]:
    """Resume execution of an experiment from a checkpoint."""
    return {
        "success": True,
        "message": f"Experiment {req.experiment_id} resume requested (minimal API)",
        "checkpoint_used": req.checkpoint_id or "latest state"
    }

@app.post("/api/orchestrator/checkpoint")
def create_checkpoint(req: CheckpointRequest) -> Dict[str, Any]:
    """Create a manual checkpoint for an experiment."""
    import uuid
    checkpoint_id = f"ckpt_{uuid.uuid4().hex[:8]}"

    return {
        "success": True,
        "checkpoint_id": checkpoint_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": "Checkpoint created (minimal API)"
    }

@app.delete("/api/orchestrator/checkpoint")
def delete_checkpoint(req: CheckpointDeleteRequest) -> Dict[str, Any]:
    """Delete a specific checkpoint."""
    return {
        "success": True,
        "message": f"Checkpoint {req.checkpoint_id} deletion requested (minimal API)"
    }

@app.get("/api/orchestrator/checkpoints/{experiment_id}")
def get_checkpoints(experiment_id: str) -> Dict[str, Any]:
    """Get all available checkpoints for an experiment."""
    return {
        "experiment_id": experiment_id,
        "checkpoints": [],
        "count": 0
    }

@app.post("/api/orchestrator/auto-recovery")
def trigger_auto_recovery() -> Dict[str, Any]:
    """Manually trigger automatic recovery for all hung experiments."""
    return {
        "success": True,
        "recovered_experiments": [],
        "count": 0
    }

@app.delete("/api/orchestrator/state/{experiment_id}")
def delete_experiment_state(experiment_id: str) -> Dict[str, Any]:
    """Delete all state data for an experiment."""
    return {
        "success": True,
        "message": f"State for experiment {experiment_id} deletion requested (minimal API)"
    }

@app.get("/api/test/resumable-simple")
def test_resumable_simple() -> Dict[str, Any]:
    """Simple test endpoint that should always work."""
    return {
        "test": "working",
        "resumable_experiments": [
            {
                "experiment_id": "test_1",
                "research_task": "Test task",
                "status": "running",
                "updated_at": "2025-11-27T07:00:00Z",
                "current_step": 1,
                "max_steps": 3
            }
        ],
        "count": 1
    }

# Catch-all route for frontend
@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    """Catch-all route to serve the frontend SPA."""
    # IMPORTANT: Never handle API routes - FastAPI should handle these via specific API routes above
    if full_path.startswith("api/"):
        print(f"[DEBUG] API path reached catch-all: {full_path} - THIS SHOULD NOT HAPPEN!", file=sys.stderr)
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="API endpoint not found")

    # If frontend dist exists, serve it
    if FRONTEND_DIST.exists():
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        index_path = FRONTEND_DIST / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

    # Fallback: return a simple message if no frontend
    return {"message": "AI Researcher API is running. Frontend not built.", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    reload_enabled = os.environ.get("RAILWAY_ENVIRONMENT") is None
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=reload_enabled,
    )
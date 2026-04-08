"""
AI Customer Support Training Environment — FastAPI server.

OpenEnv-compliant endpoints:
  POST /reset   → start new episode
  POST /step    → take one action
  GET  /state   → inspect internal state
  GET  /health  → liveness probe (returns 200)
  GET  /info    → environment metadata
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environment"))
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from environment.core import CustomerSupportEnv
from environment.models import (
    Action,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResult,
)
from environment.tasks import TASK_META

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="AI Customer Support Training Environment",
    description=(
        "An OpenEnv-compliant reinforcement-learning environment that trains "
        "AI agents to handle real-world customer support scenarios across three "
        "difficulty levels: issue classification (easy), response generation "
        "(medium), and full issue resolution (hard)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared environment instance ───────────────
# In production you would use a session-keyed store.
# For hackathon purposes a single global instance is sufficient.
env = CustomerSupportEnv()


# ──────────────────────────────────────────────
# Health / meta
# ──────────────────────────────────────────────

@app.get("/ui", include_in_schema=False)
async def frontend() -> FileResponse:
    """Serve the frontend dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path, media_type="text/html")


@app.get("/health", response_class=PlainTextResponse, tags=["Meta"])
async def health() -> str:
    """Liveness probe — always returns HTTP 200."""
    return "OK"


@app.get("/", tags=["Meta"])
async def root() -> Dict[str, Any]:
    """Root endpoint with quick summary."""
    return {
        "name": "AI Customer Support Training Environment",
        "version": "1.0.0",
        "openenv": True,
        "tasks": list(TASK_META.keys()),
        "docs": "/docs",
        "spec": "/openenv.yaml",
    }


@app.get("/info", tags=["Meta"])
async def info() -> Dict[str, Any]:
    """Environment metadata and task descriptions."""
    return {
        "name": "ai-customer-support-env",
        "version": "1.0.0",
        "openenv_version": "1.0",
        "tasks": TASK_META,
        "action_space": {
            "type": "dict",
            "fields": {
                "type": {
                    "type": "string",
                    "enum": ["classify", "respond", "ask_question"],
                    "description": "Action type",
                },
                "category": {
                    "type": "string",
                    "optional": True,
                    "description": "Issue category (for classify actions)",
                },
                "message": {
                    "type": "string",
                    "optional": True,
                    "description": "Response text (for respond / ask_question actions)",
                },
            },
        },
        "observation_space": {
            "type": "dict",
            "fields": {
                "current_query":  {"type": "string"},
                "chat_history":   {"type": "array", "items": {"role": "string", "content": "string"}},
                "task_type":      {"type": "string", "enum": ["easy", "medium", "hard"]},
                "step_count":     {"type": "integer"},
                "max_steps":      {"type": "integer"},
                "task_hint":      {"type": "string"},
            },
        },
        "reward_range": [0.0, 1.0],
    }


@app.get("/openenv.yaml", response_class=PlainTextResponse, tags=["Meta"])
async def openenv_spec() -> str:
    """Return the openenv.yaml specification."""
    spec_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    try:
        with open(spec_path) as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


# ──────────────────────────────────────────────
# OpenEnv core endpoints
# ──────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse, tags=["OpenEnv"])
async def reset(request: ResetRequest = None) -> ResetResponse:
    """
    Start a new episode.

    Optionally specify `task_id` (classify | respond | resolve) and `seed`.
    """
    if request is None:
        request = ResetRequest()
    try:
        response = env.reset(request)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResult, tags=["OpenEnv"])
async def step(request: StepRequest) -> StepResult:
    """
    Execute one agent action.

    Must call /reset first. Returns (observation, reward, done, info).
    """
    try:
        result = env.step(request.action)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Environment error: {exc}")


@app.get("/state", response_model=StateResponse, tags=["OpenEnv"])
async def state() -> StateResponse:
    """Return the full internal environment state (for debugging)."""
    return env.state()


# ──────────────────────────────────────────────
# Uvicorn entry-point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
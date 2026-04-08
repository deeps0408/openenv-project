"""
Typed Pydantic models for the AI Customer Support Training Environment.
OpenEnv-compliant action, observation, and response schemas.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class ActionType(str, Enum):
    CLASSIFY    = "classify"
    RESPOND     = "respond"
    ASK_QUESTION = "ask_question"


class IssueCategory(str, Enum):
    BILLING   = "billing"
    SHIPPING  = "shipping"
    REFUND    = "refund"
    TECHNICAL = "technical"
    GENERAL   = "general"


class TaskType(str, Enum):
    EASY   = "easy"    # Task 1 – Classification
    MEDIUM = "medium"  # Task 2 – Response Generation
    HARD   = "hard"    # Task 3 – Full Resolution


# ──────────────────────────────────────────────
# Action space
# ──────────────────────────────────────────────

class Action(BaseModel):
    """Action taken by the agent."""
    type: ActionType = Field(..., description="Type of action: classify | respond | ask_question")
    category: Optional[str] = Field(
        None,
        description="Issue category (required for 'classify' actions)",
        examples=["billing", "shipping", "refund", "technical", "general"],
    )
    message: Optional[str] = Field(
        None,
        description="Text message (required for 'respond' and 'ask_question' actions)",
    )

    class Config:
        use_enum_values = True


# ──────────────────────────────────────────────
# Observation space
# ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    """Single turn in the conversation history."""
    role: str = Field(..., description="'customer' or 'agent'")
    content: str = Field(..., description="Message text")


class Observation(BaseModel):
    """What the agent perceives — never the full internal state."""
    current_query: str      = Field(..., description="Latest customer message")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Conversation so far")
    task_type: TaskType     = Field(..., description="Difficulty of the current task")
    step_count: int         = Field(0,   description="Steps taken so far in this episode")
    max_steps: int          = Field(5,   description="Maximum allowed steps")
    task_hint: str          = Field("",  description="Optional hint for the agent")

    class Config:
        use_enum_values = True


# ──────────────────────────────────────────────
# API request / response schemas
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(
        None,
        description="Task to run: 'classify' | 'respond' | 'resolve'. Random if omitted.",
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ResetResponse(BaseModel):
    observation: Observation
    task_id: str
    episode_id: str


class StepRequest(BaseModel):
    action: Action


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(..., ge=0.0, le=1.0, description="Normalised step reward [0.0, 1.0]")
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    """Full internal state (for debugging / monitoring)."""
    episode_id: str
    task_id: str
    step_count: int
    done: bool
    cumulative_reward: float
    internal_state: Dict[str, Any]
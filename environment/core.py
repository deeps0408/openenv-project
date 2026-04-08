"""
Core environment class.

Implements the OpenEnv lifecycle:
  env.reset(task_id, seed) → ResetResponse
  env.step(action)         → StepResult
  env.state()              → StateResponse
"""
from __future__ import annotations

import random
import uuid
from typing import Any, Dict, Optional

from environment.graders import grade_classify, grade_respond, grade_resolve_step
from environment.models import (
    Action,
    ChatMessage,
    Observation,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepResult,
    TaskType,
)
from environment.tasks import (
    CLASSIFY_SCENARIOS,
    RESOLVE_SCENARIOS,
    RESPOND_SCENARIOS,
    TASK_META,
    ClassifyScenario,
    ResolveScenario,
    RespondScenario,
)


class CustomerSupportEnv:
    """
    Stateful environment instance.

    One instance = one active episode. Call reset() to start a new episode.
    """

    # ──────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._task_id: str    = ""
        self._step_count: int = 0
        self._done: bool      = True
        self._cumulative_reward: float = 0.0

        self._scenario: Any = None          # typed below per task
        self._internal_state: Dict[str, Any] = {}
        self._chat_history: list            = []

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def reset(self, request: Optional[ResetRequest] = None) -> ResetResponse:
        """Start a new episode, optionally for a specific task_id."""
        if request is None:
            request = ResetRequest()

        # Seed RNG for reproducibility
        seed = request.seed if request.seed is not None else random.randint(0, 2**31)
        rng  = random.Random(seed)

        # Choose task
        task_id = request.task_id or rng.choice(["classify", "respond", "resolve"])
        if task_id not in TASK_META:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(TASK_META.keys())}"
            )

        self._task_id      = task_id
        self._episode_id   = str(uuid.uuid4())
        self._step_count   = 0
        self._done         = False
        self._cumulative_reward = 0.0
        self._chat_history = []
        self._internal_state = {
            "task_id":              task_id,
            "seed":                 seed,
            "issue_identified":     False,
            "clarification_asked":  False,
            "solution_given":       False,
            "bonus_given":          False,
            "last_agent_message":   "",
        }

        # Pick scenario
        if task_id == "classify":
            self._scenario = rng.choice(CLASSIFY_SCENARIOS)
        elif task_id == "respond":
            self._scenario = rng.choice(RESPOND_SCENARIOS)
        elif task_id == "resolve":
            self._scenario = rng.choice(RESOLVE_SCENARIOS)

        obs = self._build_observation()
        return ResetResponse(
            observation=obs,
            task_id=self._task_id,
            episode_id=self._episode_id,
        )

    def step(self, action: Action) -> StepResult:
        """Execute one agent action and return (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset() to start a new episode."
            )

        meta        = TASK_META[self._task_id]
        max_steps   = meta["max_steps"]
        reward      = 0.0
        done        = False
        feedback    = ""

        # ── Dispatch to grader ────────────────
        if self._task_id == "classify":
            score, feedback = grade_classify(action, self._scenario)
            reward = score
            done   = True   # single-step task

        elif self._task_id == "respond":
            score, feedback = grade_respond(action, self._scenario)
            reward = score
            done   = True   # single-step task

        elif self._task_id == "resolve":
            score, done, new_state, feedback = grade_resolve_step(
                action, self._scenario, self._internal_state
            )
            reward = score
            self._internal_state.update(new_state)

        # ── Update counters ───────────────────
        self._step_count        += 1
        self._cumulative_reward += reward

        # Append agent message to chat history
        if action.message:
            self._chat_history.append(
                ChatMessage(role="agent", content=action.message)
            )
        elif action.category:
            self._chat_history.append(
                ChatMessage(role="agent", content=f"[classify: {action.category}]")
            )

        # Force done if max steps reached
        if self._step_count >= max_steps and not done:
            done     = True
            feedback += " | Max steps reached."

        self._done = done
        obs = self._build_observation()

        info = {
            "episode_id":        self._episode_id,
            "task_id":           self._task_id,
            "step":              self._step_count,
            "feedback":          feedback,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "milestones": {
                "issue_identified":    self._internal_state.get("issue_identified",    False),
                "clarification_asked": self._internal_state.get("clarification_asked", False),
                "solution_given":      self._internal_state.get("solution_given",      False),
            } if self._task_id == "resolve" else {},
        }

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=done,
            info=info,
        )

    def state(self) -> StateResponse:
        """Return full internal state for monitoring / debugging."""
        return StateResponse(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._step_count,
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            internal_state={
                **self._internal_state,
                "scenario_query": getattr(self._scenario, "query", ""),
            },
        )

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _build_observation(self) -> Observation:
        """Construct the agent-visible observation from current state."""
        meta     = TASK_META[self._task_id]
        scenario = self._scenario

        query = getattr(scenario, "query", "")
        hint  = getattr(scenario, "context_hint", "")

        task_type_map = {
            "classify": TaskType.EASY,
            "respond":  TaskType.MEDIUM,
            "resolve":  TaskType.HARD,
        }

        history = list(self._chat_history)

        # For resolve: add customer query as first chat turn if no history
        if self._task_id == "resolve" and not history:
            history = [ChatMessage(role="customer", content=query)]

        return Observation(
            current_query=query,
            chat_history=history,
            task_type=task_type_map[self._task_id],
            step_count=self._step_count,
            max_steps=meta["max_steps"],
            task_hint=hint,
        )
"""
Agent graders for all three tasks.

Every grader returns a tuple: (score: float [0.0–1.0], feedback: str)

Grader rules are deterministic and rule-based so scores are reproducible
without calling any external LLM.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from models import Action
from tasks import (
    POLITENESS_WORDS,
    VALID_CATEGORIES,
    ClassifyScenario,
    RespondScenario,
    ResolveScenario,
)


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def _text_lower(action: Action) -> str:
    """Return the action message lowercased (or empty string)."""
    return (action.message or "").lower()


def _contains_any(text: str, words: List[str]) -> bool:
    return any(w.lower() in text for w in words)


def _count_politeness(text: str) -> int:
    return sum(1 for w in POLITENESS_WORDS if w.lower() in text)


# ──────────────────────────────────────────────
# Task 1 – Easy: Classification Grader
# ──────────────────────────────────────────────

def grade_classify(action: Action, scenario: ClassifyScenario) -> Tuple[float, str]:
    """
    Scoring:
      +1.0  correct category
       0.0  wrong category or invalid category
    """
    if action.type != "classify":
        return 0.0, "Action type must be 'classify'."

    if not action.category:
        return 0.0, "No category provided in action."

    category = (action.category or "").strip().lower()

    if category not in VALID_CATEGORIES:
        return 0.0, (
            f"'{category}' is not a valid category. "
            f"Choose from: {', '.join(VALID_CATEGORIES)}."
        )

    if category == scenario.correct_category:
        return 1.0, f"Correct! Category '{category}' matches expected '{scenario.correct_category}'."

    return 0.0, (
        f"Wrong category '{category}'. "
        f"Expected '{scenario.correct_category}'."
    )


# ──────────────────────────────────────────────
# Task 2 – Medium: Response Generation Grader
# ──────────────────────────────────────────────

def grade_respond(action: Action, scenario: RespondScenario) -> Tuple[float, str]:
    """
    Scoring rubric (four dimensions → averaged):

    Relevance   (0.0–0.4): Response mentions at least one required keyword.
    Politeness  (0.0–0.3): Contains 1+ politeness words.
    Length      (0.0–0.2): Response ≥ 20 words.
    Actionable  (0.0–0.1): Mentions next step / resolution hint.

    Total max = 1.0
    """
    if action.type not in ("respond",):
        return 0.0, "Action type must be 'respond'."

    text = _text_lower(action)

    if not text:
        return 0.0, "Empty response message."

    score = 0.0
    notes: List[str] = []

    # ── Relevance (0.4) ──────────────────────
    if _contains_any(text, scenario.required_keywords):
        score += 0.40
        notes.append("✔ Relevant to the issue.")
    else:
        notes.append(
            f"✘ Response does not address the issue. "
            f"Missing keywords: {scenario.required_keywords}."
        )

    # ── Politeness (0.3) ─────────────────────
    pol_count = _count_politeness(text)
    if pol_count >= 2:
        score += 0.30
        notes.append("✔ Polite tone (2+ politeness words).")
    elif pol_count == 1:
        score += 0.15
        notes.append("~ Partially polite (1 politeness word).")
    else:
        notes.append("✘ Response lacks polite/empathetic language.")

    # ── Length ≥ 20 words (0.2) ──────────────
    word_count = len(text.split())
    if word_count >= 20:
        score += 0.20
        notes.append(f"✔ Sufficient length ({word_count} words).")
    elif word_count >= 10:
        score += 0.10
        notes.append(f"~ Response is somewhat short ({word_count} words).")
    else:
        notes.append(f"✘ Response too short ({word_count} words).")

    # ── Actionable / resolution hint (0.1) ───
    resolution_words = [
        "will", "can", "shall", "initiate", "process", "refund",
        "contact", "resolve", "fix", "investigate", "escalate",
        "check", "look into", "arrange", "send", "provide",
    ]
    if _contains_any(text, resolution_words):
        score += 0.10
        notes.append("✔ Response is actionable.")
    else:
        notes.append("✘ No clear next step or action mentioned.")

    score = min(round(score, 4), 1.0)
    feedback = " | ".join(notes)
    return score, feedback


# ──────────────────────────────────────────────
# Task 3 – Hard: Full Resolution Grader (per-step)
# ──────────────────────────────────────────────

# Progressive reward milestones
_MILESTONE_ISSUE_IDENTIFIED    = "issue_identified"
_MILESTONE_CLARIFICATION_ASKED = "clarification_asked"
_MILESTONE_SOLUTION_GIVEN      = "solution_given"

_MILESTONE_REWARDS: Dict[str, float] = {
    _MILESTONE_ISSUE_IDENTIFIED:    0.20,
    _MILESTONE_CLARIFICATION_ASKED: 0.20,
    _MILESTONE_SOLUTION_GIVEN:      0.30,
}
_COMPLETION_BONUS = 0.30   # awarded when all three milestones are done


def grade_resolve_step(
    action: Action,
    scenario: ResolveScenario,
    internal_state: Dict[str, Any],
) -> Tuple[float, bool, Dict[str, Any], str]:
    """
    Evaluate one step of the full-resolution task.

    Returns
    -------
    step_reward : float   – reward for this single step [0.0, 1.0]
    done        : bool    – True if episode should end
    new_state   : dict    – updated internal state
    feedback    : str
    """
    text   = _text_lower(action)
    state  = dict(internal_state)          # shallow copy to avoid mutation
    reward = 0.0
    notes: List[str] = []
    done   = False

    issue_id  = state.get(_MILESTONE_ISSUE_IDENTIFIED,    False)
    clarif    = state.get(_MILESTONE_CLARIFICATION_ASKED, False)
    solution  = state.get(_MILESTONE_SOLUTION_GIVEN,      False)

    # ── Penalty: empty or repetition ─────────
    if not text:
        return 0.0, False, state, "✘ Empty message — no reward."

    last_agent_msg = state.get("last_agent_message", "")
    if text and text == last_agent_msg:
        return 0.0, False, state, "✘ Repeated message — penalty applied."

    state["last_agent_message"] = text

    # ── Milestone 1: Identify the issue ──────
    issue_words = [
        "billing", "shipping", "refund", "technical",
        "charge", "payment", "order", "delay", "wrong",
        "understand", "issue", "problem", "concern", "complaint",
    ]
    if not issue_id and _contains_any(text, issue_words):
        state[_MILESTONE_ISSUE_IDENTIFIED] = True
        reward += _MILESTONE_REWARDS[_MILESTONE_ISSUE_IDENTIFIED]
        notes.append("✔ Issue identified (+0.20).")

    # ── Milestone 2: Ask for clarification ───
    clarif_triggers = (
        scenario.clarification_keywords
        + ["?", "could you", "can you", "please provide", "share", "let me know"]
    )
    is_question = action.type == "ask_question" or "?" in text
    if (
        not clarif
        and state.get(_MILESTONE_ISSUE_IDENTIFIED, False)
        and is_question
        and _contains_any(text, clarif_triggers)
    ):
        state[_MILESTONE_CLARIFICATION_ASKED] = True
        reward += _MILESTONE_REWARDS[_MILESTONE_CLARIFICATION_ASKED]
        notes.append("✔ Clarification requested (+0.20).")

    # ── Milestone 3: Provide solution ────────
    solution_words = [
        "refund", "refunded", "resolve", "resolved", "fix", "fixed",
        "process", "processed", "initiated", "arrange", "arranged",
        "replacement", "compensate", "credit", "waive", "investigate",
        "escalate", "team", "within", "business day",
    ]
    if (
        not solution
        and state.get(_MILESTONE_CLARIFICATION_ASKED, False)
        and action.type == "respond"
        and _contains_any(text, solution_words)
    ):
        state[_MILESTONE_SOLUTION_GIVEN] = True
        reward += _MILESTONE_REWARDS[_MILESTONE_SOLUTION_GIVEN]
        notes.append("✔ Solution provided (+0.30).")

    # ── Completion bonus ─────────────────────
    all_done = (
        state.get(_MILESTONE_ISSUE_IDENTIFIED,    False)
        and state.get(_MILESTONE_CLARIFICATION_ASKED, False)
        and state.get(_MILESTONE_SOLUTION_GIVEN,      False)
    )
    if all_done and not state.get("bonus_given", False):
        state["bonus_given"] = True
        reward += _COMPLETION_BONUS
        notes.append("🎉 Full resolution bonus (+0.30)!")
        done = True

    # ── Politeness bonus (small, up to 0.05) ─
    pol_count = _count_politeness(text)
    if pol_count >= 1 and reward > 0:
        pol_bonus = min(pol_count * 0.02, 0.05)
        reward    = min(reward + pol_bonus, 1.0)
        notes.append(f"✔ Polite language (+{pol_bonus:.2f}).")

    if not notes:
        notes.append("~ No milestone reached in this step.")

    reward = min(round(reward, 4), 1.0)
    return reward, done, state, " | ".join(notes)
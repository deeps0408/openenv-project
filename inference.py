"""
inference.py — Baseline agent for the AI Customer Support Training Environment.

Mandatory log format (do NOT alter field names, ordering, or structure):
  [START] {"episode": N, "task_id": "...", "seed": N}
  [STEP]  {"episode": N, "step": N, "action": {...}, "reward": N, "done": bool,
            "cumulative_reward": N, "feedback": "..."}
  [END]   {"episode": N, "task_id": "...", "total_reward": N, "score": N, "steps": N}

Environment variables required:
  API_BASE_URL   – LLM API base URL  (e.g. https://api.openai.com/v1)
  MODEL_NAME     – Model identifier  (e.g. gpt-4o-mini)
  HF_TOKEN       – API key           (forwarded as Bearer token)

Usage:
  python inference.py
  python inference.py --env-url http://localhost:7860  # override env server URL
  python inference.py --episodes 2 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# Default environment server (local FastAPI)
DEFAULT_ENV_URL = "http://localhost:7860"

# Tasks to evaluate (one episode per task for baseline)
EVAL_TASKS = ["classify", "respond", "resolve"]

# Seeds for reproducibility
TASK_SEEDS = {"classify": 1001, "respond": 2001, "resolve": 3001}


# ──────────────────────────────────────────────
# Structured logging helpers  (DO NOT MODIFY FORMAT)
# ──────────────────────────────────────────────

def log_start(episode: int, task_id: str, seed: int) -> None:
    record = {"episode": episode, "task_id": task_id, "seed": seed}
    print(f"[START] {json.dumps(record)}", flush=True)


def log_step(
    episode: int,
    step: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    cumulative_reward: float,
    feedback: str,
) -> None:
    record = {
        "episode":           episode,
        "step":              step,
        "action":            action,
        "reward":            reward,
        "done":              done,
        "cumulative_reward": round(cumulative_reward, 4),
        "feedback":          feedback,
    }
    print(f"[STEP] {json.dumps(record)}", flush=True)


def log_end(
    episode: int,
    task_id: str,
    total_reward: float,
    score: float,
    steps: int,
) -> None:
    record = {
        "episode":      episode,
        "task_id":      task_id,
        "total_reward": round(total_reward, 4),
        "score":        round(score, 4),
        "steps":        steps,
    }
    print(f"[END] {json.dumps(record)}", flush=True)


# ──────────────────────────────────────────────
# OpenAI client (uses env vars)
# ──────────────────────────────────────────────

def build_openai_client() -> OpenAI:
    return OpenAI(
        api_key=HF_TOKEN or "sk-placeholder",
        base_url=API_BASE_URL,
    )


# ──────────────────────────────────────────────
# LLM prompts per task
# ──────────────────────────────────────────────

SYSTEM_CLASSIFY = """\
You are a customer support AI. Your job is to classify customer queries.
Valid categories: billing, shipping, refund, technical, general.

ALWAYS respond with a valid JSON object — no markdown, no explanation:
{"type": "classify", "category": "<one of the five categories>"}
"""

SYSTEM_RESPOND = """\
You are an empathetic, professional customer support agent.
Write a helpful, polite response to the customer's complaint.
Your response must:
- Acknowledge the specific issue
- Use polite/empathetic language (sorry, apologize, thank you, etc.)
- Be at least 20 words long
- Mention a concrete next step or resolution

ALWAYS respond with a valid JSON object — no markdown, no explanation:
{"type": "respond", "message": "<your response here>"}
"""

SYSTEM_RESOLVE = """\
You are a senior customer support agent handling a complex issue.
Resolve the customer's problem across multiple turns by following this order:
  Turn 1: Identify and acknowledge the issue type.
  Turn 2: Ask a clarifying question (order ID, email, dates, etc.).
  Turn 3: Provide a concrete solution and close the ticket.

Use polite, empathetic language throughout.

ALWAYS respond with a valid JSON object — no markdown, no explanation.
Use one of these formats:
  Turns 1 or 3: {"type": "respond", "message": "<your message>"}
  Turn 2:       {"type": "ask_question", "message": "<your question>"}
"""


def build_user_prompt_classify(obs: Dict[str, Any]) -> str:
    return f'Customer query: "{obs["current_query"]}"\n\nClassify this query.'


def build_user_prompt_respond(obs: Dict[str, Any]) -> str:
    hint = obs.get("task_hint", "")
    return (
        f'Customer complaint: "{obs["current_query"]}"\n'
        + (f"Hint: {hint}\n" if hint else "")
        + "\nWrite a professional response."
    )


def build_user_prompt_resolve(obs: Dict[str, Any], step: int) -> str:
    history = obs.get("chat_history", [])
    history_text = "\n".join(
        f"  {m['role'].upper()}: {m['content']}" for m in history
    ) if history else f"  CUSTOMER: {obs['current_query']}"

    hint = obs.get("task_hint", "")
    return (
        f"Conversation so far:\n{history_text}\n\n"
        f"This is step {step} of the resolution process.\n"
        + (f"Hint: {hint}\n" if hint else "")
        + "\nWhat do you do next?"
    )


# ──────────────────────────────────────────────
# LLM → action conversion
# ──────────────────────────────────────────────

def call_llm(client: OpenAI, system: str, user: str) -> Dict[str, Any]:
    """Call the LLM and return a parsed JSON action dict."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    raw = response.choices[0].message.content or ""

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw   = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract first {...} block
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise ValueError(f"Could not parse LLM output as JSON:\n{raw}")


def get_action_for_task(
    client: OpenAI,
    task_id: str,
    obs: Dict[str, Any],
    step: int,
) -> Dict[str, Any]:
    """Route to the correct prompt template and get an action."""
    if task_id == "classify":
        return call_llm(client, SYSTEM_CLASSIFY, build_user_prompt_classify(obs))
    elif task_id == "respond":
        return call_llm(client, SYSTEM_RESPOND,  build_user_prompt_respond(obs))
    elif task_id == "resolve":
        return call_llm(client, SYSTEM_RESOLVE,  build_user_prompt_resolve(obs, step))
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


# ──────────────────────────────────────────────
# Environment HTTP helpers
# ──────────────────────────────────────────────

def env_reset(env_url: str, task_id: str, seed: int) -> Dict[str, Any]:
    resp = requests.post(
        f"{env_url}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(env_url: str, action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{env_url}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_health(env_url: str) -> bool:
    try:
        resp = requests.get(f"{env_url}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────

def run_episode(
    client: OpenAI,
    env_url: str,
    task_id: str,
    seed: int,
    episode_num: int,
) -> Dict[str, Any]:
    """Run one full episode and return summary metrics."""

    # ── Reset environment ─────────────────────
    reset_data = env_reset(env_url, task_id, seed)
    obs        = reset_data["observation"]

    log_start(episode_num, task_id, seed)

    cumulative_reward = 0.0
    step_num          = 0

    while True:
        step_num += 1

        # ── Get action from LLM ───────────────
        try:
            action = get_action_for_task(client, task_id, obs, step_num)
        except Exception as exc:
            # Fallback action on LLM failure
            print(f"[WARN] LLM error on step {step_num}: {exc}", file=sys.stderr)
            action = {"type": "respond", "message": "I apologize for any inconvenience. Let me help you resolve this issue."}

        # ── Send action to environment ────────
        step_data         = env_step(env_url, action)
        reward            = step_data["reward"]
        done              = step_data["done"]
        info              = step_data.get("info", {})
        obs               = step_data["observation"]
        cumulative_reward += reward
        feedback          = info.get("feedback", "")

        log_step(episode_num, step_num, action, reward, done, cumulative_reward, feedback)

        if done:
            break

        # Safety cap (should not be needed — env enforces max_steps)
        if step_num >= 10:
            break

    # Score = total normalised reward (already in [0,1] per step)
    # For single-step tasks this equals the step reward directly.
    # For multi-step tasks we cap the sum at 1.0.
    score = min(round(cumulative_reward, 4), 1.0)

    log_end(episode_num, task_id, cumulative_reward, score, step_num)

    return {
        "episode":      episode_num,
        "task_id":      task_id,
        "total_reward": round(cumulative_reward, 4),
        "score":        score,
        "steps":        step_num,
    }


# ──────────────────────────────────────────────
# Main entry-point
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline inference for AI Customer Support Env")
    parser.add_argument(
        "--env-url",
        default=os.getenv("ENV_URL", DEFAULT_ENV_URL),
        help="Base URL of the environment server (default: http://localhost:7860)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run per task (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    env_url  = args.env_url.rstrip("/")
    episodes = args.episodes

    # ── Wait for environment to be ready ──────
    print(f"[INFO] Connecting to environment at {env_url}", file=sys.stderr)
    for attempt in range(30):
        if env_health(env_url):
            print("[INFO] Environment is ready.", file=sys.stderr)
            break
        print(f"[INFO] Waiting for environment... attempt {attempt + 1}/30", file=sys.stderr)
        time.sleep(3)
    else:
        print("[ERROR] Environment did not become healthy in time.", file=sys.stderr)
        sys.exit(1)

    # ── Build LLM client ──────────────────────
    client = build_openai_client()

    # ── Run one episode per task ──────────────
    all_results: List[Dict[str, Any]] = []
    episode_counter = 0

    for task_id in EVAL_TASKS:
        for ep in range(episodes):
            episode_counter += 1
            seed = args.seed if args.seed is not None else TASK_SEEDS.get(task_id, episode_counter * 1000)
            seed += ep  # vary across multi-episode runs

            try:
                result = run_episode(client, env_url, task_id, seed, episode_counter)
                all_results.append(result)
            except Exception as exc:
                print(f"[ERROR] Episode {episode_counter} failed: {exc}", file=sys.stderr)
                # Log a zero-score entry so evaluation can continue
                log_end(episode_counter, task_id, 0.0, 0.0, 0)
                all_results.append({
                    "episode":      episode_counter,
                    "task_id":      task_id,
                    "total_reward": 0.0,
                    "score":        0.0,
                    "steps":        0,
                })

    # ── Aggregate summary ─────────────────────
    if all_results:
        avg_score = sum(r["score"] for r in all_results) / len(all_results)
        task_scores: Dict[str, list] = {}
        for r in all_results:
            task_scores.setdefault(r["task_id"], []).append(r["score"])

        summary = {
            "total_episodes":   len(all_results),
            "average_score":    round(avg_score, 4),
            "per_task_avg":     {
                tid: round(sum(scores) / len(scores), 4)
                for tid, scores in task_scores.items()
            },
            "model":     MODEL_NAME,
            "env_url":   env_url,
        }
        print(f"\n[SUMMARY] {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
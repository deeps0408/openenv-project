#!/usr/bin/env python3
"""
validate.py — Pre-submission validation script.

Runs all checklist items automatically and prints a pass/fail report.
Run this before submitting to Hugging Face.

Usage:
  python validate.py                          # tests localhost:7860
  python validate.py --env-url <URL>          # tests remote Space
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
import yaml

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results: List[Tuple[str, str, str]] = []   # (check_name, status, detail)


def record(name: str, passed: bool, detail: str = "", warn: bool = False) -> None:
    status = PASS if passed else (WARN if warn else FAIL)
    results.append((name, status, detail))
    icon = "✅" if passed else ("⚠️ " if warn else "❌")
    print(f"  {icon}  {name}: {detail}")


# ──────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────

def check_health(base: str) -> bool:
    try:
        r = requests.get(f"{base}/health", timeout=10)
        ok = r.status_code == 200
        record("GET /health returns 200", ok, f"status={r.status_code}")
        return ok
    except Exception as e:
        record("GET /health returns 200", False, str(e))
        return False


def check_reset(base: str) -> Dict[str, Any]:
    for task_id in ("classify", "respond", "resolve"):
        try:
            r = requests.post(
                f"{base}/reset",
                json={"task_id": task_id, "seed": 42},
                timeout=15,
            )
            ok = r.status_code == 200
            data = r.json() if ok else {}
            has_obs = "observation" in data
            has_tid = data.get("task_id") == task_id
            record(
                f"POST /reset task_id={task_id}",
                ok and has_obs and has_tid,
                f"status={r.status_code}, has_observation={has_obs}, task_id_match={has_tid}",
            )
        except Exception as e:
            record(f"POST /reset task_id={task_id}", False, str(e))

    # Return last reset data for subsequent checks
    try:
        r = requests.post(f"{base}/reset", json={"task_id": "classify", "seed": 42}, timeout=15)
        return r.json()
    except Exception:
        return {}


def check_step(base: str) -> None:
    # Step with a valid action
    for action, task_id, seed in [
        ({"type": "classify", "category": "billing"}, "classify", 1001),
        ({"type": "respond", "message": "We sincerely apologize for the delay. Our team is looking into your order immediately."}, "respond", 2001),
        ({"type": "respond", "message": "I understand you have been charged twice. This is a billing issue and I am here to help resolve it."}, "resolve", 3001),
    ]:
        try:
            requests.post(f"{base}/reset", json={"task_id": task_id, "seed": seed}, timeout=10)
            r = requests.post(f"{base}/step", json={"action": action}, timeout=15)
            ok = r.status_code == 200
            data = r.json() if ok else {}
            has_reward = "reward" in data
            reward = data.get("reward", -1)
            in_range = 0.0 <= reward <= 1.0 if has_reward else False
            record(
                f"POST /step task_id={task_id}",
                ok and has_reward and in_range,
                f"status={r.status_code}, reward={reward}, in_range={in_range}",
            )
        except Exception as e:
            record(f"POST /step task_id={task_id}", False, str(e))


def check_state(base: str) -> None:
    try:
        requests.post(f"{base}/reset", json={"task_id": "classify", "seed": 99}, timeout=10)
        r = requests.get(f"{base}/state", timeout=10)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        has_fields = all(k in data for k in ("episode_id", "task_id", "step_count", "done"))
        record("GET /state", ok and has_fields, f"status={r.status_code}, has_required_fields={has_fields}")
    except Exception as e:
        record("GET /state", False, str(e))


def check_openenv_yaml(base: str) -> None:
    try:
        r = requests.get(f"{base}/openenv.yaml", timeout=10)
        ok = r.status_code == 200
        if not ok:
            record("GET /openenv.yaml", False, f"status={r.status_code}")
            return
        spec = yaml.safe_load(r.text)
        has_name     = "name" in spec
        has_tasks    = "tasks" in spec and len(spec["tasks"]) >= 3
        has_endpoints = "endpoints" in spec
        has_action   = "action_space" in spec
        has_obs      = "observation_space" in spec
        all_ok = has_name and has_tasks and has_endpoints and has_action and has_obs
        record(
            "openenv.yaml valid",
            all_ok,
            f"name={has_name}, tasks={has_tasks}(≥3), endpoints={has_endpoints}, "
            f"action_space={has_action}, observation_space={has_obs}",
        )
    except Exception as e:
        record("openenv.yaml valid", False, str(e))


def check_reward_range(base: str) -> None:
    """Verify all tasks return rewards in [0.0, 1.0]."""
    tasks_actions = [
        ("classify", 1001, {"type": "classify", "category": "billing"}),
        ("classify", 1001, {"type": "classify", "category": "shipping"}),   # wrong → 0.0
        ("respond",  2001, {"type": "respond",  "message": "We apologize for the inconvenience with your order. Our team will investigate the delay and get back to you within 24 hours."}),
        ("resolve",  3001, {"type": "respond",  "message": "I understand the issue with billing. Let me help you resolve this problem."}),
    ]

    all_in_range = True
    for task_id, seed, action in tasks_actions:
        try:
            requests.post(f"{base}/reset", json={"task_id": task_id, "seed": seed}, timeout=10)
            r = requests.post(f"{base}/step", json={"action": action}, timeout=15)
            reward = r.json().get("reward", -1)
            in_range = 0.0 <= reward <= 1.0
            if not in_range:
                all_in_range = False
        except Exception:
            all_in_range = False

    record("All rewards in [0.0, 1.0]", all_in_range, "Checked 4 step results across tasks")


def check_typed_models(base: str) -> None:
    """Verify typed Pydantic validation (invalid action returns 422)."""
    try:
        requests.post(f"{base}/reset", json={"task_id": "classify", "seed": 1}, timeout=10)
        # Send malformed action (missing required 'type' field)
        r = requests.post(f"{base}/step", json={"action": {"category": "billing"}}, timeout=10)
        returns_error = r.status_code in (400, 422)
        record(
            "Typed models enforce validation",
            returns_error,
            f"Invalid action → HTTP {r.status_code} (expected 400 or 422)",
        )
    except Exception as e:
        record("Typed models enforce validation", False, str(e))


def check_three_tasks(base: str) -> None:
    """Verify three distinct task_ids exist and work."""
    tasks_found = []
    for task_id in ("classify", "respond", "resolve"):
        try:
            r = requests.post(
                f"{base}/reset", json={"task_id": task_id, "seed": 7}, timeout=10
            )
            if r.status_code == 200 and r.json().get("task_id") == task_id:
                tasks_found.append(task_id)
        except Exception:
            pass
    record(
        "3+ tasks implemented",
        len(tasks_found) >= 3,
        f"Found: {tasks_found}",
    )


def check_info(base: str) -> None:
    try:
        r = requests.get(f"{base}/info", timeout=10)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        record("GET /info", ok, f"status={r.status_code}, keys={list(data.keys())[:6]}")
    except Exception as e:
        record("GET /info", False, str(e))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenEnv pre-submission validator")
    p.add_argument("--env-url", default="http://localhost:7860", help="Environment base URL")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = args.env_url.rstrip("/")

    print(f"\n{'='*60}")
    print(f"  OpenEnv Pre-Submission Validator")
    print(f"  Target: {base}")
    print(f"{'='*60}\n")

    # Wait for server
    print("🔌 Connecting to environment server...")
    for i in range(15):
        try:
            if requests.get(f"{base}/health", timeout=5).status_code == 200:
                print("   → Connected!\n")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("   → Could not connect. Is the server running?\n")
        sys.exit(1)

    print("🔍 Running checks...\n")

    check_health(base)
    check_reset(base)
    check_step(base)
    check_state(base)
    check_openenv_yaml(base)
    check_reward_range(base)
    check_typed_models(base)
    check_three_tasks(base)
    check_info(base)

    # ── Summary ───────────────────────────────
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    warned = sum(1 for _, s, _ in results if s == WARN)
    total  = len(results)

    print(f"\n{'='*60}")
    print(f"  Validation Summary")
    print(f"{'='*60}")
    print(f"  Total checks : {total}")
    print(f"  Passed       : {passed} ✅")
    print(f"  Failed       : {failed} ❌")
    print(f"  Warnings     : {warned} ⚠️")
    print(f"{'='*60}")

    if failed == 0:
        print("\n🎉 All checks passed! Ready to submit.\n")
        sys.exit(0)
    else:
        print(f"\n⛔ {failed} check(s) failed. Fix before submitting.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
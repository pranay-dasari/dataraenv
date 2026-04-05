#!/usr/bin/env python3
"""
DataraEnv Agent Training Guide & Demo Script
=============================================

This script demonstrates the full agent workflow for DataraEnv.
It works entirely locally (no LLM needed) by using handcrafted
task-optimal actions, so you can understand the interaction loop
before connecting a real model.

Usage:
    # 1. Start the server (in another terminal)
    uvicorn datara_env.server:app --host 0.0.0.0 --port 8000

    # 2. Run this demo
    python scripts/demo_agent.py

Architecture Overview:
    ┌──────────┐   POST /reset    ┌────────────┐
    │  Agent   │ ───────────────► │  DataraEnv  │
    │ (you /   │   POST /step     │  (server)   │
    │  LLM)    │ ◄──────────────► │             │
    └──────────┘   GET /state     └────────────┘

    1. Agent calls POST /reset?task_id=... → gets initial observation
    2. Agent reads instructions + context from observation
    3. Agent crafts a JSON message and calls POST /step
    4. Environment grades the message, returns reward + feedback
    5. Agent improves and repeats until done=true or max_steps
"""

import json
import requests
import sys

ENV_URL = "http://localhost:8000"


# ═══════════════════════════════════════════════════════════════
#  TASK-OPTIMAL ACTIONS (what a perfect agent would submit)
# ═══════════════════════════════════════════════════════════════

OPTIMAL_ACTIONS = {
    "pii_masking_easy": {
        "columns": {
            "customer_id": {"is_pii": True, "masking_strategy": "tokenize"},
            "full_name":   {"is_pii": True, "masking_strategy": "hash"},
            "phone_number":{"is_pii": True, "masking_strategy": "tokenize"},
            "email":       {"is_pii": True, "masking_strategy": "tokenize"},
            "age":         {"is_pii": False, "masking_strategy": "none"},
            "city":        {"is_pii": False, "masking_strategy": "none"},
            "signup_at":   {"is_pii": False, "masking_strategy": "none"},
        }
    },
    "relational_config_medium": {
        "generation_order": ["customers", "accounts", "transactions"],
        "rows": {"customers": 5000, "accounts": 8000, "transactions": 100000},
        "dp_mode": "standard",
        "constraints": [
            "accounts.customer_id -> customers.customer_id",
            "transactions.account_id -> accounts.account_id",
        ],
    },
    "eval_review_hard": {
        "accept": False,
        "reasons": [
            "Fidelity score 0.88 is below the 0.9 threshold",
            "Data is not ready for production deployment",
        ],
        "remediation_steps": [
            "Improve fidelity by tuning distribution matching",
            "Regenerate synthetic data with stricter quality constraints",
        ],
    },
}


def banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def check_server():
    """Verify server is reachable."""
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=5)
        r.raise_for_status()
        print("✅ Server is healthy")
        return True
    except requests.ConnectionError:
        print("❌ Server is not running. Start it with:")
        print("   uvicorn datara_env.server:app --host 0.0.0.0 --port 8000")
        return False


def run_task(task_id: str, action_payload: dict):
    """Run a single task episode and print the interaction."""
    banner(f"Task: {task_id}")

    # ── Step 1: Reset ────────────────────────────────────
    print("📋 Resetting environment...")
    reset_resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    print(f"   Episode ID : {obs['episode_id']}")
    print(f"   Task ID    : {obs['task_id']}")
    print(f"   Max Steps  : 5")
    print(f"   Context    : {obs['context'][:80]}...")
    print()

    # ── Step 2: Agent reads instructions ─────────────────
    print("📖 Instructions (first 200 chars):")
    print(f"   {obs['instructions'][:200]}...")
    print()

    # ── Step 3: Agent submits action ─────────────────────
    message = json.dumps(action_payload)
    print(f"🤖 Agent submitting action ({len(message)} chars)...")
    print(f"   Preview: {message[:120]}...")
    print()

    step_resp = requests.post(
        f"{ENV_URL}/step",
        json={"message": message},
    )
    step_resp.raise_for_status()
    result = step_resp.json()

    # ── Step 4: Read reward and feedback ─────────────────
    obs = result["observation"]
    print(f"📊 Results:")
    print(f"   Score      : {obs['score_so_far']:.3f}")
    print(f"   Reward     : {result['reward']:.3f}")
    print(f"   Done       : {result['done']}")
    print(f"   Feedback   : {obs.get('feedback', 'N/A')}")
    print()

    # ── Step 5: Check final state ────────────────────────
    state_resp = requests.get(f"{ENV_URL}/state")
    state = state_resp.json()
    print(f"🔍 Final State:")
    print(f"   Step           : {state['step']}")
    print(f"   Cumulative     : {state['cumulative_reward']:.3f}")
    print(f"   History Length : {len(state['history'])}")

    return obs["score_so_far"]


def main():
    banner("DataraEnv Agent Training Demo")

    print("This demo shows the full agent workflow:\n")
    print("  1. Server starts (uvicorn)")
    print("  2. Agent calls POST /reset to get task instructions")
    print("  3. Agent reads context (schema/report) and crafts a JSON response")
    print("  4. Agent calls POST /step with {\"message\": \"<JSON>\"}")
    print("  5. Server grades the response and returns score + feedback")
    print("  6. Agent can iterate (up to max_steps) to improve score")
    print("  7. Episode ends when done=true or max_steps reached")
    print()

    if not check_server():
        sys.exit(1)

    scores = {}
    for task_id, action in OPTIMAL_ACTIONS.items():
        scores[task_id] = run_task(task_id, action)

    banner("Summary")
    for task_id, score in scores.items():
        emoji = "🟢" if score >= 0.9 else "🟡" if score >= 0.7 else "🔴"
        print(f"  {emoji} {task_id:30s}  score={score:.3f}")

    avg = sum(scores.values()) / len(scores)
    print(f"\n  📈 Overall Average: {avg:.3f}")

    banner("How to Train Your Own Agent")
    print("""
  The interaction loop for agent training is simple:

  ┌─ Training Loop ──────────────────────────────────────────┐
  │                                                          │
  │  for each episode:                                       │
  │    1. obs = POST /reset?task_id=<task>                   │
  │                                                          │
  │    for each step (up to max_steps):                      │
  │      2. Read obs.instructions + obs.context              │
  │      3. Call your LLM to generate a JSON response        │
  │      4. result = POST /step {"message": "<json>"}        │
  │      5. obs = result.observation                         │
  │      6. reward = result.reward                           │
  │                                                          │
  │      if result.done: break                               │
  │                                                          │
  │    7. Use final obs.score_so_far as episode return       │
  │    8. Update agent policy using reward signal             │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

  Key concepts:
  • message field: Always a JSON string matching task schema
  • score_so_far: Best score achieved (0-1), monotonically non-decreasing
  • reward: Improvement delta from this step (can be negative due to time penalty)
  • feedback: Text hint from grader to guide next attempt
  • done: True when score is high enough or max_steps reached

  Training approaches:
  ├── Supervised Fine-Tuning (SFT)
  │   • Collect expert trajectories (like this demo)
  │   • Fine-tune LLM on (instruction, context) → optimal JSON
  │
  ├── Reinforcement Learning (RL / RLHF)
  │   • Use score_so_far as environment reward
  │   • PPO/DPO over multi-step episodes
  │   • Agent learns to read feedback and improve iteratively
  │
  └── Prompt Engineering + Few-Shot
      • Use inference.py as a starting baseline
      • Add few-shot examples of correct JSON for each task
      • Use obs.feedback to refine the next prompt
""")


if __name__ == "__main__":
    main()

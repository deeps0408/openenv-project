# 🎯 AI Customer Support Training Environment

> An **OpenEnv-compliant reinforcement-learning environment** that trains AI agents to resolve real-world customer support issues — from issue classification to full multi-turn resolution.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://openenv.dev)
[![Python 3.11](https://img.shields.io/badge/python-3.11-brightgreen)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 What Problem Does This Solve?

Companies like Amazon, Flipkart, and Shopify rely on AI agents to handle thousands of customer interactions daily. Training such agents requires a controlled, replayable environment that:

- Generates diverse customer scenarios
- Rewards correct behaviour step-by-step (not just at the end)
- Evaluates responses objectively via rule-based graders
- Supports multi-turn interactions where the agent must gather information before acting

This project provides exactly that — a gym-style environment your agent can learn from.

---

## 🗂️ Project Structure

```
ai-customer-support-env/
├── main.py          # FastAPI server (all OpenEnv endpoints)
├── environment.py   # Core env class: reset() / step() / state()
├── models.py        # Typed Pydantic schemas (actions, observations, responses)
├── tasks.py         # Scenario datasets for all three tasks
├── graders.py       # Deterministic rule-based graders (score 0.0 – 1.0)
├── inference.py     # Baseline agent (uses OpenAI client, structured logs)
├── openenv.yaml     # OpenEnv specification
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Environment Overview

### Full Lifecycle (one episode)

```
env.reset(task_id="resolve", seed=42)
# → observation: customer query + chat history + task hints

while not done:
    action = agent.decide(observation)       # classify / respond / ask_question
    observation, reward, done, info = env.step(action)
    # reward ∈ [0.0, 1.0], partial progress signals at each step
```

### Internal State (hidden from agent)

```python
{
  "task_id":             "resolve",
  "seed":                42,
  "issue_identified":    False,
  "clarification_asked": False,
  "solution_given":      False,
  "bonus_given":         False,
  "last_agent_message":  ""
}
```

---

## 🧩 Tasks

### Task 1 — Issue Classification `(easy)`

| Property       | Value |
|----------------|-------|
| `task_id`      | `classify` |
| `max_steps`    | 1 |
| `score_range`  | `[0.0, 1.0]` |
| Action type    | `classify` |

**Scenario:** The agent receives a raw customer message and must label it with one of five categories: `billing`, `shipping`, `refund`, `technical`, `general`.

**Grader logic:**

```
score = 1.0   if category == correct_category
score = 0.0   otherwise
```

**Example:**

```json
Customer: "I was charged twice for my last order."

Action:  {"type": "classify", "category": "billing"}
Reward:  1.0  ✅

Action:  {"type": "classify", "category": "shipping"}
Reward:  0.0  ❌
```

---

### Task 2 — Response Generation `(medium)`

| Property       | Value |
|----------------|-------|
| `task_id`      | `respond` |
| `max_steps`    | 1 |
| `score_range`  | `[0.0, 1.0]` |
| Action type    | `respond` |

**Scenario:** The agent receives a customer complaint and must write a response scored on four dimensions:

| Dimension       | Max Score | Criteria |
|-----------------|-----------|----------|
| **Relevance**   | 0.40 | Contains at least one required keyword for the issue type |
| **Politeness**  | 0.30 | ≥2 politeness words: sorry, apologize, thank you, etc. |
| **Length**      | 0.20 | Response is ≥20 words |
| **Actionable**  | 0.10 | Mentions a concrete next step or resolution |

**Example:**

```json
Customer: "My order has been delayed for 5 days!"

Action: {
  "type": "respond",
  "message": "We sincerely apologize for the delay with your order. We understand how frustrating this must be. Our team is actively investigating the shipment and will provide an update within 24 hours."
}
Reward: 1.0  ✅  (relevant + polite + long enough + actionable)
```

---

### Task 3 — Full Issue Resolution `(hard)`

| Property       | Value |
|----------------|-------|
| `task_id`      | `resolve` |
| `max_steps`    | 5 |
| `score_range`  | `[0.0, 1.0]` |
| Action types   | `respond`, `ask_question` |

**Scenario:** A multi-turn episode. The agent must complete three milestones in order:

```
Step 1 → Identify the issue          (+0.20)
Step 2 → Ask a clarifying question   (+0.20)
Step 3 → Provide a concrete solution (+0.30)
         Completion bonus             (+0.30)
         ────────────────────────────────────
         Maximum total               = 1.0
```

**Penalties:**
- Empty message: `+0.00`
- Repeated message: `+0.00`
- Skipping milestones: lower cumulative reward

**Example interaction:**

```
CUSTOMER: "I got charged twice for my order. This is unacceptable!"

Step 1 (respond):     "I understand you've been double-charged. This is a billing
                       issue and I'm here to help resolve it immediately."
                       → reward: 0.20 ✅  (issue identified)

Step 2 (ask_question): "Could you please share your order ID or transaction date
                        so I can look into this?"
                       → reward: 0.20 ✅  (clarification asked)

Step 3 (respond):     "Thank you for the details. I've initiated a full refund for
                        the duplicate charge. You'll see it in 3–5 business days."
                       → reward: 0.30 + 0.30 bonus = 0.60 ✅  (solution + completion)

Total episode score: 1.0 🎉
```

---

## 📐 Action Space

```yaml
type: object
properties:
  type:
    type: string
    enum: [classify, respond, ask_question]
  category:
    type: string
    enum: [billing, shipping, refund, technical, general]
    optional: true    # required when type = classify
  message:
    type: string
    optional: true    # required when type = respond or ask_question
```

---

## 👁️ Observation Space

```yaml
type: object
properties:
  current_query:   string     # latest customer message
  chat_history:    array      # [{role: "customer"|"agent", content: "..."}]
  task_type:       string     # easy | medium | hard
  step_count:      integer    # steps taken so far
  max_steps:       integer    # episode step limit
  task_hint:       string     # coaching hint (optional)
```

---

## 🏆 Reward Design

All rewards are normalised to `[0.0, 1.0]`. The reward function is designed to provide **partial progress signals** at every step, not just at termination.

| Task     | Signal type    | Key insight |
|----------|----------------|-------------|
| Classify | Binary         | All-or-nothing; encourages precision |
| Respond  | Rubric-based   | Partial credit for each quality dimension |
| Resolve  | Milestone-based | Cumulative; agent is rewarded for each correct step |

This design prevents reward hacking (skipping steps) while still encouraging the agent to make forward progress every turn.

---

## 🚀 Setup & Usage

### Option 1 — Local Python

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the environment server
python main.py
# → Runs on http://localhost:7860

# 3. Run the baseline agent (in a separate terminal)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-your-key-here"

python inference.py
```

### Option 2 — Docker

```bash
# Build image
docker build -t ai-support-env .

# Run the environment server
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-your-key-here" \
  ai-support-env

# Run inference against the dockerised environment
python inference.py --env-url http://localhost:7860
```

### Option 3 — Docker Compose (env + inference together)

```bash
docker compose up
```

---

## 📡 API Reference

All endpoints are documented interactively at `http://localhost:7860/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness probe (returns 200 + "OK") |
| `GET`  | `/`       | Environment metadata |
| `GET`  | `/info`   | Detailed action/observation space |
| `GET`  | `/openenv.yaml` | OpenEnv YAML spec |
| `POST` | `/reset`  | Start new episode |
| `POST` | `/step`   | Execute one action |
| `GET`  | `/state`  | Full internal state (debug) |

### `/reset` Request

```json
{
  "task_id": "classify",   // "classify" | "respond" | "resolve" | null (random)
  "seed": 1001             // optional, for reproducibility
}
```

### `/step` Request

```json
{
  "action": {
    "type": "classify",
    "category": "billing"
  }
}
```

### `/step` Response

```json
{
  "observation": { "current_query": "...", "chat_history": [], "task_type": "easy", ... },
  "reward": 1.0,
  "done": true,
  "info": {
    "episode_id": "uuid",
    "task_id": "classify",
    "step": 1,
    "feedback": "✔ Correct! Category 'billing' matches expected 'billing'.",
    "cumulative_reward": 1.0,
    "milestones": {}
  }
}
```

---

## 📊 Baseline Scores

Produced by `inference.py` with `MODEL_NAME=gpt-4o-mini`:

| Task     | Difficulty | Expected Score | Notes |
|----------|------------|----------------|-------|
| classify | Easy       | ~0.85          | Occasional misclassification of ambiguous queries |
| respond  | Medium     | ~0.80          | Usually hits relevance + politeness; sometimes misses keywords |
| resolve  | Hard       | ~0.70          | Completes 2–3 milestones; occasionally skips clarification |

> Re-run with `python inference.py --seed 42` for reproducible scores.

---

## 📋 Structured Log Format

`inference.py` emits logs to stdout in this exact format (required by the evaluator):

```
[START] {"episode": 1, "task_id": "classify", "seed": 1001}
[STEP]  {"episode": 1, "step": 1, "action": {"type": "classify", "category": "billing"}, "reward": 1.0, "done": true, "cumulative_reward": 1.0, "feedback": "✔ Correct!"}
[END]   {"episode": 1, "task_id": "classify", "total_reward": 1.0, "score": 1.0, "steps": 1}

[START] {"episode": 2, "task_id": "respond", "seed": 2001}
[STEP]  {"episode": 2, "step": 1, "action": {...}, "reward": 0.8, "done": true, "cumulative_reward": 0.8, "feedback": "..."}
[END]   {"episode": 2, "task_id": "respond", "total_reward": 0.8, "score": 0.8, "steps": 1}

[SUMMARY] {"total_episodes": 3, "average_score": 0.85, "per_task_avg": {...}, ...}
```

---

## 🌍 Environment Variables

| Variable      | Required | Description |
|---------------|----------|-------------|
| `API_BASE_URL` | ✅ | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME`   | ✅ | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN`     | ✅ | Hugging Face / API key |
| `PORT`         | ❌ | Server port (default: `7860`) |
| `ENV_URL`      | ❌ | Override for inference.py (default: `http://localhost:7860`) |

---

## 🐳 Hugging Face Spaces Deployment

1. Create a new **Space** with Docker runtime.
2. Tag it with `openenv`.
3. Push all project files.
4. Set Secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
5. The Space will auto-build and expose port 7860.

---

## 📝 Pre-Submission Checklist

- [x] Hugging Face Space deployment with `openenv` tag
- [x] `/health` returns HTTP 200
- [x] `/reset` starts a valid episode
- [x] `openenv.yaml` present and valid
- [x] Typed Pydantic models for all actions and observations
- [x] `step()` / `reset()` / `state()` endpoints implemented
- [x] Dockerfile builds cleanly with `docker build .`
- [x] `inference.py` runs without errors
- [x] 3 tasks implemented with graders scoring 0.0–1.0
- [x] Structured `[START]` / `[STEP]` / `[END]` logs emitted
- [x] Runtime < 20 minutes on vcpu=2, memory=8GB

---

## 📄 License

MIT License — see `LICENSE` for details.
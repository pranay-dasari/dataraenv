---
title: DataraEnv Demo
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# DataraEnv — Synthetic Data Engineering Environment

**DataraEnv** is an [OpenEnv](https://openenv.dev)-compliant reinforcement learning environment that simulates real-world synthetic data engineering tasks under India's **DPDP (Digital Personal Data Protection) Act, 2023**.

Data engineers at enterprises across India and APAC perform these tasks daily: identifying PII fields, configuring synthetic data generation pipelines, and reviewing quality gates before deployment. DataraEnv turns those real workflows into a structured benchmark so AI agents can be evaluated — and trained — on them.

---

## Why this environment exists

Enterprises cannot use raw customer data for AI training, analytics, or testing under DPDP. Synthetic data is the compliant path — but generating it correctly requires domain knowledge: which fields are PII, how to preserve relational integrity, when data quality is good enough to deploy.

There is currently no public RL benchmark that tests these skills. DataraEnv fills that gap.

---

## Environment overview

DataraEnv is a **FastAPI HTTP server** that exposes the OpenEnv interface over REST. An AI agent interacts with it in a loop:

1. `POST /reset` — start a new episode, receive task instructions and schema context
2. `POST /step` — submit a JSON answer, receive a score (0–1), feedback, and done signal
3. `GET /state` — inspect internal episode state at any time

Each episode has a maximum of **5 steps**. The reward system tracks the agent's best score and returns the improvement delta each step, with a small time penalty (−0.01) to encourage efficiency.

```
Agent  ──POST /reset──►  DataraEnv
       ◄── observation ──
       ──POST /step ────►  DataraEnv  ──► grader ──► score + feedback
       ◄── obs + reward ──
       (repeat up to 5 steps or until done=True)
```

---

## Observation space

Every API response returns a `DataraObservation` object:

```json
{
  "episode_id": "uuid-string",
  "task_id": "pii_masking_easy",
  "instructions": "You are a data engineer working with Datara...",
  "context": "Table: customers\nColumns:\n- customer_id: ...",
  "step": 0,
  "done": false,
  "score_so_far": 0.0,
  "messages": [],
  "feedback": null
}
```

| Field | Type | Description |
|---|---|---|
| `episode_id` | string | UUID for the current episode |
| `task_id` | string | Which task is active |
| `instructions` | string | Full task prompt shown to the agent |
| `context` | string | Schema, report, or data the agent must analyze |
| `step` | int | Current step index (0-based) |
| `done` | bool | Whether the episode has ended |
| `score_so_far` | float [0,1] | Best score achieved so far in this episode |
| `messages` | list[str] | History of all agent inputs this episode |
| `feedback` | string | Grader feedback from the last step |

---

## Action space

Agents submit a `DataraAction` object to `POST /step`:

```json
{
  "message": "{\"columns\": {\"customer_id\": {\"is_pii\": true, \"masking_strategy\": \"tokenize\"}}}"
}
```

| Field | Type | Description |
|---|---|---|
| `message` | string | The agent's answer as a JSON string (primary field) |
| `action_type` | string | Optional: type label for structured actions |
| `config` | dict | Optional: structured config parameters |
| `rationale` | string | Optional: agent's reasoning (logged, not graded) |

> **Important:** The `message` field must contain your JSON answer as a **string** (double-encoded). Use `json.dumps(your_answer)` in Python before setting this field.

---

## Reward function

The reward system is designed to prevent score inflation while rewarding genuine improvement:

```
current_score         = grader output [0.0, 1.0]
best_score            = max(previous_best, current_score)
reward                = (best_score − previous_best) − 0.01
```

- **Incremental**: every step returns a reward, not just the final step
- **Best-score tracking**: submitting the same answer twice earns 0 improvement → net negative reward due to time penalty
- **Time penalty**: −0.01 per step encourages solving tasks in fewer attempts
- **Score clamped**: `score_so_far` is always in [0.0, 1.0]

---

## Tasks

Three tasks spanning easy → medium → hard difficulty. All are modeled on real DPDP compliance workflows.

### Task 1 — PII Masking `pii_masking_easy`

**Difficulty:** Easy | **Max steps:** 5 | **Done when:** score ≥ 0.95

**Objective:** Identify PII columns in a customer table and assign DPDP-safe masking strategies.

**Context given to agent:**
```
Table: customers
Columns:
- customer_id: Unique identifier for each customer
- full_name: Customer's full name
- phone_number: Customer's phone number
- email: Customer's email address
- age: Customer's age in years
- city: Customer's city of residence
- signup_at: When customer signed up (timestamp)
```

**Expected response format:**
```json
{
  "columns": {
    "customer_id":  {"is_pii": true,  "masking_strategy": "tokenize"},
    "full_name":    {"is_pii": true,  "masking_strategy": "hash"},
    "phone_number": {"is_pii": true,  "masking_strategy": "tokenize"},
    "email":        {"is_pii": true,  "masking_strategy": "tokenize"},
    "age":          {"is_pii": false, "masking_strategy": "none"},
    "city":         {"is_pii": false, "masking_strategy": "none"},
    "signup_at":    {"is_pii": false, "masking_strategy": "none"}
  }
}
```

**Available masking strategies:** `tokenize`, `hash`, `generalize`, `none`

**Grading (14 points total, normalized to [0,1]):**
- +1 point per column for correct `is_pii` classification
- +1 point per column for appropriate masking strategy (PII columns accept any of `tokenize`, `hash`, `generalize`; non-PII must be `none`)

---

### Task 2 — Relational Config `relational_config_medium`

**Difficulty:** Medium | **Max steps:** 5 | **Done when:** score ≥ 0.90

**Objective:** Design a synthetic data generation config for a 3-table relational schema, preserving primary key / foreign key relationships.

**Schema:**
```
customers (customer_id PK)
  └── accounts (account_id PK, customer_id FK)
        └── transactions (transaction_id PK, account_id FK)
```

**Expected response format:**
```json
{
  "generation_order": ["customers", "accounts", "transactions"],
  "rows": {"customers": 5000, "accounts": 8000, "transactions": 100000},
  "dp_mode": "standard",
  "constraints": [
    "accounts.customer_id -> customers.customer_id",
    "transactions.account_id -> accounts.account_id"
  ]
}
```

**Grading breakdown:**

| Component | Points | Criteria |
|---|---|---|
| Generation order | 0.4 | Exact match: customers → accounts → transactions |
| Generation order | 0.2 | Partial: correct dependency direction |
| Row ratios | 0.2 | accounts ≥ customers × 1.2 AND transactions ≥ accounts × 10 |
| Constraints | 0.2 | Both FK constraints present and exact |
| Constraints | 0.1 | Partial: 2 constraints with correct `->` format |
| DP mode: `standard` | 0.2 | Full credit — standard mode is valid |
| DP mode: `dp` | 0.1 + 0.1 | Base credit + epsilon in [1.0, 10.0] |

---

### Task 3 — Evaluation Review `eval_review_hard`

**Difficulty:** Hard | **Max steps:** 5 | **Done when:** score ≥ 0.80

**Objective:** Review a synthetic data quality report and make a deployment decision with justification.

**Quality thresholds:**
| Metric | Threshold | Direction |
|---|---|---|
| `fidelity.avg` | ≥ 0.90 | higher is better |
| `correlation_similarity` | ≥ 0.85 | higher is better |
| `constraint_pass_rate` | ≥ 0.95 | higher is better |
| `privacy_risk` | ≤ 0.20 | lower is better |

**Report given to agent (in `context` field):**
```json
{
  "fidelity": {"avg": 0.88},
  "correlation_similarity": 0.86,
  "constraint_pass_rate": 0.97,
  "privacy_risk": 0.18
}
```

**Expected response format:**
```json
{
  "accept": false,
  "reasons": ["Fidelity score 0.88 is below the 0.9 threshold"],
  "remediation_steps": [
    "Improve distribution matching to raise fidelity",
    "Regenerate data with stricter quality constraints"
  ]
}
```

> Note: `fidelity.avg = 0.88` fails the ≥ 0.90 threshold. The correct decision is **reject**.

**Grading breakdown:**

| Component | Points | Criteria |
|---|---|---|
| Accept/reject decision | 0.5 | Correct call based on thresholds |
| Reasons | 0.25 | Mentions failing metric keywords (fidelity, correlation, privacy) |
| Remediation | 0.25 | Mentions fidelity/distribution fix + regeneration |

---

## Reward function summary across all tasks

| Task | Done threshold | Max steps | Key partial-credit mechanism |
|---|---|---|---|
| PII Masking | score ≥ 0.95 | 5 | Per-column scoring (2 pts each) |
| Relational Config | score ≥ 0.90 | 5 | Component-wise (order + rows + constraints + DP) |
| Eval Review | score ≥ 0.80 | 5 | Decision + reasoning + remediation separately scored |

---

## Setup and usage

### Prerequisites

- Python 3.10+
- An API key: either `HF_TOKEN` (HuggingFace, free tier available) or `OPENAI_API_KEY`

### Local development

```bash
# Clone and install
git clone <repo-url>
cd datara-openenv
pip install -r requirements.txt

# Start the environment server
uvicorn datara_env.server:app --host 0.0.0.0 --port 8000

# Verify it's running
curl http://localhost:8000/health
# → {"status": "healthy", "environment": "DataraEnv"}
```

### Run the baseline agent

```bash
# Option A: HuggingFace (free, recommended for hackathon)
export HF_TOKEN="hf_..."
export MODEL_NAME="Qwen/Qwen3-32B"
export LLM_BASE_URL="https://router.huggingface.co/v1"

# Option B: OpenAI
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"
export LLM_BASE_URL="https://api.openai.com/v1"

# Run evaluation
python inference.py
```

### Docker deployment

```bash
# Build
docker build -t datara-openenv .

# Run
docker run -p 8000:8000 \
  -e HF_TOKEN="hf_..." \
  datara-openenv
```

### Manual interaction (without an LLM)

You can interact with the environment directly using curl:

```bash
# Step 1: Start an episode
curl -X POST "http://localhost:8000/reset?task_id=pii_masking_easy"

# Step 2: Read instructions and context from the response, then submit your answer
curl -X POST "http://localhost:8000/step" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "{\"columns\":{\"customer_id\":{\"is_pii\":true,\"masking_strategy\":\"tokenize\"},\"full_name\":{\"is_pii\":true,\"masking_strategy\":\"hash\"},\"phone_number\":{\"is_pii\":true,\"masking_strategy\":\"tokenize\"},\"email\":{\"is_pii\":true,\"masking_strategy\":\"tokenize\"},\"age\":{\"is_pii\":false,\"masking_strategy\":\"none\"},\"city\":{\"is_pii\":false,\"masking_strategy\":\"none\"},\"signup_at\":{\"is_pii\":false,\"masking_strategy\":\"none\"}}}"
  }'

# Step 3: Check internal state
curl http://localhost:8000/state
```

### Run the demo agent (no API key needed)

```bash
# Start the server first, then in another terminal:
python demo_agent.py
```

The demo agent uses hardcoded optimal answers for all three tasks. Useful for verifying the server is working correctly and understanding the expected response formats.

---

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | one of these two | — | HuggingFace API token (preferred for free usage) |
| `OPENAI_API_KEY` | one of these two | — | OpenAI API key |
| `LLM_BASE_URL` | no | `https://router.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | no | `Qwen/Qwen3-32B` | Model to use for inference |
| `DATARA_ENV_URL` | no | `http://localhost:8000` | DataraEnv server URL |
| `MAX_STEPS` | no | `5` | Max steps per episode |
| `TEMPERATURE` | no | `0.1` | LLM sampling temperature |
| `EPISODES_PER_TASK` | no | `3` | How many episodes to run per task |
| `DATARA_SEED` | no | — | Integer seed for deterministic random task selection |

---

## API endpoints

| Method | Endpoint | Description | Returns |
|---|---|---|---|
| `POST` | `/reset?task_id=<id>` | Start a new episode. `task_id` is optional — random if omitted | `DataraObservation` |
| `POST` | `/step` | Submit one action | `StepResponse` (observation + reward + done + info) |
| `GET` | `/state` | Current internal episode state | `DataraState` |
| `GET` | `/health` | Health check | `{"status": "healthy"}` |
| `GET` | `/` | API info and task list | info JSON |

**Valid task IDs:** `pii_masking_easy`, `relational_config_medium`, `eval_review_hard`

---

## Baseline performance scores

Evaluated using `Qwen/Qwen3-32B` via HuggingFace router, 3 episodes per task, `MAX_STEPS=5`, `TEMPERATURE=0.1`:

```
Model: Qwen/Qwen3-32B
Episodes per task: 3
--------------------------------------------------
pii_masking_easy:         avg_score = 0.857
  Individual scores:      [0.857, 0.857, 0.857]

relational_config_medium: avg_score = 0.733
  Individual scores:      [0.700, 0.800, 0.700]

eval_review_hard:         avg_score = 0.650
  Individual scores:      [0.600, 0.700, 0.650]
--------------------------------------------------
Overall average:          0.747  across 9 episodes
```

To reproduce:
```bash
export HF_TOKEN="hf_..."
export EPISODES_PER_TASK=3
python inference.py
```

---

## Running tests

```bash
pip install pytest
pytest test_graders.py -v
```

The test suite covers:
- Correct answers score 1.0 and terminate
- Partial answers receive partial credit
- Invalid JSON returns 0 and allows retry (`done=False`)
- Repeated identical submissions do not inflate scores
- Episodes terminate at `max_steps` regardless of score
- `state()` and `step()` raise `ValueError` before `reset()` is called

---

## Project structure

```
datara-openenv/
├── datara_env/
│   ├── __init__.py          # Package root
│   ├── models.py            # Pydantic schemas: Observation, Action, State, Reward
│   ├── tasks.py             # Task configs + graders (easy / medium / hard)
│   ├── environment.py       # DataraEnv core: reset() / step() / state()
│   └── server.py            # FastAPI app: /reset, /step, /state, /health
├── openenv.yaml             # OpenEnv metadata and runtime config
├── inference.py             # Baseline agent using OpenAI-compatible client
├── demo_agent.py            # Deterministic demo agent (no LLM required)
├── test_graders.py          # Pytest test suite
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container for HF Spaces + local docker
└── README.md                # This file
```

---

## Architecture

```
inference.py (agent)
      │
      │  POST /reset, POST /step
      ▼
server.py (FastAPI)
      │
      ▼
environment.py (DataraEnv)
      ├── tasks.py → load_task()      picks TaskConfig by task_id
      └── tasks.py → grade_message()  routes to correct grader
                          │
                          ├── grade_pii_masking()
                          ├── grade_relational_config()
                          └── grade_eval_review()
```
```
datara-openenv/
├── datara_env/
│   ├── __init__.py
│   ├── models.py           # Pydantic Action / Observation / State / Reward
│   ├── tasks.py            # Task configs + graders (easy / medium / hard)
│   ├── environment.py      # DataraEnv (core OpenEnv Environment)
│   └── server.py           # FastAPI app exposing /reset, /step, /state
├── openenv.yaml            # OpenEnv metadata and runtime config
├── inference.py            # Baseline agent using OpenAI client
├── requirements.txt        # Python dependencies
├── Dockerfile              # For HF Space + local docker
└── README.md               # This file
```

---

## OpenEnv validation

```bash
openenv validate openenv.yaml
```

This checks required fields, endpoint structure, task definitions, and Docker configuration.

---

## Roadmap (personal project extensions)

- [ ] Add 10+ task schemas across healthcare, fintech, e-commerce industries
- [ ] Randomize schemas per episode (column names, table structures) to prevent memorization
- [ ] Add consent management tasks (DPDP Chapter II compliance)
- [ ] Add right-to-erasure tasks (cascade delete config)
- [ ] Add cross-border data transfer decision tasks
- [ ] Multi-session persistence (episode history across server restarts)
- [ ] Leaderboard endpoint for multi-agent comparison
- [ ] Replace keyword-matching graders with LLM-as-judge for open-ended answers

---

## License

This project is base version 1.0.0 for datara
This project is part of the OpenEnv hackathon and follows competition guidelines.
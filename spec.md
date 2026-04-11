
## 1. Repository layout

Target structure (Python, FastAPI, OpenEnv-style):

```text
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
└── README.md               # Docs per spec
```

Python version: **3.10+ (ideally 3.11)**

***

## 2. Core environment concept

**Name:** `datara-openenv`  
**Env class:** `DataraEnv`  

**Real-world domain:**  
Simulate a **data / ML engineer** using a synthetic data platform (Datara) to:

1. Identify PII fields and choose DPDP‑safe masking strategies.  
2. Configure synthetic data generation for relational schemas (PK/FK, row counts, DP settings).  
3. Evaluate synthetic data quality/privacy reports and decide to accept/reject with remediation steps.

This matches OpenEnv’s “production-oriented, non-toy” requirement. [turing](https://www.turing.com/blog/evaluating-tool-using-agents-in-production-oriented-environments-with-openenv)

***

## 3. Pydantic models (Action, Observation, State, Reward)

File: `datara_env/models.py`

### 3.1 Base models

Assume OpenEnv provides abstract base classes or type protocols; if not, these can be plain Pydantic models but must be used consistently.

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DataraObservation(BaseModel):
    """
    Observation shown to the agent at each step.
    """
    task_id: str                     # e.g. "pii_masking_easy"
    instructions: str                # high-level instructions for the task
    context: str                     # current context: schema / report / text
    step: int                        # current step index (starting at 0)
    done: bool                       # whether the episode has ended
    score_so_far: float = Field(0.0, ge=0.0, le=1.0)
    messages: List[str] = []         # optional: previous agent messages or hints


class DataraAction(BaseModel):
    """
    Action provided by the agent. For now, a single textual message.
    Typically JSON or structured text.
    """
    message: str


class DataraState(BaseModel):
    """
    Internal state of the environment for debugging and state() calls.
    """
    episode_id: str
    task_id: str
    step: int
    cumulative_reward: float
    max_steps: int


class DataraReward(BaseModel):
    """
    Typed reward model (competition asks for typed Reward).
    """
    value: float
    done: bool
    info: Dict[str, Any] = {}
```

These models must be used for:

- HTTP request/response typing in FastAPI  
- Internal state tracking  
- OpenEnv validation (typed models).

***

## 4. Tasks & graders

File: `datara_env/tasks.py`

We define a common **TaskConfig** and three tasks: **easy**, **medium**, **hard**.

### 4.1 TaskConfig interface

```python
from typing import Tuple, Callable
from pydantic import BaseModel
from .models import DataraState


class TaskConfig(BaseModel):
    id: str
    difficulty: str                  # "easy" | "medium" | "hard"
    instructions: str
    initial_context: str
    max_steps: int
    grader_name: str                 # for logging/debug
    # grader: message x state -> (reward_delta, done, updated_context)
    # implemented as a function in code, not a Pydantic field
```

Implementation detail: keep actual grader functions as plain functions; `TaskConfig` holds metadata.

### 4.2 Task 1 – PII & masking (easy)

**Scenario:** Single table schema (`customers`) with columns; agent must mark PII and choose masking.

- Columns example: `customer_id`, `full_name`, `phone_number`, `email`, `age`, `city`, `signup_at`.  
- Agent responds with JSON like:

```json
{
  "columns": {
    "customer_id": {"is_pii": true, "masking_strategy": "tokenize"},
    "full_name": {"is_pii": true, "masking_strategy": "hash"},
    "phone_number": {"is_pii": true, "masking_strategy": "tokenize"},
    "email": {"is_pii": true, "masking_strategy": "tokenize"},
    "age": {"is_pii": false, "masking_strategy": "none"},
    "city": {"is_pii": false, "masking_strategy": "none"},
    "signup_at": {"is_pii": false, "masking_strategy": "none"}
  }
}
```

**Grader logic:**

- Ground-truth dict with `is_pii` and allowed `masking_strategy` for each column.  
- Parse JSON; if invalid JSON → strong negative reward and may terminate.  
- For each column: +1 for correct `is_pii`, +1 for acceptable `masking_strategy`.  
- Normalize to  by dividing by (2 * num_columns). [grandviewresearch](https://www.grandviewresearch.com/horizon/outlook/synthetic-data-generation-market/india)
- Provide partial reward every time the agent sends JSON (not only final step).  
- Episode ends when `max_steps` reached or when agent has produced a high enough score (e.g. ≥ 0.95).

### 4.3 Task 2 – Relational config (medium)

**Scenario:** 3-table relational schema (`customers`, `accounts`, `transactions`) with PK/FK.

Agent outputs JSON like:

```json
{
  "generation_order": ["customers", "accounts", "transactions"],
  "rows": {
    "customers": 5000,
    "accounts": 8000,
    "transactions": 100000
  },
  "dp_mode": "dp",
  "dp_epsilon": 5.0,
  "constraints": [
    "accounts.customer_id -> customers.customer_id",
    "transactions.customer_id -> customers.customer_id"
  ]
}
```

**Grader criteria:**

- `generation_order` respects dependencies (parents before children).  
- `rows` follow sensible ratios:  
  - `accounts >= customers`  
  - `transactions >= accounts`  
- `constraints` list exactly matches known PK/FK relationships.  
- `dp_mode` is either `"standard"` or `"dp"`; if `"dp"`, check `dp_epsilon` in e.g. [1.0, 10.0].  

Scoring approach:

- 0.4 for correct ordering  
- 0.2 for row ratio sanity  
- 0.2 for exact constraints  
- 0.2 for DP config correctness  

Return reward in; allow partial progress at each step. Terminate when score stabilizes or steps exhausted. [grandviewresearch](https://www.grandviewresearch.com/horizon/outlook/synthetic-data-generation-market/india)

### 4.4 Task 3 – Evaluation & decision (hard)

**Scenario:** Agent receives a **synthetic data evaluation report** in the `context` field. Precomputed JSON embedded as text:

- `fidelity`: per-column distribution similarity, average value [grandviewresearch](https://www.grandviewresearch.com/horizon/outlook/synthetic-data-generation-market/india)
- `correlation_similarity`: [grandviewresearch](https://www.grandviewresearch.com/horizon/outlook/synthetic-data-generation-market/india)
- `constraint_pass_rate`: [grandviewresearch](https://www.grandviewresearch.com/horizon/outlook/synthetic-data-generation-market/india)
- `privacy_risk`:, lower is better [grandviewresearch](https://www.grandviewresearch.com/horizon/outlook/synthetic-data-generation-market/india)

Example (stringified JSON in `context`):

```json
{
  "fidelity": {"avg": 0.88},
  "correlation_similarity": 0.86,
  "constraint_pass_rate": 0.97,
  "privacy_risk": 0.18
}
```

Agent responds with:

```json
{
  "accept": true,
  "reasons": [
    "High column fidelity and correlation similarity",
    "Constraints mostly respected",
    "Privacy risk acceptable for DP mode"
  ],
  "remediation_steps": []
}
```

or

```json
{
  "accept": false,
  "reasons": [
    "High privacy risk despite good fidelity"
  ],
  "remediation_steps": [
    "Enable differential privacy mode",
    "Lower epsilon from 8.0 to 3.0",
    "Regenerate with stricter constraints"
  ]
}
```

**Grader:**

- Apply fixed thresholds, e.g.:  
  - `fidelity.avg >= 0.9`  
  - `correlation_similarity >= 0.85`  
  - `constraint_pass_rate >= 0.95`  
  - `privacy_risk <= 0.2`  
- If all pass → correct decision is `accept: true`.  
- If any critical metric fails → correct is `accept: false`.  
- Base score 0.5 for correct `accept` / `reject`.  
- Add up to +0.25 if `reasons` mention the correct metrics (keywords: “fidelity”, “correlation”, “privacy risk”).  
- Add up to +0.25 if `remediation_steps` are appropriate (e.g. lowering epsilon, enabling DP, regening data).  

Total score per action in; partial progress if the agent refines their decision across steps. [grandviewresearch](https://www.grandviewresearch.com/horizon/outlook/synthetic-data-generation-market/india)

### 4.5 Task factory

Implement:

```python
from typing import Dict
from .models import DataraState
from .tasks_definitions import (
    pii_masking_task_config,
    relational_config_task_config,
    eval_review_task_config,
)

TASK_CONFIGS: Dict[str, TaskConfig] = {
    "pii_masking_easy": pii_masking_task_config,
    "relational_config_medium": relational_config_task_config,
    "eval_review_hard": eval_review_task_config,
}


def load_task(task_id: str | None) -> TaskConfig:
    if task_id is None:
        # simple random choice or rotate across tasks
        import random
        return random.choice(list(TASK_CONFIGS.values()))
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return TASK_CONFIGS[task_id]
```

Each `TaskConfig` is paired with a `grade(message: str, state: DataraState)` function (can be implemented as `if state.task_id == ...: grade_X` in `tasks.py`).

***

## 5. Environment implementation

File: `datara_env/environment.py`

### 5.1 Environment skeleton

```python
from typing import Tuple, Dict, Any, Optional
from uuid import uuid4

from .models import DataraObservation, DataraAction, DataraState, DataraReward
from .tasks import TaskConfig, load_task


class DataraEnv:
    """
    Core OpenEnv environment implementing reset/step/state.
    """

    def __init__(self):
        self._state: Optional[DataraState] = None
        self._task_config: Optional[TaskConfig] = None

    def reset(self, task_id: Optional[str] = None) -> DataraObservation:
        self._task_config = load_task(task_id)
        self._state = DataraState(
            episode_id=str(uuid4()),
            task_id=self._task_config.id,
            step=0,
            cumulative_reward=0.0,
            max_steps=self._task_config.max_steps,
        )

        return DataraObservation(
            task_id=self._task_config.id,
            instructions=self._task_config.instructions,
            context=self._task_config.initial_context,
            step=0,
            done=False,
            score_so_far=0.0,
            messages=[],
        )

    def step(self, action: DataraAction) -> Tuple[DataraObservation, DataraReward]:
        assert self._state is not None and self._task_config is not None

        self._state.step += 1

        # Call task-specific grader: must return (reward_delta, done, updated_context)
        reward_delta, done, updated_context = self._grade(
            action.message, self._state
        )

        # Small time penalty to discourage very long episodes
        time_penalty = -0.01
        reward_value = reward_delta + time_penalty

        self._state.cumulative_reward += reward_value

        # Normalize cumulative reward to [0, 1]
        normalized_score = max(0.0, min(1.0, self._state.cumulative_reward))

        episode_done = done or self._state.step >= self._state.max_steps

        obs = DataraObservation(
            task_id=self._task_config.id,
            instructions=self._task_config.instructions,
            context=updated_context,
            step=self._state.step,
            done=episode_done,
            score_so_far=normalized_score,
            messages=[action.message],
        )

        reward = DataraReward(
            value=reward_value,
            done=episode_done,
            info={"normalized_score": normalized_score},
        )

        return obs, reward

    def state(self) -> DataraState:
        return self._state

    # Internal: route to correct grader
    def _grade(self, message: str, state: DataraState) -> Tuple[float, bool, str]:
        # Example sketch: choose based on state.task_id
        from .tasks import grade_pii_masking, grade_relational_config, grade_eval_review

        if state.task_id == "pii_masking_easy":
            return grade_pii_masking(message, state)
        if state.task_id == "relational_config_medium":
            return grade_relational_config(message, state)
        if state.task_id == "eval_review_hard":
            return grade_eval_review(message, state)

        raise ValueError(f"No grader for task_id={state.task_id}")
```

***

## 6. FastAPI server and HTTP contract

File: `datara_env/server.py`

### 6.1 HTTP contract (important for validation)

`/reset` must return **just the observation** (as in many OpenEnv examples).  

`/step` must return an object with fields: `observation`, `reward`, `done`, `info`. [github](https://github.com/meta-pytorch/OpenEnv)

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Any, Dict, Optional

from .environment import DataraEnv
from .models import DataraObservation, DataraAction, DataraState, DataraReward

app = FastAPI(title="DataraEnv OpenEnv server")
env = DataraEnv()


class StepResponse(BaseModel):
    observation: DataraObservation
    reward: float
    done: bool
    info: Dict[str, Any]


@app.post("/reset", response_model=DataraObservation)
def reset(task_id: Optional[str] = Query(default=None)):
    """
    Initialize a new episode for the given task_id (or random if None).
    """
    return env.reset(task_id=task_id)


@app.post("/step", response_model=StepResponse)
def step(action: DataraAction):
    """
    Execute one environment step.
    """
    observation, reward_obj = env.step(action)
    return StepResponse(
        observation=observation,
        reward=reward_obj.value,
        done=reward_obj.done,
        info=reward_obj.info,
    )


@app.get("/state", response_model=DataraState)
def state():
    """
    Return current internal state.
    """
    return env.state()
```

This shape matches the competition’s requirement: `step(action) → observation, reward, done, info`. [deepfabric](https://www.deepfabric.dev/blog/introduction_to_openenv)

***

## 7. `openenv.yaml`

Root file: `openenv.yaml`

```yaml
name: datara-openenv
version: 0.1.0
description: >
  DataraEnv: an OpenEnv environment simulating realistic synthetic data engineering
  tasks under India's DPDP regime (PII detection, relational config, evaluation).

image: datara-openenv:latest
app_port: 8000
base_path: /

entrypoint:
  - uvicorn
  - datara_env.server:app
  - --host
  - 0.0.0.0
  - --port
  - "8000"

tags:
  - openenv
  - data
  - synthetic-data
  - dpdp
  - enterprise

tasks:
  - id: pii_masking_easy
    description: Identify PII columns and choose DPDP-safe masking strategies.
    difficulty: easy
  - id: relational_config_medium
    description: Design relational synthetic data generation config preserving PK/FK.
    difficulty: medium
  - id: eval_review_hard
    description: Review synthetic data evaluation and decide deployability.
    difficulty: hard
```

This must pass `openenv validate`. [pypi](https://pypi.org/project/openenv-core/0.2.0/)

***

## 8. Baseline inference script (`inference.py`)

### 8.1 Requirements

- Filename: `inference.py` (root).  
- Uses **OpenAI client** (`openai` package).  
- Reads env vars: `OPENAI_API_KEY`, `API_BASE_URL`, `MODEL_NAME`.  
- Runs all 3 tasks, prints reproducible average scores, under 20 minutes.

### 8.2 Implementation

```python
# inference.py
import os
import json
import requests
from typing import List

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ENV_SERVER_URL = os.getenv("DATARA_ENV_URL", "http://localhost:8000")

MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL,
)


SYSTEM_PROMPT = """You are a senior data engineer using Datara, a synthetic data platform.
You are interacting with an evaluation environment. Follow the task instructions,
reason step by step, and when you are asked to output JSON, respond with STRICTLY valid JSON.
Do not include comments or extra text outside the JSON when JSON is requested.
"""


def call_model(observation: dict) -> str:
    user_prompt = (
        f"Task ID: {observation['task_id']}\n"
        f"Step: {observation['step']}\n"
        f"Instructions:\n{observation['instructions']}\n\n"
        f"Context:\n{observation['context']}\n\n"
        f"Current normalized score: {observation['score_so_far']}"
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


def run_episode(task_id: str) -> float:
    # reset
    obs = requests.post(
        f"{ENV_SERVER_URL}/reset",
        params={"task_id": task_id},
        timeout=30,
    ).json()

    for step in range(MAX_STEPS):
        if obs["done"]:
            break

        response_text = call_model(obs)
        action = {"message": response_text}

        result = requests.post(
            f"{ENV_SERVER_URL}/step",
            json=action,
            timeout=30,
        ).json()

        obs = result["observation"]
        if result["done"]:
            break

    final_score = obs.get("score_so_far", 0.0)
    return float(final_score)


def main():
    tasks: List[str] = [
        "pii_masking_easy",
        "relational_config_medium",
        "eval_review_hard",
    ]
    episodes_per_task = int(os.getenv("EPISODES_PER_TASK", "3"))

    for task in tasks:
        scores = [run_episode(task) for _ in range(episodes_per_task)]
        avg_score = sum(scores) / len(scores)
        print(f"{task}: avg_score={avg_score:.3f} over {episodes_per_task} episodes")


if __name__ == "__main__":
    main()
```

***

## 9. Dependencies (`requirements.txt`)

Minimal:

```text
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.7.0
openai==1.30.0
requests==2.32.0

# OpenEnv core – adjust version to latest required by validator
openenv-core==0.2.0
```

(If `openenv-core` defines base types, you can extend them; otherwise, your Pydantic models are enough for this competition.)

***

## 10. Dockerfile

Root: `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY datara_env ./datara_env
COPY openenv.yaml .
COPY inference.py .
COPY README.md .

EXPOSE 8000

CMD ["uvicorn", "datara_env.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

This should work for both local `docker build/run` and Hugging Face Spaces. [github](https://github.com/meta-pytorch/OpenEnv)

***

## 11. README content (outline Windsurf should fill)

`README.md` must include:

1. **Overview & Motivation**  
   - What DataraEnv is, why synthetic data + DPDP matters, real-world personas.

2. **Action / Observation / State spaces**  
   - JSON examples for `/reset` and `/step` responses.  
   - Field definitions for `DataraObservation`, `DataraAction`, `DataraState`.

3. **Tasks & Difficulty**  
   - Detailed description of each task, what the agent must do, and how grading works.  
   - Mark which is easy / medium / hard.

4. **Reward Function**  
   - Explanation of partial credit, time penalty, normalization.

5. **Setup & Usage**  
   - Local: `pip install -r requirements.txt`, `uvicorn datara_env.server:app`.  
   - Docker: `docker build`, `docker run`.  
   - HF Space: mention that it’s Docker-based and tagged with `openenv`.

6. **Baseline Scores**  
   - Example output from `python inference.py` with model name, episodes, and average scores per task.

7. **OpenEnv Validation**  
   - Instructions to run `openenv validate` against `openenv.yaml`.


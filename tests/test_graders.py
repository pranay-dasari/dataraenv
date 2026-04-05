import json
import pytest
from datara_env.tasks import grade_pii_masking, grade_relational_config, grade_eval_review
from datara_env.models import DataraState
from datara_env.environment import DataraEnv


@pytest.fixture
def base_state():
    return DataraState(
        episode_id="test-id",
        task_id="test-task",
        step=0,
        cumulative_reward=0.0,
        max_steps=5
    )


# ── PII Masking Grader ──────────────────────────────────────

def test_grade_pii_masking_correct(base_state):
    action = {
        "columns": {
            "customer_id": {"is_pii": True, "masking_strategy": "tokenize"},
            "full_name": {"is_pii": True, "masking_strategy": "hash"},
            "phone_number": {"is_pii": True, "masking_strategy": "tokenize"},
            "email": {"is_pii": True, "masking_strategy": "tokenize"},
            "age": {"is_pii": False, "masking_strategy": "none"},
            "city": {"is_pii": False, "masking_strategy": "none"},
            "signup_at": {"is_pii": False, "masking_strategy": "none"}
        }
    }
    score, done, feedback = grade_pii_masking(json.dumps(action), base_state)
    assert score == 1.0
    assert done is True


def test_grade_pii_masking_partial(base_state):
    action = {
        "columns": {
            "customer_id": {"is_pii": True, "masking_strategy": "tokenize"},
            "full_name": {"is_pii": False, "masking_strategy": "none"}  # Wrong
        }
    }
    score, done, feedback = grade_pii_masking(json.dumps(action), base_state)
    assert 0.0 < score < 1.0


def test_grade_pii_masking_invalid_json(base_state):
    """Invalid JSON should return score 0, done=False (retry allowed)."""
    score, done, feedback = grade_pii_masking("not valid json", base_state)
    assert score == 0.0
    assert done is False


def test_grade_pii_masking_missing_columns(base_state):
    """Missing 'columns' key should also return done=False (consistent with JSON error)."""
    action = {"data": "something"}
    score, done, feedback = grade_pii_masking(json.dumps(action), base_state)
    assert score == 0.0
    assert done is False  # Fix: now consistent with JSON parse failure


# ── Relational Config Grader ────────────────────────────────

def test_grade_relational_config_correct(base_state):
    action = {
        "generation_order": ["customers", "accounts", "transactions"],
        "rows": {"customers": 100, "accounts": 200, "transactions": 2000},
        "dp_mode": "standard",
        "constraints": [
            "accounts.customer_id -> customers.customer_id",
            "transactions.account_id -> accounts.account_id"
        ]
    }
    score, done, feedback = grade_relational_config(json.dumps(action), base_state)
    assert score >= 0.9


def test_grade_relational_config_dp_mode_fairness(base_state):
    """Standard mode should get same total DP score (0.2) as valid DP mode."""
    standard_action = {
        "generation_order": ["customers", "accounts", "transactions"],
        "rows": {"customers": 100, "accounts": 200, "transactions": 2000},
        "dp_mode": "standard",
        "constraints": [
            "accounts.customer_id -> customers.customer_id",
            "transactions.account_id -> accounts.account_id"
        ]
    }
    dp_action = {
        "generation_order": ["customers", "accounts", "transactions"],
        "rows": {"customers": 100, "accounts": 200, "transactions": 2000},
        "dp_mode": "dp",
        "dp_epsilon": 5.0,
        "constraints": [
            "accounts.customer_id -> customers.customer_id",
            "transactions.account_id -> accounts.account_id"
        ]
    }
    score_std, _, _ = grade_relational_config(json.dumps(standard_action), base_state)
    score_dp, _, _ = grade_relational_config(json.dumps(dp_action), base_state)
    assert score_std == score_dp  # Both should get 1.0


def test_grade_relational_config_invalid_json(base_state):
    score, done, feedback = grade_relational_config("not json", base_state)
    assert score == 0.0
    assert done is False


# ── Eval Review Grader ──────────────────────────────────────

def test_grade_eval_review_correct_reject(base_state):
    """Fidelity 0.88 < 0.9 → correct decision is reject."""
    action = {
        "accept": False,
        "reasons": ["Low fidelity score"],
        "remediation_steps": ["Improve fidelity"]
    }
    score, done, feedback = grade_eval_review(json.dumps(action), base_state)
    assert score >= 0.8


def test_grade_eval_review_wrong_accept(base_state):
    """Accepting when fidelity fails should be penalized."""
    action = {
        "accept": True,
        "reasons": ["Everything looks good"],
        "remediation_steps": []
    }
    score, _, _ = grade_eval_review(json.dumps(action), base_state)
    assert score < 0.5


def test_grade_eval_review_invalid_json(base_state):
    score, done, feedback = grade_eval_review("not json", base_state)
    assert score == 0.0
    assert done is False


# ── Reward Inflation (Environment) ──────────────────────────

def test_no_reward_inflation():
    """Submitting the same answer repeatedly should NOT inflate the score."""
    env = DataraEnv()
    from datara_env.models import DataraAction

    env.reset(task_id="pii_masking_easy")

    # Send a partial answer that scores ~0.14
    partial_msg = json.dumps({
        "columns": {
            "customer_id": {"is_pii": True, "masking_strategy": "tokenize"},
        }
    })

    scores = []
    for _ in range(4):
        obs, reward = env.step(DataraAction(message=partial_msg))
        scores.append(obs.score_so_far)

    # Score should NOT increase across identical submissions
    # (cumulative_reward = max of all scores, which stays constant)
    assert scores[-1] == scores[0], f"Score inflated from {scores[0]} to {scores[-1]}"


# ── Done at max_steps ───────────────────────────────────────

def test_done_at_max_steps():
    """Episode should terminate at max_steps even with low scores."""
    env = DataraEnv()
    from datara_env.models import DataraAction

    env.reset(task_id="pii_masking_easy")
    done = False
    for i in range(10):
        obs, reward = env.step(DataraAction(message="invalid"))
        done = obs.done
        if done:
            break
    assert done is True


# ── Environment Contract ────────────────────────────────────

def test_state_before_reset():
    """state() before reset() should raise ValueError."""
    env = DataraEnv()
    with pytest.raises(ValueError, match="not reset"):
        env.state()


def test_step_before_reset():
    """step() before reset() should raise ValueError."""
    env = DataraEnv()
    from datara_env.models import DataraAction
    with pytest.raises(ValueError, match="not reset"):
        env.step(DataraAction(message="test"))

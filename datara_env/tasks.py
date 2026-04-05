import json
import os
from typing import Tuple, Dict, Any
from pydantic import BaseModel
from .models import DataraState


class TaskConfig(BaseModel):
    id: str
    difficulty: str                  # "easy" | "medium" | "hard"
    instructions: str
    initial_context: str
    max_steps: int
    grader_name: str                 # for logging/debug


# Task 1: PII & masking (easy)
pii_masking_task_config = TaskConfig(
    id="pii_masking_easy",
    difficulty="easy",
    instructions="""You are a data engineer working with Datara, a synthetic data platform.
Your task is to identify PII (Personally Identifiable Information) columns in the customer table and choose appropriate DPDP-safe masking strategies.

Available masking strategies:
- "none": No masking needed (not PII)
- "tokenize": Replace with random tokens (good for IDs, emails, phones)
- "hash": Apply cryptographic hash (good for names, identifiers)
- "generalize": Convert to broader categories (good for ages, locations)

Please analyze the schema and respond with JSON in this format:
{
  "columns": {
    "column_name": {"is_pii": true/false, "masking_strategy": "strategy"}
  }
}

Example columns: customer_id, full_name, phone_number, email, age, city, signup_at""",
    initial_context="""Table: customers
Columns:
- customer_id: Unique identifier for each customer
- full_name: Customer's full name
- phone_number: Customer's phone number
- email: Customer's email address
- age: Customer's age in years
- city: Customer's city of residence
- signup_at: When customer signed up (timestamp)""",
    max_steps=5,
    grader_name="grade_pii_masking"
)

# Ground truth for PII masking task
PII_MASKING_GROUND_TRUTH = {
    "customer_id": {"is_pii": True, "masking_strategy": "tokenize"},
    "full_name": {"is_pii": True, "masking_strategy": "hash"},
    "phone_number": {"is_pii": True, "masking_strategy": "tokenize"},
    "email": {"is_pii": True, "masking_strategy": "tokenize"},
    "age": {"is_pii": False, "masking_strategy": "none"},
    "city": {"is_pii": False, "masking_strategy": "none"},
    "signup_at": {"is_pii": False, "masking_strategy": "none"}
}


def grade_pii_masking(message: str, state: DataraState) -> Tuple[float, bool, str]:
    """Grade PII masking task responses."""
    try:
        response = json.loads(message)
        if "columns" not in response:
            return 0.0, False, "Invalid JSON format: missing 'columns' key. Please include a 'columns' object."
        
        columns = response["columns"]
        score = 0.0
        total_possible = len(PII_MASKING_GROUND_TRUTH) * 2  # 2 points per column
        
        for col_name, ground_truth in PII_MASKING_GROUND_TRUTH.items():
            if col_name not in columns:
                continue
            
            predicted = columns[col_name]
            
            # Check PII detection
            if predicted.get("is_pii") == ground_truth["is_pii"]:
                score += 1
            
            # Check masking strategy
            predicted_strategy = predicted.get("masking_strategy", "none")
            if ground_truth["is_pii"]:
                # For PII columns, check if strategy is acceptable
                acceptable_strategies = ["tokenize", "hash", "generalize"]
                if predicted_strategy in acceptable_strategies:
                    score += 1
            else:
                # For non-PII columns, should be "none"
                if predicted_strategy == "none":
                    score += 1
        
        normalized_score = score / total_possible
        done = normalized_score >= 0.95 or state.step >= state.max_steps - 1
        
        feedback = f"Current score: {normalized_score:.3f}. "
        if normalized_score >= 0.95:
            feedback += "Excellent! All columns correctly identified and masked."
        elif normalized_score >= 0.8:
            feedback += "Good progress. Review remaining columns."
        else:
            feedback += "Keep working. Check PII identification and masking strategies."
        
        return normalized_score, done, feedback
        
    except json.JSONDecodeError:
        return 0.0, False, "Invalid JSON format. Please provide valid JSON."


# Task 2: Relational config (medium)
relational_config_task_config = TaskConfig(
    id="relational_config_medium",
    difficulty="medium",
    instructions="""You are configuring synthetic data generation for a relational database schema.
Your task is to design the generation configuration preserving primary key/foreign key relationships.

Schema:
- customers (customer_id PK, name, email, age)
- accounts (account_id PK, customer_id FK, account_type, balance)
- transactions (transaction_id PK, account_id FK, amount, timestamp)

Requirements:
1. generation_order: List tables in dependency order (parents before children)
2. rows: Specify row counts for each table (maintain realistic ratios)
3. dp_mode: Either "standard" or "dp" for differential privacy
4. dp_epsilon: If dp_mode is "dp", set epsilon between 1.0 and 10.0
5. constraints: List foreign key relationships as "table.column -> parent_table.column"

Respond with JSON in this format:
{
  "generation_order": ["table1", "table2", "table3"],
  "rows": {"table1": 1000, "table2": 2000, "table3": 5000},
  "dp_mode": "dp" or "standard",
  "dp_epsilon": 5.0,
  "constraints": ["child_table.fk -> parent_table.pk"]
}""",
    initial_context="""Relational Schema:
customers (customer_id PK) -> accounts (customer_id FK) -> transactions (account_id FK)

Expected ratios:
- accounts should have >= customers (multiple accounts per customer)
- transactions should be much larger than accounts (many transactions per account)""",
    max_steps=5,
    grader_name="grade_relational_config"
)

# Ground truth for relational config task
RELATIONAL_CONFIG_GROUND_TRUTH = {
    "generation_order": ["customers", "accounts", "transactions"],
    "min_ratios": {
        "accounts": 1.2,  # accounts >= customers * 1.2
        "transactions": 10.0  # transactions >= accounts * 10
    },
    "constraints": [
        "accounts.customer_id -> customers.customer_id",
        "transactions.account_id -> accounts.account_id"
    ]
}


def grade_relational_config(message: str, state: DataraState) -> Tuple[float, bool, str]:
    """Grade relational configuration task responses."""
    try:
        response = json.loads(message)
        
        score = 0.0
        
        # Check generation order (0.4 points)
        if "generation_order" in response:
            order = response["generation_order"]
            if order == RELATIONAL_CONFIG_GROUND_TRUTH["generation_order"]:
                score += 0.4
            elif (len(order) == 3 and "customers" in order and 
                  order.index("customers") < order.index("accounts") and
                  order.index("accounts") < order.index("transactions")):
                score += 0.2  # Partial credit for correct dependency order
        
        # Check row ratios (0.2 points)
        if "rows" in response:
            rows = response["rows"]
            if all(table in rows for table in ["customers", "accounts", "transactions"]):
                customers_rows = rows["customers"]
                accounts_rows = rows["accounts"]
                transactions_rows = rows["transactions"]
                
                ratios_correct = (
                    accounts_rows >= customers_rows * RELATIONAL_CONFIG_GROUND_TRUTH["min_ratios"]["accounts"] and
                    transactions_rows >= accounts_rows * RELATIONAL_CONFIG_GROUND_TRUTH["min_ratios"]["transactions"]
                )
                if ratios_correct:
                    score += 0.2
        
        # Check constraints (0.2 points)
        if "constraints" in response:
            constraints = response["constraints"]
            if set(constraints) == set(RELATIONAL_CONFIG_GROUND_TRUTH["constraints"]):
                score += 0.2
            elif len(constraints) == 2 and all("->" in c for c in constraints):
                score += 0.1  # Partial credit
        
        # Check DP config (0.2 points)
        dp_mode = response.get("dp_mode", "")
        if dp_mode == "standard":
            score += 0.2  # Standard mode is fully valid
        elif dp_mode == "dp":
            score += 0.1  # Base credit for choosing DP
            if "dp_epsilon" in response:
                epsilon = response["dp_epsilon"]
                if 1.0 <= epsilon <= 10.0:
                    score += 0.1  # Valid epsilon range
        
        done = score >= 0.9 or state.step >= state.max_steps - 1
        
        feedback = f"Current score: {score:.3f}. "
        if score >= 0.9:
            feedback += "Excellent configuration!"
        elif score >= 0.7:
            feedback += "Good progress. Review remaining requirements."
        else:
            feedback += "Keep working. Check all requirements carefully."
        
        return score, done, feedback
        
    except json.JSONDecodeError:
        return 0.0, False, "Invalid JSON format. Please provide valid JSON."


# Task 3: Evaluation & decision (hard)
eval_review_task_config = TaskConfig(
    id="eval_review_hard",
    difficulty="hard",
    instructions="""You are reviewing a synthetic data evaluation report to decide if the data is ready for deployment.
Your task is to analyze the quality metrics and make an accept/reject decision with reasoning.

Evaluation metrics:
- fidelity: Column distribution similarity (0-1, higher is better)
- correlation_similarity: Correlation preservation (0-1, higher is better)  
- constraint_pass_rate: Constraint satisfaction rate (0-1, higher is better)
- privacy_risk: Privacy risk score (0-1, lower is better)

Quality thresholds:
- fidelity.avg >= 0.9
- correlation_similarity >= 0.85
- constraint_pass_rate >= 0.95
- privacy_risk <= 0.2

Respond with JSON in this format:
{
  "accept": true/false,
  "reasons": ["reason1", "reason2", ...],
  "remediation_steps": ["step1", "step2", ...]  // empty if accept=true
}

If accept=false, provide specific remediation steps like:
- "Enable differential privacy mode"
- "Lower epsilon from X to Y"
- "Regenerate with stricter constraints"
- "Improve correlation preservation" """,
    initial_context="""Synthetic Data Evaluation Report:
{
  "fidelity": {"avg": 0.88},
  "correlation_similarity": 0.86,
  "constraint_pass_rate": 0.97,
  "privacy_risk": 0.18
}

Review these metrics against the quality thresholds and make your decision.""",
    max_steps=5,
    grader_name="grade_eval_review"
)


def grade_eval_review(message: str, state: DataraState) -> Tuple[float, bool, str]:
    """Grade evaluation review task responses."""
    try:
        response = json.loads(message)
        
        # Current metrics from context
        fidelity_avg = 0.88
        correlation_similarity = 0.86
        constraint_pass_rate = 0.97
        privacy_risk = 0.18
        
        # Determine correct decision
        thresholds_met = (
            fidelity_avg >= 0.9 and
            correlation_similarity >= 0.85 and
            constraint_pass_rate >= 0.95 and
            privacy_risk <= 0.2
        )
        
        correct_accept = thresholds_met  # If thresholds are met, accept should be true
        
        score = 0.0
        
        # Check accept/reject decision (0.5 points)
        if "accept" in response:
            if response["accept"] == correct_accept:
                score += 0.5
        
        # Check reasons (0.25 points)
        if "reasons" in response and isinstance(response["reasons"], list):
            reasons = response["reasons"]
            reasons_text = " ".join(reasons).lower()
            
            # Look for relevant keywords
            relevant_keywords = []
            if fidelity_avg < 0.9:
                relevant_keywords.append("fidelity")
            if correlation_similarity < 0.85:
                relevant_keywords.append("correlation")
            if privacy_risk > 0.2:
                relevant_keywords.append("privacy")
            
            keyword_matches = sum(1 for keyword in relevant_keywords if keyword in reasons_text)
            if relevant_keywords:
                score += 0.25 * (keyword_matches / len(relevant_keywords))
        
        # Check remediation steps (0.25 points)
        if not correct_accept and "remediation_steps" in response:
            remediation = response["remediation_steps"]
            if isinstance(remediation, list) and len(remediation) > 0:
                remediation_text = " ".join(remediation).lower()
                
                # Check for appropriate remediation suggestions
                good_suggestions = 0
                if fidelity_avg < 0.9 and any(word in remediation_text for word in ["fidelity", "distribution", "quality"]):
                    good_suggestions += 1
                if "regenerate" in remediation_text:
                    good_suggestions += 1
                if "constraints" in remediation_text:
                    good_suggestions += 1
                
                score += 0.25 * min(1.0, good_suggestions / 2)
        elif correct_accept:
            # Instructions say remediation should be empty when accepting
            remediation = response.get("remediation_steps", [])
            if not remediation or len(remediation) == 0:
                score += 0.25  # Full credit: correctly empty remediation
            else:
                score += 0.1   # Partial: accepted but included unnecessary remediation
        
        done = score >= 0.8 or state.step >= state.max_steps - 1
        
        feedback = f"Current score: {score:.3f}. "
        if score >= 0.8:
            feedback += "Excellent analysis and decision!"
        elif score >= 0.6:
            feedback += "Good analysis. Review your reasoning and remediation steps."
        else:
            feedback += "Keep working. Focus on the metrics that fail thresholds."
        
        return score, done, feedback
        
    except json.JSONDecodeError:
        return 0.0, False, "Invalid JSON format. Please provide valid JSON."


# Task factory
from typing import Dict, Union

TASK_CONFIGS: Dict[str, TaskConfig] = {
    "pii_masking_easy": pii_masking_task_config,
    "relational_config_medium": relational_config_task_config,
    "eval_review_hard": eval_review_task_config,
}


def load_task(task_id: Union[str, None]) -> TaskConfig:
    if task_id is None:
        import random
        # Use a seed for reproducible random choice if env var set
        seed = os.getenv("DATARA_SEED")
        if seed:
            random.seed(int(seed))
        return random.choice(list(TASK_CONFIGS.values()))
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return TASK_CONFIGS[task_id]


# Grader routing function
def grade_message(message: str, state: DataraState) -> Tuple[float, bool, str]:
    """Route grading to appropriate task-specific grader."""
    if state.task_id == "pii_masking_easy":
        return grade_pii_masking(message, state)
    elif state.task_id == "relational_config_medium":
        return grade_relational_config(message, state)
    elif state.task_id == "eval_review_hard":
        return grade_eval_review(message, state)
    else:
        raise ValueError(f"No grader for task_id={state.task_id}")

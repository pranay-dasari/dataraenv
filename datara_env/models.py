from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DataraObservation(BaseModel):
    """
    Observation shown to the agent at each step.
    """
    episode_id: str
    task_id: str                     # e.g. "pii_masking_easy"
    instructions: str                # high-level instructions for the task
    context: str                     # current context: schema / report / text
    step: int                        # current step index (starting at 0)
    done: bool                       # whether the episode has ended
    score_so_far: float = Field(0.0, ge=0.0, le=1.0)
    messages: List[str] = Field(default_factory=list)
    feedback: Optional[str] = None


class DataraAction(BaseModel):
    """
    Action provided by the agent. For now, a single textual message.
    Typically JSON or structured text.
    """
    message: str = Field(..., description="Textual message or stringified action.")
    action_type: Optional[str] = Field(None, description="The type of action being performed.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Action configuration parameters.")
    rationale: Optional[str] = Field(None, description="Agent's internal reasoning.")


class DataraState(BaseModel):
    """
    Internal state of the environment for debugging and state() calls.
    """
    episode_id: str
    task_id: str
    step: int
    cumulative_reward: float
    max_steps: int
    history: List[str] = Field(default_factory=list)

    def get_history(self) -> List[str]:
        return self.history


class DataraReward(BaseModel):
    """
    Typed reward model (competition asks for typed Reward).
    """
    value: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

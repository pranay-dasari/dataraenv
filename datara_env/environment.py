import json
from typing import Tuple, Dict, Any, Optional, Union
from uuid import uuid4

from .models import DataraObservation, DataraAction, DataraState, DataraReward
from .tasks import TaskConfig, load_task, grade_message


class DataraEnv:
    """
    Core OpenEnv environment implementing reset/step/state.
    """

    def __init__(self):
        self._state: Optional[DataraState] = None
        self._task_config: Optional[TaskConfig] = None

    def reset(self, task_id: Union[str, None] = None) -> DataraObservation:
        self._task_config = load_task(task_id)
        self._state = DataraState(
            episode_id=str(uuid4()),
            task_id=self._task_config.id,
            step=0,
            cumulative_reward=0.0,
            max_steps=self._task_config.max_steps,
        )

        return DataraObservation(
            episode_id=self._state.episode_id,
            task_id=self._task_config.id,
            instructions=self._task_config.instructions,
            context=self._task_config.initial_context,
            step=0,
            done=False,
            score_so_far=0.0,
            messages=[],
            feedback=None
        )

    def step(self, action: DataraAction) -> Tuple[DataraObservation, DataraReward]:
        state = self._state
        task_config = self._task_config
        if state is None or task_config is None:
            raise ValueError("Environment not reset. Call reset() first.")

        state.step += 1

        # Call task-specific grader: must return (reward_delta, done, feedback)
        # Handle both legacy message and structured config
        if action.message:
            input_data = action.message
        else:
            # If no message, use the config/type as the input to the grader
            input_data = json.dumps({
                "action_type": action.action_type,
                "config": action.config,
                "rationale": action.rationale
            })

        current_score, done, feedback = self._grade(
            input_data, state
        )

        # Graders return absolute scores, not deltas.
        # Track the best score seen so far; reward is the improvement.
        previous_best = state.cumulative_reward
        state.cumulative_reward = max(previous_best, current_score)

        # Reward = improvement over previous best - small time penalty
        time_penalty = 0.01
        reward_value = (state.cumulative_reward - previous_best) - time_penalty

        # Clamp displayed score to [0, 1]
        normalized_score = max(0.0, min(1.0, state.cumulative_reward))

        episode_done = done or state.step >= task_config.max_steps

        # Update history
        state.history.append(input_data)
        
        obs = DataraObservation(
            episode_id=state.episode_id,
            task_id=task_config.id,
            instructions=task_config.instructions,
            context=task_config.initial_context, # Preserve original context
            step=state.step,
            done=episode_done,
            score_so_far=normalized_score,
            messages=state.history,
            feedback=feedback
        )

        reward = DataraReward(
            value=reward_value,
            done=episode_done,
            info={"normalized_score": normalized_score},
        )

        return obs, reward

    def state(self) -> DataraState:
        if self._state is None:
            raise ValueError("Environment not reset. Call reset() first.")
        return self._state

    # Internal: route to correct grader
    def _grade(self, message: str, state: DataraState) -> Tuple[float, bool, str]:
        return grade_message(message, state)

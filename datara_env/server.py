from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional, Union

from .environment import DataraEnv
from .models import DataraObservation, DataraAction, DataraState, DataraReward

app = FastAPI(title="DataraEnv OpenEnv server")
env = DataraEnv()


class StepResponse(BaseModel):
    observation: DataraObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    message: str


@app.post("/reset", response_model=DataraObservation, responses={400: {"model": ErrorResponse}})
def reset(task_id: Union[str, None] = Query(default=None)):
    """
    Initialize a new episode for the given task_id (or random if None).
    """
    try:
        return env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail={"error": "Value Error", "message": str(e)}
        )


@app.post("/step", response_model=StepResponse, responses={400: {"model": ErrorResponse}})
def step(action: DataraAction):
    """
    Execute one environment step.
    """
    try:
        observation, reward_obj = env.step(action)
        return StepResponse(
            observation=observation,
            reward=reward_obj.value,
            done=reward_obj.done,
            info=reward_obj.info,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail={"error": "Value Error", "message": str(e)}
        )
    except Exception as e:
        # Catch unexpected errors to maintain same schema
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error", "message": str(e)}
        )


@app.get("/state", response_model=DataraState, responses={400: {"model": ErrorResponse}})
def state():
    """
    Return current internal state.
    """
    try:
        return env.state()
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail={"error": "Value Error", "message": str(e)}
        )


@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "environment": "DataraEnv"}


@app.get("/")
def root():
    """
    Root endpoint with basic info.
    """
    return {
        "name": "DataraEnv OpenEnv Server",
        "version": "0.1.0",
        "description": "Synthetic data engineering environment for DPDP compliance",
        "endpoints": {
            "reset": "/reset",
            "step": "/step", 
            "state": "/state",
            "health": "/health"
        },
        "tasks": [
            "pii_masking_easy",
            "relational_config_medium", 
            "eval_review_hard"
        ]
    }

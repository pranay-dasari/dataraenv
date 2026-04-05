from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional, Union

from .environment import DataraEnv
from .models import DataraObservation, DataraAction, DataraState

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


@app.get("/", response_class=HTMLResponse)
def root():
    """
    Root endpoint with basic info.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DataraEnv Demo</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background: linear-gradient(135deg, #3b82f6 0%, #4f46e5 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                color: #1f2937;
            }
            .card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 2.5rem;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                max-width: 500px;
                width: 90%;
            }
            h1 {
                margin-top: 0;
                font-size: 1.5rem;
                color: #111827;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                font-size: 0.95rem;
                color: #4b5563;
                margin-bottom: 2rem;
            }
            h2 {
                font-size: 1.1rem;
                color: #374151;
                margin-bottom: 1rem;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 0.5rem;
            }
            ul {
                list-style: none;
                padding: 0;
                margin: 0 0 1.5rem 0;
            }
            li {
                background: #f3f4f6;
                padding: 0.5rem 1rem;
                margin-bottom: 0.5rem;
                border-radius: 6px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 0.85rem;
                color: #374151;
            }
            .btn {
                display: block;
                width: 100%;
                text-align: center;
                background: #4f46e5;
                color: white;
                text-decoration: none;
                padding: 0.75rem;
                border-radius: 8px;
                font-weight: 600;
                transition: background 0.2s;
                box-sizing: border-box;
                font-size: 1rem;
            }
            .btn:hover {
                background: #4338ca;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>DataraEnv — Synthetic Data Lab for DPDP Compliance</h1>
            <div class="subtitle">Synthetic data engineering environment for DPDP compliance</div>
            
            <h2>Available Endpoints</h2>
            <ul>
                <li>POST /reset</li>
                <li>POST /step</li>
                <li>GET /state</li>
                <li>GET /health</li>
                <li>GET /docs</li>
            </ul>

            <h2>Available Tasks</h2>
            <ul>
                <li>pii_masking_easy</li>
                <li>relational_config_medium</li>
                <li>eval_review_hard</li>
            </ul>

            <a href="/docs" class="btn">Open API Docs</a>
        </div>
    </body>
    </html>
    """
    return html_content

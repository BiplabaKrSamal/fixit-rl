"""
api/server.py
-------------
REST API for human control dashboard.
Exposes: training status, policy inspection, freeze/unfreeze, rollback, live eval.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.policy import LinearQAgent
from core.mdp import Action, N_ACTIONS
from evaluation.offline_eval import OfflinePolicyEvaluator, collect_episodes
from simulator.env import ScriptedBaseline, SimConfig, VoicebotEnv
from train import train

app = FastAPI(title="Fixit RL Control API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_agent: Optional[LinearQAgent] = None
_training_thread: Optional[threading.Thread] = None
_training_status = {"running": False, "progress": 0, "results": None, "log": []}
_output_dir = "outputs"

Path(_output_dir).mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class TrainRequest(BaseModel):
    episodes: int = 2000
    eval_episodes: int = 300
    seed: int = 42


class FreezeRequest(BaseModel):
    action: int


class RollbackRequest(BaseModel):
    steps_back: int = 1


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def start_training(req: TrainRequest):
    global _training_thread, _training_status, _agent
    if _training_status["running"]:
        raise HTTPException(400, "Training already running.")

    def _run():
        global _agent
        _training_status["running"] = True
        _training_status["log"] = []
        try:
            results = train(
                n_episodes=req.episodes,
                eval_episodes=req.eval_episodes,
                seed=req.seed,
                output_dir=_output_dir,
            )
            _training_status["results"] = results
            # Load the trained agent
            _agent = LinearQAgent.load(f"{_output_dir}/agent.pkl")
        except Exception as e:
            _training_status["results"] = {"error": str(e)}
        finally:
            _training_status["running"] = False
            _training_status["progress"] = 100

    _training_thread = threading.Thread(target=_run, daemon=True)
    _training_thread.start()
    return {"message": "Training started.", "episodes": req.episodes}


@app.get("/status")
def get_status():
    if Path(f"{_output_dir}/results.json").exists():
        with open(f"{_output_dir}/results.json") as f:
            results = json.load(f)
        _training_status["results"] = results
    return _training_status


@app.get("/results")
def get_results():
    path = Path(f"{_output_dir}/results.json")
    if not path.exists():
        raise HTTPException(404, "No results yet. Run /train first.")
    with open(path) as f:
        return json.load(f)


@app.get("/inspect")
def inspect_policy():
    agent = _load_agent()
    return agent.inspect()


@app.post("/freeze")
def freeze_action(req: FreezeRequest):
    if req.action < 0 or req.action >= N_ACTIONS:
        raise HTTPException(400, f"Action must be 0–{N_ACTIONS-1}")
    agent = _load_agent()
    agent.freeze_action(req.action)
    agent.save(f"{_output_dir}/agent.pkl")
    return {"message": f"Action {req.action} ({Action(req.action).name}) frozen.", "frozen": list(agent.frozen_actions)}


@app.post("/unfreeze")
def unfreeze_action(req: FreezeRequest):
    agent = _load_agent()
    agent.unfreeze_action(req.action)
    agent.save(f"{_output_dir}/agent.pkl")
    return {"message": f"Action {req.action} ({Action(req.action).name}) unfrozen.", "frozen": list(agent.frozen_actions)}


@app.post("/rollback")
def rollback(req: RollbackRequest):
    agent = _load_agent()
    success = agent.rollback(req.steps_back)
    if not success:
        raise HTTPException(400, "Not enough snapshots to roll back that far.")
    agent.save(f"{_output_dir}/agent.pkl")
    return {"message": f"Rolled back {req.steps_back} checkpoint(s).", "step": agent.total_steps}


@app.get("/actions")
def list_actions():
    return {
        "actions": [
            {"id": a.value, "name": a.name}
            for a in Action
        ]
    }


@app.get("/simulate")
def run_demo(n_episodes: int = 20):
    """Quick simulation to show current policy behaviour."""
    agent = _load_agent()
    env = VoicebotEnv(SimConfig(seed=777))
    results = []
    for _ in range(min(n_episodes, 50)):
        obs = env.reset()
        done = False
        total_r = 0.0
        turns = 0
        while not done:
            act = agent.select_action(obs, greedy=True)
            obs, r, done, info = env.step(act)
            total_r += r
            turns += 1
        results.append({
            "return": round(total_r, 3),
            "turns": turns,
            "completed": info.get("task_completed", False),
        })
    import numpy as np
    returns = [r["return"] for r in results]
    return {
        "n_episodes": len(results),
        "mean_return": round(float(np.mean(returns)), 3),
        "std_return": round(float(np.std(returns)), 3),
        "completion_rate": round(float(np.mean([r["completed"] for r in results])), 3),
        "episodes": results,
    }


# ------------------------------------------------------------------
# Serve dashboard
# ------------------------------------------------------------------

dashboard_dir = Path(__file__).parent.parent / "dashboard"

if dashboard_dir.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_dir), html=True), name="static")

    @app.get("/")
    def serve_dashboard():
        return FileResponse(str(dashboard_dir / "index.html"))


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_agent() -> LinearQAgent:
    global _agent
    agent_path = f"{_output_dir}/agent.pkl"
    if _agent is None:
        if Path(agent_path).exists():
            _agent = LinearQAgent.load(agent_path)
        else:
            # Return untrained agent for inspection
            _agent = LinearQAgent()
    return _agent


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

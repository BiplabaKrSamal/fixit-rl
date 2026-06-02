# Fixit RL — Voicebot Dialogue Policy Learner

A reinforcement learning system that makes a voicebot a better turn-level decision-maker over time, learning from conversation experience — cautiously, from delayed and noisy signals, without deploying an exploratory policy to real users.

## Quick Start

```bash
pip install -r requirements.txt

# Train (2000 episodes, ~8 seconds)
python train.py --episodes 2000 --eval-episodes 300

# Launch dashboard + API
python api/server.py
# → http://localhost:8000
```

## What It Does

| Component | File | Description |
|---|---|---|
| MDP Definition | `core/mdp.py` | State (14 features), 7 actions, shaped reward |
| Simulator | `simulator/env.py` | Voicebot conversation simulator + scripted baseline |
| RL Agent | `agent/policy.py` | Linear Q-learning: train, freeze, rollback, inspect |
| Offline Eval | `evaluation/offline_eval.py` | Clipped PDIS importance sampling with CI |
| API | `api/server.py` | FastAPI control server |
| Dashboard | `dashboard/index.html` | Human control UI |

## Results (2000 episodes, seed=42)

| Metric | Scripted Baseline | Trained Policy |
|---|---|---|
| Completion rate | ~50% (no learning) | **97%** |
| Episode return | 3.09 | 3.07 |
| Learning curve | Flat | Monotonically increasing |

The agent starts at −0.8 return (random exploration) and converges to the baseline level while achieving significantly higher task completion — it learns to use CONFIRM and CLARIFY strategically instead of always PROCEEDing.

## Human Control

The dashboard exposes:
- **Freeze/unfreeze** individual actions (e.g. freeze ESCALATE to prevent the bot from ever escalating)  
- **Rollback** to pre-training, mid-training, or post-training checkpoint
- **Weight inspection** — full Q-weight heatmap (feature × action)
- **Offline evaluation** with IPS estimate, 95% CI, and warnings

## Architecture

```
Conversation → [State features] → Linear Q(s,a) → Action
                                        ↑
                          TD(0) updates from replay buffer
                                        ↑
                          Human gates: freeze / rollback / inspect
```

## Design Document

See [DESIGN.md](DESIGN.md) for full MDP formulation, reward analysis, offline evaluation assumptions, reward hacking analysis, and exploration strategy.

## Deployment

Dockerfile + `render.yaml` included for one-click Render deployment.

```bash
docker build -t fixit-rl .
docker run -p 8000:8000 fixit-rl
```

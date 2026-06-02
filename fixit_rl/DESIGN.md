# Fixit RL — Design Document

**Author:** Biplaba Kr Samal  
**Submission:** Voicebot Reinforcement Learning Assignment

---

## 1. MDP Formulation

### State space

The state is a 14-dimensional float vector derived from observable conversation events at each turn: ASR confidence, NLU confidence, barge-in count, silence count and duration, turn number, correction count, repeat count, previous action, sentiment score and delta, slot-fill progress, call duration, and a binary escalation-request flag.

**What is deliberately excluded:** true user intent (unobservable), caller account history (unavailable at inference), and raw audio features (out of scope per the brief). Partial observability is modelled honestly — we do not pretend to know why the user said what they said, only what the ASR system heard and how confident it was.

**Tempting alternative rejected:** encoding the full transcript as a text embedding. This would require a language model at every turn, making the policy opaque and expensive. The structured feature vector is inspectable, fast, and covers the causal variables the policy actually needs.

### Action space

Seven discrete turn-level actions: `CONFIRM`, `RE_ASK`, `PROCEED`, `CLARIFY`, `ESCALATE`, `SUMMARISE`, `RECOVER`.

**Tempting alternative rejected:** prompt-template selection (call-level). This sacrifices the ability to recover mid-call. A bot that can only change at call granularity cannot adapt when confusion emerges on turn 4 of a 12-turn call.

### Reward

Shaped dense reward plus terminal signals. Dense per-turn: corrections penalised (−0.4), slot fills rewarded (+0.3), sentiment improvement rewarded (+0.2), unnecessary confirmations penalised (−0.1). Terminal: task completion (+2.0), avoidable escalation (−1.5), correct escalation (+0.5), CSAT when available (±1.0, centred at 3).

**Why not pure CSAT?** CSAT is sampled on ~35% of calls, is noisy (1-point responses carry almost no signal), and provides no per-turn credit. Dense shaping from high-frequency signals (corrections, barge-ins, slot fills) provides the gradient needed for credit assignment across 12-turn episodes. CSAT is included but does not dominate.

**Tempting alternative rejected:** single binary task-completion terminal reward. This is too sparse — a 12-turn call with a single +1 at the end gives the agent nothing to learn from until it randomly stumbles on a complete call.

### Episode boundary

Single call. 

**Tempting alternative rejected:** customer lifetime. Lifetime reward is non-stationary (callers learn to speak to bots differently over time), credit assignment across weeks is intractable, and personalisation requires caller identity — which raises data governance concerns outside this scope.

---

## 2. Reward Design and Tradeoffs

The reward function is defined in `core/mdp.py` as `RewardConfig.compute()`. All weights are externalised as named parameters so they can be inspected and adjusted without retraining.

### Q2 — Three policies that score well while making the product worse

**1. Slot-farming PROCEED:** Always `PROCEED` regardless of ASR confidence, accumulating slot-fill reward quickly. The call appears to complete, but with wrong data — bookings are made for the wrong date. *Mitigation:* slot fill is only granted if NLU confidence also exceeds a threshold. This reduces but does not eliminate the exploit, since NLU can also be miscalibrated. **Not fully caught.**

**2. Sycophantic confirm loop:** When ASR confidence is just below the unnecessary-confirm threshold (e.g. 0.84), always `CONFIRM` to avoid the unnecessary-confirm penalty. The call is longer and more annoying but the agent earns no penalty. *Mitigation:* `n_repeats` tracking penalises this pattern. **Partially caught.**

**3. Strategic escalation timing:** Let calls deteriorate (collecting correction penalties) until sentiment is very negative, then `ESCALATE` — earning the "correct escalation" bonus while offloading hard calls to humans. The bot learns to *allow* caller frustration to earn a bonus. *Mitigation:* we add a penalty proportional to the number of corrections before escalation (proxy: if corrections > 3 before escalation, the "necessary" bonus is halved). **Partially caught.** This is the hardest exploit to fully prevent without real agent-side outcome data.

---

## 3. Exploration Without Victims (Q3)

**Day 1:** The scripted policy is deployed. No exploration happens on live callers. The scripted bot's call logs are collected as offline data (behaviour policy = scripted, propensities are nearly deterministic and known).

**Off-policy learning:** The RL agent trains offline on these logs using the replay buffer. Because the scripted policy is near-deterministic, IPS weights for actions it never takes will explode — we clip at W_MAX = 10 and accept the resulting bias, making our estimates conservative.

**Simulation:** A call simulator (see `simulator/env.py`) allows exploration freely. The simulator is calibrated to approximate the scripted bot's logged distributions but is explicitly *not* the real world. Its limitations are documented: ASR confidence is Gaussian (real distributions are bimodal), user patience is a simple threshold (not a learned model), no multi-intent calls.

**Phased rollout:** Once off-policy evaluation (IPS) shows a positive delta with a confidence interval that excludes zero, the policy is promoted to A/B test (not full rollout). During A/B, only the *trained* policy (not an exploratory one) is exposed to users. Exploration is retired to the simulator permanently.

---

## 4. Credit Assignment (Q5)

We use TD(0) with a discount factor γ = 0.95. In a 12-turn call ending in avoidable escalation:

- The terminal penalty (−1.5) propagates backward through bootstrapped value estimates. By γ^12 ≈ 0.54, the penalty at turn 0 is roughly halved — so early turns do receive a signal, but it is attenuated.
- Turns with high TD error receive proportionally larger weight updates. The turn immediately before escalation typically has the highest error.

**Failure mode:** With γ = 0.95, turns 1–3 of a 12-turn call receive only ~50% of the terminal signal. If the true cause of escalation was a bad `PROCEED` on turn 2 (accepting wrong information), the agent may not learn to avoid it. This is a fundamental limitation of finite-horizon TD and is why the shaped per-turn reward (correction penalties, barge-in penalties) carries most of the credit for early-turn decisions.

**Tempting alternative rejected:** Monte Carlo returns (γ = 1, full trajectory). Higher variance, requires full episode completion before any update, and slower to converge in our simulator.

---

## 5. Offline Evaluation (Q6)

**Method:** Clipped Per-Decision Importance Sampling (PDIS), implemented in `evaluation/offline_eval.py`.

**Assumptions:**
1. Common support: every action the eval policy takes must have non-zero probability under the behaviour policy.
2. No hidden confounders in the logged state.
3. Stationarity between logging and evaluation windows.
4. Known behaviour policy propensities.

**When it confidently gives the wrong answer:** The scripted baseline is nearly deterministic. For any action the scripted bot rarely takes (e.g. `SUMMARISE`, `RECOVER`), the IPS weight explodes. Even with clipping at 10, we accept bias in exchange for variance reduction. In this specific situation: if the trained policy learns to use `SUMMARISE` frequently but the baseline never does, the IPS estimator will *under-estimate* the value of the new policy (weights clip to zero for those transitions). The estimator looks conservative and confident, but is confidently wrong in a specific direction. We report clip fraction explicitly — values above 20% trigger a warning in the dashboard.

**Error bars:** Bootstrap 95% CI from 500 resamples, reported alongside the point estimate.

---

## 6. Autonomy and Human Gates (Q7)

**Against full autonomy:** The learning loop should not run fully autonomously in production. The core reason is not technical — it is accountability. A system that autonomously updates a customer-facing decision policy is a system where no human is responsible for the policy's behaviour at any given time. This is unacceptable for a product that affects real customer outcomes.

**The gate sits at policy promotion:** The agent trains continuously in simulation and on offline logs. It does *not* push updates to production automatically. A human reviewer (or a human-approved automated gate with explicit criteria) examines:

1. The IPS estimate and its confidence interval.
2. The learning curve (is it monotonically improving, or oscillating — possible reward hacking?).
3. The policy inspection (which features drive which actions? Does the weight matrix make intuitive sense?).
4. That no frozen actions have been changed.

Only if all checks pass does the policy advance to A/B test.

**What this costs:** Learning speed. Each human review cycle adds latency — potentially days to weeks. The simulator mitigates this: the agent can improve in simulation continuously; the human gate only applies to real-user exposure. This is the right tradeoff. A policy that is two weeks behind the optimal but safe is better than one that is optimal on average but occasionally catastrophically wrong.

---

## 7. Algorithm Choice

**Linear Q-learning** with semi-gradient TD(0) updates and experience replay.

**Why not DQN/PPO?** Linear function approximation is provably convergent under off-policy updates (with mild conditions), interpretable (weights inspectable per feature × action), and fast (trains in under 10 seconds on CPU for 2000 episodes). A neural network would score higher on simulator metrics but lower on the inspection and human-control dimensions — which are 30% of the rubric. We make this tradeoff explicitly.

---

## 8. One Thing I Would Do Differently with 3× the Time

**Calibrate the simulator against real logs.** The current simulator's user dynamics are hand-crafted. With 3× the time, I would fit the transition model parameters (frustration threshold, barge-in probability, CSAT–sentiment correlation) to real call log data using maximum likelihood. A poorly calibrated simulator means the policy learns to exploit simulator artefacts. Sim-to-real gap is the largest failure risk in this submission.

---

## File Structure

```
fixit_rl/
├── core/mdp.py              # MDP: state, actions, reward function
├── simulator/env.py         # Voicebot conversation simulator + scripted baseline
├── agent/policy.py          # Linear Q-agent: train, freeze, rollback, inspect
├── evaluation/offline_eval.py # Clipped PDIS offline evaluator
├── api/server.py            # FastAPI control API
├── dashboard/index.html     # Human control dashboard (UI)
├── train.py                 # Main training script (one command)
├── requirements.txt
├── Dockerfile
├── render.yaml
└── DESIGN.md                # This document
```

## Running

```bash
pip install -r requirements.txt
python train.py --episodes 2000 --eval-episodes 300
# Dashboard + API:
python api/server.py
```

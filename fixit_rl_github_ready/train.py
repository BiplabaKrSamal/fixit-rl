"""
train.py
--------
Main entry point. Trains the RL policy, evaluates it offline, and produces
evidence of improvement over the scripted baseline.

Usage:
    python train.py [--episodes 2000] [--eval-episodes 500] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from agent.policy import LinearQAgent
from core.mdp import Action, RewardConfig
from evaluation.offline_eval import OfflinePolicyEvaluator, collect_episodes
from simulator.env import ScriptedBaseline, SimConfig, VoicebotEnv


def train(
    n_episodes: int = 3000,
    eval_episodes: int = 500,
    seed: int = 42,
    output_dir: str = "outputs",
) -> dict:
    Path(output_dir).mkdir(exist_ok=True)

    rng_seed = seed
    env = VoicebotEnv(SimConfig(seed=rng_seed))
    eval_env = VoicebotEnv(SimConfig(seed=rng_seed + 1))

    agent = LinearQAgent(
        lr=5e-4,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=n_episodes * 5,
        replay_capacity=20_000,
        batch_size=64,
    )

    baseline = ScriptedBaseline()
    reward_cfg = RewardConfig()

    print("=" * 60)
    print("  Fixit RL — Voicebot Dialogue Policy Learner")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Evaluate baseline (scripted bot)
    # ------------------------------------------------------------------
    print("\n[1/4] Evaluating scripted baseline...")
    baseline_returns = []
    baseline_completions = []
    baseline_escalations = []

    for _ in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        total_r = 0.0
        while not done:
            act = baseline.select_action(obs)
            obs, r, done, info = eval_env.step(act)
            total_r += r
        baseline_returns.append(total_r)
        baseline_completions.append(int(info.get("task_completed", False)))
        baseline_escalations.append(int(act == int(Action.ESCALATE)))

    baseline_mean = float(np.mean(baseline_returns))
    baseline_std = float(np.std(baseline_returns))
    baseline_completion_rate = float(np.mean(baseline_completions))
    baseline_escalation_rate = float(np.mean(baseline_escalations))

    print(f"  Baseline return:         {baseline_mean:.3f} ± {baseline_std:.3f}")
    print(f"  Baseline completion:     {baseline_completion_rate:.1%}")
    print(f"  Baseline escalation:     {baseline_escalation_rate:.1%}")

    # ------------------------------------------------------------------
    # Step 2: Train agent
    # ------------------------------------------------------------------
    print(f"\n[2/4] Training agent for {n_episodes} episodes...")

    # Take snapshot before training (for rollback demo)
    agent.snapshot(label="pre-training")

    learning_curve = []
    window = 100
    ep_returns: list[float] = []
    ep_completions: list[int] = []
    losses: list[float] = []

    t0 = time.time()

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0

        while not done:
            act = agent.select_action(obs)
            obs2, r, done, info = env.step(act)
            agent.store(obs, act, r, obs2, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            obs = obs2
            total_r += r

        ep_returns.append(total_r)
        ep_completions.append(int(info.get("task_completed", False)))

        if (ep + 1) % window == 0:
            mean_r = float(np.mean(ep_returns[-window:]))
            mean_c = float(np.mean(ep_completions[-window:]))
            mean_l = float(np.mean(losses[-window:])) if losses else 0.0
            learning_curve.append({
                "episode": ep + 1,
                "mean_return": round(mean_r, 4),
                "completion_rate": round(mean_c, 4),
                "mean_loss": round(mean_l, 6),
                "epsilon": round(agent.epsilon, 4),
            })
            elapsed = time.time() - t0
            print(
                f"  ep {ep+1:5d} | return={mean_r:7.3f} | "
                f"completion={mean_c:.1%} | ε={agent.epsilon:.3f} | "
                f"loss={mean_l:.5f} | {elapsed:.1f}s"
            )

        # Mid-training snapshot
        if ep + 1 == n_episodes // 2:
            agent.snapshot(label="mid-training")

    agent.snapshot(label="post-training")

    print(f"\n  Training complete in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Evaluate trained policy (on-policy, greedy)
    # ------------------------------------------------------------------
    print(f"\n[3/4] Evaluating trained policy (greedy, n={eval_episodes})...")
    trained_returns = []
    trained_completions = []
    trained_escalations = []

    for _ in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        total_r = 0.0
        last_act = int(Action.PROCEED)
        while not done:
            act = agent.select_action(obs, greedy=True)
            obs, r, done, info = eval_env.step(act)
            total_r += r
            last_act = act
        trained_returns.append(total_r)
        trained_completions.append(int(info.get("task_completed", False)))
        trained_escalations.append(int(last_act == int(Action.ESCALATE)))

    trained_mean = float(np.mean(trained_returns))
    trained_std = float(np.std(trained_returns))
    trained_completion_rate = float(np.mean(trained_completions))
    trained_escalation_rate = float(np.mean(trained_escalations))

    print(f"  Trained return:          {trained_mean:.3f} ± {trained_std:.3f}")
    print(f"  Trained completion:      {trained_completion_rate:.1%}")
    print(f"  Trained escalation:      {trained_escalation_rate:.1%}")
    print(f"  Delta vs baseline:       {trained_mean - baseline_mean:+.3f}")

    # ------------------------------------------------------------------
    # Step 4: Offline evaluation (IPS) of trained policy from baseline logs
    # ------------------------------------------------------------------
    print(f"\n[4/4] Offline (IPS) evaluation of trained policy...")

    ope_env = VoicebotEnv(SimConfig(seed=rng_seed + 99))
    ope_episodes, ope_eval_probs = collect_episodes(
        ope_env, agent, n_episodes=200, behaviour_policy=baseline
    )

    evaluator = OfflinePolicyEvaluator(gamma=0.95, clip_max=10.0)
    ope_result = evaluator.evaluate(ope_episodes, ope_eval_probs)

    print(f"  IPS estimate:            {ope_result.estimate:.3f} ± {ope_result.std_error:.3f}")
    print(f"  95% CI:                  [{ope_result.confidence_interval_95[0]:.3f}, {ope_result.confidence_interval_95[1]:.3f}]")
    print(f"  Baseline (logged):       {ope_result.baseline_value:.3f}")
    print(f"  Clip fraction:           {ope_result.clip_fraction:.1%}")
    if ope_result.warnings:
        for w in ope_result.warnings:
            print(f"  ⚠ WARNING: {w}")

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    agent.save(f"{output_dir}/agent.pkl")
    policy_inspection = agent.inspect()

    results = {
        "baseline": {
            "mean_return": round(baseline_mean, 4),
            "std": round(baseline_std, 4),
            "completion_rate": round(baseline_completion_rate, 4),
            "escalation_rate": round(baseline_escalation_rate, 4),
        },
        "trained": {
            "mean_return": round(trained_mean, 4),
            "std": round(trained_std, 4),
            "completion_rate": round(trained_completion_rate, 4),
            "escalation_rate": round(trained_escalation_rate, 4),
        },
        "delta": round(trained_mean - baseline_mean, 4),
        "delta_pct": round((trained_mean - baseline_mean) / max(abs(baseline_mean), 1e-6) * 100, 2),
        "offline_ipe": {
            "estimate": round(ope_result.estimate, 4),
            "std_error": round(ope_result.std_error, 4),
            "ci_95": [round(x, 4) for x in ope_result.confidence_interval_95],
            "clip_fraction": round(ope_result.clip_fraction, 4),
            "warnings": ope_result.warnings,
        },
        "learning_curve": learning_curve,
        "policy_inspection": policy_inspection,
    }

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline return:  {baseline_mean:.3f}")
    print(f"  Trained return:   {trained_mean:.3f}  ({results['delta_pct']:+.1f}%)")
    print(f"  IPS estimate:     {ope_result.estimate:.3f} (offline, from baseline logs)")
    print(f"  Results saved to: {output_dir}/")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train voicebot RL policy")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )

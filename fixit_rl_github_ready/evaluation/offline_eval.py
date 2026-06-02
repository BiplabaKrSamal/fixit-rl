"""
evaluation/offline_eval.py
--------------------------
Off-policy evaluation: estimate the value of a new policy π_e
from data collected under a behaviour policy π_b.

Method: Per-Decision Importance Sampling (PDIS) / IPS.
We use the clipped (capped) variant to reduce variance.

STATED ASSUMPTIONS:
1. Common support: every action π_e would take must have non-zero
   probability under π_b. Violated when π_e explores regions the
   scripted bot never visited (e.g. RECOVER, SUMMARISE).
2. No hidden confounders: the logged state fully captures what
   determined the behaviour policy's action. Partially violated
   (real bots may have caller ID, account history not in our log).
3. Stationarity: the distribution of conversations does not shift
   between data collection and policy deployment. Violated as soon
   as π_e is deployed.
4. The behaviour policy's propensities are known. In practice they
   are estimated (scripted bot: deterministic → near-0 propensity
   for non-chosen actions, which causes IPS weights to explode).

WHEN IT GIVES THE WRONG ANSWER:
- If π_b is nearly deterministic (scripted baseline), IPS weights
  for any action the baseline rarely takes will explode, producing
  wildly high variance estimates. Even clipping at W_MAX doesn't
  fully fix this — clipping introduces bias.
- If the new policy exploits state regions π_b never visited,
  the estimator extrapolates (actually: refuses to — gives NaN or
  clips to zero). This is conservative but misleading.
- If logged rewards are systematically biased (e.g. frustrated callers
  hang up before CSAT survey, so logged CSAT skews positive), the
  estimated value is biased upward for any policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Transition:
    state:       np.ndarray
    action:      int
    reward:      float
    next_state:  np.ndarray
    done:        bool
    behaviour_prob: float   # π_b(a|s) — probability assigned by behaviour policy


@dataclass
class Episode:
    transitions: List[Transition]

    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)


@dataclass
class OPEResult:
    estimate: float
    std_error: float
    n_episodes: int
    n_clipped: int
    clip_fraction: float
    baseline_value: float
    confidence_interval_95: Tuple[float, float]
    warnings: List[str]


class OfflinePolicyEvaluator:
    """
    Clipped Per-Decision Importance Sampling (PDIS).

    For each episode i with T turns:
        ρ_t = π_e(a_t|s_t) / π_b(a_t|s_t)
        ρ_{1:t} = Π_{j=1}^{t} ρ_j   (cumulative product)
        PDIS(i) = Σ_t γ^t * clip(ρ_{1:t}, 0, W_MAX) * r_t

    Aggregate: mean over episodes, with bootstrap CI.
    """

    def __init__(
        self,
        gamma: float = 0.95,
        clip_max: float = 10.0,   # W_MAX — higher = lower bias, higher variance
        bootstrap_samples: int = 500,
        min_behaviour_prob: float = 1e-4,  # floor to prevent /0
    ):
        self.gamma = gamma
        self.clip_max = clip_max
        self.bootstrap_samples = bootstrap_samples
        self.min_behaviour_prob = min_behaviour_prob

    def evaluate(
        self,
        episodes: List[Episode],
        eval_policy_probs: List[List[np.ndarray]],   # π_e(·|s_t) for each ep, each turn
    ) -> OPEResult:
        """
        Parameters
        ----------
        episodes : logged episodes under behaviour policy
        eval_policy_probs : for each episode i, turn t: array of shape (n_actions,)
                            giving π_e's action probabilities at that state.
        """
        warnings: list[str] = []
        ep_values: list[float] = []
        n_clipped = 0
        total_weights = 0

        for i, ep in enumerate(episodes):
            ep_val = 0.0
            cum_ratio = 1.0
            ep_probs = eval_policy_probs[i]

            for t, trans in enumerate(ep.transitions):
                pi_e_a = float(ep_probs[t][trans.action])
                pi_b_a = max(float(trans.behaviour_prob), self.min_behaviour_prob)

                ratio = pi_e_a / pi_b_a
                cum_ratio *= ratio
                total_weights += 1

                if cum_ratio > self.clip_max:
                    n_clipped += 1
                    cum_ratio = self.clip_max  # clip and continue (biased but stable)

                ep_val += (self.gamma ** t) * cum_ratio * trans.reward

            ep_values.append(ep_val)

        if total_weights == 0:
            raise ValueError("No transitions found.")

        ep_arr = np.array(ep_values)
        estimate = float(np.mean(ep_arr))
        std_error = float(np.std(ep_arr) / np.sqrt(len(ep_arr)))
        clip_frac = n_clipped / max(total_weights, 1)

        # Bootstrap 95% CI
        boot_means = [
            float(np.mean(np.random.choice(ep_arr, size=len(ep_arr), replace=True)))
            for _ in range(self.bootstrap_samples)
        ]
        ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

        baseline_value = float(np.mean([ep.total_reward for ep in episodes]))

        if clip_frac > 0.2:
            warnings.append(
                f"High clip fraction ({clip_frac:.1%}): estimate is biased. "
                "The behaviour policy rarely took actions the eval policy prefers. "
                "Treat this estimate as a lower bound."
            )
        if std_error > abs(estimate) * 0.5:
            warnings.append(
                "Standard error is large relative to estimate. "
                "Collect more data or reduce clip_max before trusting this number."
            )
        if estimate > baseline_value * 2:
            warnings.append(
                "Estimated value is >2× baseline — likely an optimistic bias. "
                "Check for common-support violations."
            )

        return OPEResult(
            estimate=estimate,
            std_error=std_error,
            n_episodes=len(episodes),
            n_clipped=n_clipped,
            clip_fraction=clip_frac,
            baseline_value=baseline_value,
            confidence_interval_95=(ci_lo, ci_hi),
            warnings=warnings,
        )


# ------------------------------------------------------------------
# Helper: collect episodes from a policy in the simulator
# ------------------------------------------------------------------

def collect_episodes(
    env,
    policy,
    n_episodes: int,
    behaviour_policy=None,
    epsilon_for_probs: float = 0.1,
) -> Tuple[List[Episode], List[List[np.ndarray]]]:
    """
    Collect n_episodes of experience. Returns episodes + eval policy probs.
    
    behaviour_policy: if provided, actions are taken by behaviour_policy
                      (for off-policy collection). Otherwise uses `policy`.
    epsilon_for_probs: smoothing for behaviour prob (scripted = deterministic).
    """
    from core.mdp import N_ACTIONS

    episodes = []
    all_eval_probs = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        transitions = []
        eval_probs = []

        while not done:
            # Behaviour action
            if behaviour_policy is not None:
                act = behaviour_policy.select_action(obs)
            else:
                act = policy.select_action(obs)

            # Behaviour probability (scripted = near-deterministic)
            b_prob_vec = np.full(N_ACTIONS, epsilon_for_probs / N_ACTIONS)
            b_prob_vec[act] += (1 - epsilon_for_probs)
            b_prob = float(b_prob_vec[act])

            # Eval policy probabilities (softmax over Q-values)
            if hasattr(policy, "q_values"):
                q = policy.q_values(obs)
                q_stable = q - q.max()
                exp_q = np.exp(q_stable / 0.5)   # temperature=0.5
                pi_e = exp_q / exp_q.sum()
                for fa in policy.frozen_actions:
                    pi_e[fa] = 0.0
                if pi_e.sum() > 0:
                    pi_e /= pi_e.sum()
            else:
                pi_e = b_prob_vec  # fallback

            obs2, rew, done, info = env.step(act)
            transitions.append(Transition(obs, act, rew, obs2, done, b_prob))
            eval_probs.append(pi_e)
            obs = obs2

        episodes.append(Episode(transitions))
        all_eval_probs.append(eval_probs)

    return episodes, all_eval_probs

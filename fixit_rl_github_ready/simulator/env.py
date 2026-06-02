"""
simulator/env.py
----------------
Minimal voicebot conversation simulator.

LIMITATIONS (stated honestly):
- User patience / frustration modelled as a simple threshold, not a learned model.
- ASR confidence is synthetic Gaussian noise — real ASR distributions are
  bimodal (near-certain or very uncertain), not Gaussian.
- Sentiment is a proxy computed from correction/barge-in events, not from
  an actual acoustic or NLP sentiment model.
- Task complexity is fixed per episode; real calls vary by task type.
- We do not model multi-intent calls or mid-call topic changes.
- No speech synthesis artefacts (TTS failures, codec dropouts).
These limitations mean training here over-estimates how well the policy
will do on real traffic. See offline evaluation for a partial correction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from core.mdp import (
    Action,
    ConversationState,
    RewardConfig,
    N_ACTIONS,
    STATE_DIM,
)


@dataclass
class SimConfig:
    max_turns:            int   = 15
    n_slots:              int   = 4
    base_asr_confidence:  float = 0.78   # mean ASR confidence in this task domain
    asr_noise_std:        float = 0.15
    frustration_threshold: int  = 3      # corrections before user hangs up
    csat_sample_rate:     float = 0.35   # 35% of calls get CSAT
    seed:                 Optional[int] = None


class VoicebotEnv:
    """
    Gym-like environment (no Gym dependency).

    Observations: np.ndarray of shape (STATE_DIM,)
    Actions: int in [0, N_ACTIONS)
    """

    def __init__(self, config: SimConfig | None = None, reward_cfg: RewardConfig | None = None):
        self.cfg = config or SimConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._state: ConversationState = self._initial_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._state = self._initial_state()
        self._done = False
        self._task_completed = False
        self._total_corrections = 0
        return self._state.to_vector()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert not self._done, "Call reset() after episode ends."
        action_e = Action(action)
        prev_state = self._state
        next_state, info = self._transition(prev_state, action_e)
        terminal = info["terminal"]
        csat = self._sample_csat(next_state, terminal)

        reward = self.reward_cfg.compute(
            state=prev_state,
            action=action_e,
            next_state=next_state,
            terminal=terminal,
            task_completed=info["task_completed"],
            escalation_was_necessary=info.get("escalation_necessary"),
            csat=csat,
        )

        self._state = next_state
        self._done = terminal

        info["csat"] = csat
        return next_state.to_vector(), reward, terminal, info

    @property
    def observation_dim(self) -> int:
        return STATE_DIM

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    # ------------------------------------------------------------------
    # Internal dynamics
    # ------------------------------------------------------------------

    def _initial_state(self) -> ConversationState:
        return ConversationState(
            asr_confidence=float(np.clip(self._rng.normal(self.cfg.base_asr_confidence,
                                                           self.cfg.asr_noise_std), 0, 1)),
            nlu_confidence=float(np.clip(self._rng.normal(0.75, 0.12), 0, 1)),
            n_barge_ins=0,
            n_silences=0,
            last_silence_duration=0.0,
            turn_number=0,
            n_corrections=0,
            n_repeats=0,
            last_action=int(Action.PROCEED),
            sentiment_score=float(self._rng.uniform(-0.1, 0.3)),
            sentiment_delta=0.0,
            slots_filled=0,
            slots_total=self.cfg.n_slots,
            call_duration_s=0.0,
            escalation_requested=False,
        )

    def _transition(
        self, state: ConversationState, action: Action
    ) -> Tuple[ConversationState, dict]:
        """Simulate next state given action. Returns (next_state, info_dict)."""
        rng = self._rng

        # Simulate new ASR confidence
        new_asr = float(np.clip(rng.normal(self.cfg.base_asr_confidence,
                                            self.cfg.asr_noise_std), 0, 1))
        new_nlu = float(np.clip(rng.normal(0.75, 0.12), 0, 1))

        n_corrections = state.n_corrections
        n_repeats = state.n_repeats
        n_barge_ins = state.n_barge_ins
        n_silences = state.n_silences
        slots_filled = state.slots_filled
        sentiment = state.sentiment_score
        escalation_requested = state.escalation_requested
        call_duration = state.call_duration_s + rng.uniform(5, 25)  # turn takes 5-25s
        silence_dur = 0.0

        task_completed = False
        terminal = False
        escalation_necessary = None

        # ---- Action effects ----

        if action == Action.PROCEED:
            # Might misunderstand
            if new_asr < 0.55:
                n_corrections += 1
                sentiment -= 0.15
            elif slots_filled < state.slots_total:
                slots_filled += 1

        elif action == Action.CONFIRM:
            if state.asr_confidence < 0.60:
                # Good confirm — user gives clean response
                if slots_filled < state.slots_total:
                    slots_filled += 1
                sentiment += 0.05
            else:
                # Unnecessary confirm — mild annoyance
                n_repeats += 1
                sentiment -= 0.05

        elif action == Action.RE_ASK:
            n_repeats += 1
            silence_dur = rng.uniform(1, 4)
            if silence_dur > 2:
                n_silences += 1
            # Re-ask can rescue confused user
            if rng.random() < 0.6:
                if slots_filled < state.slots_total:
                    slots_filled += 1

        elif action == Action.CLARIFY:
            # Good for ambiguous situations
            if new_nlu < 0.65:
                if slots_filled < state.slots_total:
                    slots_filled += 1
                sentiment += 0.08
            else:
                n_repeats += 1

        elif action == Action.ESCALATE:
            terminal = True
            # Was escalation necessary?
            escalation_necessary = (
                n_corrections >= 3
                or escalation_requested
                or sentiment < -0.5
            )
            if not escalation_necessary:
                sentiment -= 0.2  # unnecessary escalation annoys users

        elif action == Action.SUMMARISE:
            sentiment += 0.1   # users like feeling heard
            silence_dur = rng.uniform(3, 8)
            if silence_dur > 2:
                n_silences += 1

        elif action == Action.RECOVER:
            n_repeats += 1
            sentiment += 0.15   # explicit apology helps
            call_duration += 10  # takes more time

        # User frustration → spontaneous escalation request
        if n_corrections >= self.cfg.frustration_threshold and not terminal:
            if rng.random() < 0.4:
                escalation_requested = True

        # Barge-in probability increases with frustration
        if sentiment < -0.3 and rng.random() < 0.25:
            n_barge_ins += 1

        # Sentiment delta
        new_sentiment_delta = sentiment - state.sentiment_score

        # Terminal conditions
        if not terminal:
            if slots_filled >= state.slots_total:
                task_completed = True
                terminal = True
            elif state.turn_number + 1 >= self.cfg.max_turns:
                terminal = True
            elif escalation_requested and rng.random() < 0.7:
                terminal = True
                action = Action.ESCALATE  # type: ignore
                escalation_necessary = True

        next_state = ConversationState(
            asr_confidence=new_asr,
            nlu_confidence=new_nlu,
            n_barge_ins=n_barge_ins,
            n_silences=n_silences,
            last_silence_duration=silence_dur,
            turn_number=state.turn_number + 1,
            n_corrections=n_corrections,
            n_repeats=n_repeats,
            last_action=int(action),
            sentiment_score=float(np.clip(sentiment, -1.0, 1.0)),
            sentiment_delta=float(new_sentiment_delta),
            slots_filled=slots_filled,
            slots_total=state.slots_total,
            call_duration_s=call_duration,
            escalation_requested=escalation_requested,
        )

        info = {
            "terminal": terminal,
            "task_completed": task_completed,
            "escalation_necessary": escalation_necessary,
        }
        return next_state, info

    def _sample_csat(self, state: ConversationState, terminal: bool) -> Optional[float]:
        if not terminal:
            return None
        if self._rng.random() > self.cfg.csat_sample_rate:
            return None
        # CSAT correlates with sentiment and task completion
        mean_csat = 3.0 + state.sentiment_score * 1.5
        csat = float(np.clip(self._rng.normal(mean_csat, 0.8), 1, 5))
        return round(csat * 2) / 2  # quantise to 0.5 steps


# ------------------------------------------------------------------
# Scripted baseline policy (current production bot)
# ------------------------------------------------------------------

class ScriptedBaseline:
    """
    Approximates the current hand-written rule-based voicebot policy.
    Always confirms if ASR < 0.65, escalates if corrections > 3,
    otherwise proceeds. No learning.
    """

    def select_action(self, obs: np.ndarray) -> int:
        # Decode relevant features from vector (same order as ConversationState.to_vector)
        asr_conf    = obs[0]
        n_corrections = obs[6] * 5.0   # un-normalise
        escalation_req = obs[13] > 0.5

        if escalation_req or n_corrections >= 3:
            return int(Action.ESCALATE)
        if asr_conf < 0.65:
            return int(Action.CONFIRM)
        return int(Action.PROCEED)

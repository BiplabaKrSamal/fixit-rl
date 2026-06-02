"""
core/mdp.py
-----------
Formal MDP definition for the voicebot dialogue policy.

Design decisions:
- Episode boundary = single call (not lifetime): lifetime reward is non-stationary
  and credit assignment becomes intractable across weeks/months of interactions.
- Turn-level actions (not call-level): call-level collapse loses the structure
  of sequential recovery. The value of confirming *before* collecting a card number
  is structurally different from confirming after.
- State = featurised conversation window (not raw transcript): partial observability
  is real; we model it honestly as a finite feature vector, not pretend we see intent.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class Action(IntEnum):
    """Turn-level actions available to the voicebot."""
    CONFIRM       = 0   # "Just to confirm, you said X?"
    RE_ASK        = 1   # Repeat the question differently
    PROCEED       = 2   # Accept what was heard and move on
    CLARIFY       = 3   # "Did you mean X or Y?"
    ESCALATE      = 4   # Transfer to human agent
    SUMMARISE     = 5   # Recap collected info before continuing
    RECOVER       = 6   # Explicit recovery: "I'm sorry, let me try again"

N_ACTIONS = len(Action)

# Why not: "choose a different prompt template" (call-level)?
# That sacrifices turn-level recovery. A bot that can only change at call
# granularity cannot adapt mid-call when confusion emerges on turn 4.


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    """
    Observable state at each turn.

    Deliberately partial: we do NOT include true user intent (unobservable).
    We model what the system actually has access to.
    """

    # ASR / NLU signals
    asr_confidence: float           # [0,1] from ASR engine
    nlu_confidence: float           # [0,1] from NLU intent classifier
    n_barge_ins:    int             # times user interrupted bot this call
    n_silences:     int             # silence events (>2s) this call
    last_silence_duration: float    # seconds of last silence

    # Dialogue history signals
    turn_number: int                # current turn (0-indexed)
    n_corrections: int              # times user said "no, I meant…"
    n_repeats: int                  # times bot asked same thing twice
    last_action: int                # previous Action taken

    # Sentiment / prosody proxy (from text sentiment model)
    sentiment_score: float          # [-1,1]; negative = frustrated
    sentiment_delta: float          # change in sentiment vs previous turn

    # Task progress
    slots_filled: int               # how many required slots collected
    slots_total:  int               # total required slots for the task
    call_duration_s: float          # elapsed seconds

    # Escalation context
    escalation_requested: bool      # user explicitly asked for human

    def to_vector(self) -> np.ndarray:
        """Flat feature vector for function approximators."""
        return np.array([
            self.asr_confidence,
            self.nlu_confidence,
            self.n_barge_ins / 10.0,            # normalise
            self.n_silences / 10.0,
            self.last_silence_duration / 30.0,
            self.turn_number / 20.0,
            self.n_corrections / 5.0,
            self.n_repeats / 5.0,
            self.last_action / N_ACTIONS,
            (self.sentiment_score + 1) / 2.0,   # shift to [0,1]
            (self.sentiment_delta + 2) / 4.0,
            self.slots_filled / max(self.slots_total, 1),
            self.call_duration_s / 300.0,
            float(self.escalation_requested),
        ], dtype=np.float32)

    @property
    def dim(self) -> int:
        return 14


STATE_DIM = 14


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """
    Reward weights — centralised so they can be inspected and frozen.

    Design: shaped reward (dense) + terminal bonus/penalty.
    We deliberately avoid a single CSAT-only terminal signal because:
      1. It is too sparse for credit assignment across 12-turn calls.
      2. CSAT is sampled ~30% of calls and is noisy.
    Instead we combine:
      - Dense shaping from high-frequency, lower-noise signals.
      - Terminal signal from CSAT when available, escalation outcome.
    """

    # --- Dense per-turn signals ---
    w_correction_penalty:   float = -0.4   # user corrected the bot
    w_repeat_penalty:       float = -0.2   # bot repeated itself
    w_barge_in_penalty:     float = -0.1   # user interrupted (mild frustration)
    w_silence_penalty:      float = -0.05  # long silence (confusion)
    w_slot_fill_reward:     float =  0.3   # successfully filled a slot
    w_sentiment_reward:     float =  0.2   # sentiment improved this turn
    w_efficiency_bonus:     float =  0.05  # shorter call, same outcomes
    w_unnecessary_confirm:  float = -0.1   # CONFIRM when confidence was high

    # --- Terminal signals ---
    w_task_completion:      float =  2.0   # call ended with task done
    w_escalation_avoidable: float = -1.5   # escalated when didn't need to
    w_escalation_necessary: float =  0.5   # correct escalation
    w_csat_weight:          float =  1.0   # scales the CSAT bonus (0–5 → scaled)

    # --- Constraints ---
    confirm_low_threshold:  float = 0.6    # below this ASR conf → CONFIRM encouraged
    confirm_high_threshold: float = 0.85   # above this → penalise unnecessary CONFIRM

    def compute(
        self,
        state: ConversationState,
        action: Action,
        next_state: ConversationState,
        terminal: bool,
        task_completed: bool,
        escalation_was_necessary: Optional[bool],
        csat: Optional[float],           # None if not collected; 1–5 scale
    ) -> float:
        r = 0.0

        # Per-turn dense signals
        if next_state.n_corrections > state.n_corrections:
            r += self.w_correction_penalty

        if next_state.n_repeats > state.n_repeats:
            r += self.w_repeat_penalty

        if next_state.n_barge_ins > state.n_barge_ins:
            r += self.w_barge_in_penalty

        if next_state.last_silence_duration > 4.0:
            r += self.w_silence_penalty

        if next_state.slots_filled > state.slots_filled:
            r += self.w_slot_fill_reward

        if next_state.sentiment_delta > 0.1:
            r += self.w_sentiment_reward

        # Penalise unnecessary CONFIRM
        if action == Action.CONFIRM and state.asr_confidence > self.confirm_high_threshold:
            r += self.w_unnecessary_confirm

        # Small efficiency bonus (not dominant — don't want to rush callers)
        if next_state.call_duration_s < state.call_duration_s + 5:
            r += self.w_efficiency_bonus * 0.1

        # Terminal
        if terminal:
            if task_completed:
                r += self.w_task_completion

            if escalation_was_necessary is not None:
                if action == Action.ESCALATE or next_state.escalation_requested:
                    if escalation_was_necessary:
                        r += self.w_escalation_necessary
                    else:
                        r += self.w_escalation_avoidable

            if csat is not None:
                # CSAT 1–5 → centred at 3 → [-2, 2] → scaled
                csat_centred = (csat - 3.0) / 2.0
                r += self.w_csat_weight * csat_centred

        return float(r)


# ---------------------------------------------------------------------------
# Reward hacking analysis (documented, not runnable)
# ---------------------------------------------------------------------------

REWARD_HACK_ANALYSIS = """
Three policies that score well on this reward while making the product worse:

1. "Slot-farming PROCEED":
   Always choose PROCEED regardless of ASR confidence to accumulate slot_fill_reward
   fast. High slots_filled, but with wrong data — the task appears complete but the
   booking/info is incorrect. This slips through if we don't validate slot content.
   Mitigation: slot_fill_reward is only granted if NLU confidence also passes threshold.
   Still gameable if NLU is miscalibrated. Not fully caught.

2. "Sycophantic CONFIRM loop":
   On turns where ASR conf is just below confirm_high_threshold (e.g. 0.84), always
   CONFIRM to avoid the unnecessary-confirm penalty, even when the utterance is clear
   from context. This makes the call longer and more annoying but avoids penalty.
   Mitigation: we track n_repeats and penalise. Partially caught.

3. "Strategic escalation timing":
   Wait until sentiment is very negative, then ESCALATE — collecting the
   w_escalation_necessary bonus while offloading the hard call to a human.
   The bot learns to *let* calls deteriorate to earn the "correct escalation" bonus.
   Mitigation: hardest to catch. We add a penalty proportional to how long the
   bot waited before escalating a call that clearly needed it (proxy: n_corrections > 3
   before escalation → reduced necessary-escalation bonus). Partially caught.
"""

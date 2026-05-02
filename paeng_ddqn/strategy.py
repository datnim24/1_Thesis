"""Paeng's Modified DDQN — engine adapter + Family-Based State (FBS) builder.

Bridges our `SimulationEngine` (which calls `decide(state, roaster_id)` and
`decide_restock(state)`) to the agent's discrete 8-action Q-head.

Components:
    build_paeng_state(data, sim_state, context)  -> (state_3x50, auxin_10)
    compute_feasibility_mask(engine, state, ctx) -> bool[8]
    PaengStrategy                                 -> engine adapter

Action ID → engine-tuple mapping (Paeng-DDQN-internal → engine actions):
    0 PSC    : ("PSC", line) on roaster context, infeasible on restock context
    1 NDG    : ("NDG",) on R1/R2 context, infeasible elsewhere
    2 BUSTA  : ("BUSTA",) on R2 context, infeasible elsewhere
    3 WAIT   : ("WAIT",) — always feasible
    4 RST L1_PSC   : restock context only
    5 RST L1_NDG   : restock context only
    6 RST L1_BUSTA : restock context only
    7 RST L2_PSC   : restock context only

R3 routing for action 0 is baked in here (decision D2): when the queried
roaster is R3 and the agent picks PSC, the line is chosen by
``argmax(min(rc_space, gc_psc))`` — same rule that landed the +14% gain
in `rl_hh/tools.py`. No separate action dim needed.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paeng_ddqn.agent import PaengAgent

# ---------------------------------------------------------------------------
# Constants — fixed by problem domain
# ---------------------------------------------------------------------------

SKUS: tuple[str, ...] = ("PSC", "NDG", "BUSTA")
SKU_INDEX: dict[str, int] = {s: i for i, s in enumerate(SKUS)}

ROASTERS: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")
ROASTER_INDEX: dict[str, int] = {r: i for i, r in enumerate(ROASTERS)}

# Each roaster's pipeline (consumes from this line's GC pipeline)
ROASTER_PIPELINE: dict[str, str] = {"R1": "L1", "R2": "L1", "R3": "L2", "R4": "L2", "R5": "L2"}

# Each roaster's PSC output line; R3 is flexible (decided per-batch)
PSC_OUTPUT_LINE: dict[str, str | None] = {
    "R1": "L1", "R2": "L1", "R3": None, "R4": "L2", "R5": "L2",
}

# Roaster-SKU eligibility
ROASTER_SKU_ELIGIBLE: dict[str, set[str]] = {
    "R1": {"PSC", "NDG"},
    "R2": {"PSC", "NDG", "BUSTA"},
    "R3": {"PSC"},
    "R4": {"PSC"},
    "R5": {"PSC"},
}

# Restock action mapping
RESTOCK_ACTION_MAP: dict[int, tuple[str, str]] = {
    4: ("L1", "PSC"),
    5: ("L1", "NDG"),
    6: ("L1", "BUSTA"),
    7: ("L2", "PSC"),
}


# ---------------------------------------------------------------------------
# State builder — produces (3, 50) family-based state matrix + (10,) auxin
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if b not in (0, 0.0) else default


def build_paeng_state(
    data,
    sim_state,
    context: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 2D family-based state matrix for Paeng's DDQN.

    Args:
        data:       DataLoader-shaped object (or a params dict). job_sku, job_due,
                    consume_events live in params; we read both shapes via a helper.
        sim_state:  current SimulationState
        context:    {'kind': 'roaster' | 'restock', 'roaster_id': str | None}

    Returns:
        state:  np.ndarray[float32] of shape (3, 50). Row i = features for SKU i.
        auxin:  np.ndarray[float32] of shape (10,). System-level vector.

    Width = 50 columns per row. Layout documented in PORT_NOTES.md §3.

    All entries normalized to [-1, 1] roughly so the parameter-sharing block
    treats each family symmetrically.
    """
    F = 3
    W = 50

    # Resolve params: data is either a DataLoader-like object (has .to_env_params())
    # or already a params dict.
    if hasattr(data, "to_env_params"):
        params = data.to_env_params()
    else:
        params = data

    # Pull frequently-used scalars once. Prefer .attr access on data if it's
    # a DataLoader; fall back to params dict.
    def _get(attr_name, default=None):
        if hasattr(data, attr_name):
            return getattr(data, attr_name)
        return params.get(attr_name, default)

    SL = float(_get("shift_length", params.get("SL", 480)))
    max_rc = float(_get("max_rc", 40))
    roast_time_by_sku = _get("roast_time_by_sku", {"PSC": 15, "NDG": 17, "BUSTA": 18})
    max_proc = float(max(roast_time_by_sku.values()))
    gc_capacity = _get("gc_capacity", {})
    job_sku_map = params.get("job_sku", {})              # {job_id: sku}
    job_due = params.get("job_due", {})                   # {job_id: due_slot}
    job_batches_total = params.get("job_batches", _get("job_batches", {}))

    consume_events = params.get("consume_events", _get("consume_events", {"L1": [], "L2": []}))
    consume_rate_l1 = _safe_div(len(consume_events.get("L1", [])), SL)
    consume_rate_l2 = _safe_div(len(consume_events.get("L2", [])), SL)

    # Pre-compute "completed batches per SKU"
    completed_per_sku = {sku: 0 for sku in SKUS}
    for b in sim_state.completed_batches:
        if b.sku in completed_per_sku:
            completed_per_sku[b.sku] += 1

    # Pre-compute "in-progress remaining buckets" per SKU (Hp=5)
    Hp = 5
    in_progress_buckets = {sku: [0] * Hp for sku in SKUS}
    bucket_size = max(1.0, max_proc / Hp)
    for r in ROASTERS:
        if sim_state.status.get(r) == "RUNNING":
            cb = sim_state.current_batch.get(r)
            if cb is not None and cb.sku in in_progress_buckets:
                rem = sim_state.remaining.get(r, 0)
                bidx = min(Hp - 1, int(rem // bucket_size))
                in_progress_buckets[cb.sku][bidx] += 1

    # MTO remaining + due slack per MTO SKU (PSC has none → zeros)
    mto_remaining_per_sku = {sku: 0 for sku in SKUS}
    mto_due_min_per_sku = {sku: SL for sku in SKUS}  # SL = "no urgency" sentinel
    for jid, remaining in sim_state.mto_remaining.items():
        sku = job_sku_map.get(jid)
        if sku in mto_remaining_per_sku:
            mto_remaining_per_sku[sku] += int(remaining)
            due = float(job_due.get(jid, SL))
            slack = due - float(sim_state.t)
            if slack < mto_due_min_per_sku[sku]:
                mto_due_min_per_sku[sku] = slack

    total_mto_initial = max(1, sum(job_batches_total.values()))

    # Pre-compute per-roaster scalars used in multiple rows
    roaster_status_running_sku = {r: sim_state.current_batch[r].sku
                                  if (sim_state.status.get(r) == "RUNNING"
                                      and sim_state.current_batch.get(r) is not None)
                                  else None
                                  for r in ROASTERS}

    # Build rows
    state = np.zeros((F, W), dtype=np.float32)

    for sku_idx, sku in enumerate(SKUS):
        col = 0
        # 1. completed_count_norm (PSC normalized by ~120; MTO by 5 each → cap at 1.0)
        max_expected = 120.0 if sku == "PSC" else 5.0
        state[sku_idx, col] = min(1.0, _safe_div(completed_per_sku[sku], max_expected)); col += 1

        # 2. in_progress_count (any roaster currently producing this SKU)
        in_prog = sum(1 for r in ROASTERS if roaster_status_running_sku[r] == sku)
        state[sku_idx, col] = _safe_div(in_prog, len(ROASTERS)); col += 1

        # 3-7. in_progress_remaining_buckets (Hp=5)
        for b in in_progress_buckets[sku]:
            state[sku_idx, col] = _safe_div(b, len(ROASTERS)); col += 1

        # 8. mto_remaining_norm
        state[sku_idx, col] = _safe_div(mto_remaining_per_sku[sku], total_mto_initial); col += 1

        # 9. mto_due_slack_norm (positive when slack > 0; negative when overdue)
        state[sku_idx, col] = max(-1.0, min(1.0, _safe_div(mto_due_min_per_sku[sku], SL))); col += 1

        # 10. eligible_roaster_count_norm
        elig = sum(1 for r in ROASTERS if sku in ROASTER_SKU_ELIGIBLE[r])
        state[sku_idx, col] = _safe_div(elig, len(ROASTERS)); col += 1

        # 11-15. setup_required_flags per roaster (1 if last_sku != this sku AND eligible)
        for r in ROASTERS:
            if sku in ROASTER_SKU_ELIGIBLE[r] and sim_state.last_sku.get(r) != sku:
                state[sku_idx, col] = 1.0
            col += 1

        # 16-20. last_sku_match per roaster
        for r in ROASTERS:
            if sim_state.last_sku.get(r) == sku:
                state[sku_idx, col] = 1.0
            col += 1

        # 21-25. roaster_running_this_sku
        for r in ROASTERS:
            if roaster_status_running_sku[r] == sku:
                state[sku_idx, col] = 1.0
            col += 1

        # 26-30. roaster_idle_and_eligible
        for r in ROASTERS:
            if sim_state.status.get(r) == "IDLE" and sku in ROASTER_SKU_ELIGIBLE[r]:
                state[sku_idx, col] = 1.0
            col += 1

        # 31. gc_stock_norm L1
        cap_l1 = float(gc_capacity.get(("L1", sku), 0))
        state[sku_idx, col] = _safe_div(sim_state.gc_stock.get(("L1", sku), 0), cap_l1); col += 1
        # 32. gc_stock_norm L2
        cap_l2 = float(gc_capacity.get(("L2", sku), 0))
        state[sku_idx, col] = _safe_div(sim_state.gc_stock.get(("L2", sku), 0), cap_l2); col += 1

        # 33-34. rc_norm per line (only meaningful for PSC; zero rows for MTO)
        if sku == "PSC":
            state[sku_idx, col] = _safe_div(sim_state.rc_stock.get("L1", 0), max_rc); col += 1
            state[sku_idx, col] = _safe_div(sim_state.rc_stock.get("L2", 0), max_rc); col += 1
        else:
            col += 2

        # 35-36. pipeline_busy per line
        state[sku_idx, col] = _safe_div(sim_state.pipeline_busy.get("L1", 0), 15.0); col += 1
        state[sku_idx, col] = _safe_div(sim_state.pipeline_busy.get("L2", 0), 15.0); col += 1

        # 37. proc_time_norm
        state[sku_idx, col] = _safe_div(roast_time_by_sku.get(sku, 0), max_proc); col += 1

        # 38-39. consume_rate per line (PSC only; zero rows for MTO)
        if sku == "PSC":
            state[sku_idx, col] = consume_rate_l1; col += 1
            state[sku_idx, col] = consume_rate_l2; col += 1
        else:
            col += 2

        # 40-44. roaster_remaining_norm per roaster
        for r in ROASTERS:
            state[sku_idx, col] = _safe_div(sim_state.remaining.get(r, 0), max_proc); col += 1

        # 45-49. roaster_setup_progress (only nonzero on roasters in SETUP for this sku)
        sigma = float(_get("setup_time", params.get("sigma", 5)))
        for r in ROASTERS:
            if (sim_state.status.get(r) == "SETUP"
                    and sim_state.setup_target_sku.get(r) == sku):
                progress = (sigma - sim_state.remaining.get(r, 0)) / max(1.0, sigma)
                state[sku_idx, col] = float(max(0.0, min(1.0, progress)))
            col += 1

        # 50. time_progress (same value across SKU rows but ensures the fixed width)
        state[sku_idx, col] = _safe_div(sim_state.t, SL); col += 1

        assert col == W, f"State row width mismatch: built {col} cols, expected {W}"

    # ---- Auxin (10) — system-level / context vector ----
    auxin = np.zeros(10, dtype=np.float32)
    last_action = context.get("last_action_id", -1)
    if 0 <= last_action < 8:
        auxin[last_action] = 1.0
    auxin[8] = float(context.get("last_reward_norm", 0.0))
    auxin[9] = 1.0 if context.get("kind") == "restock" else 0.0

    return state, auxin


# ---------------------------------------------------------------------------
# Feasibility mask — tells the agent which actions are valid in this context
# ---------------------------------------------------------------------------

def compute_feasibility_mask(
    engine,
    sim_state,
    context: dict,
) -> np.ndarray:
    """Return bool[8] indicating which of the 8 actions are valid.

    Uses ``engine.can_start_batch`` and ``engine.can_start_restock`` (the same
    methods CP-SAT and the GUI use), so feasibility is identical to what the
    engine would accept. Inconsistencies here would silently wedge training.
    """
    mask = np.zeros(8, dtype=bool)
    kind = context.get("kind")
    rid = context.get("roaster_id")

    if kind == "roaster" and rid is not None:
        # Action 0: PSC — must be eligible AND startable on at least one line
        if "PSC" in ROASTER_SKU_ELIGIBLE.get(rid, set()):
            if rid == "R3":
                feas_l1 = engine.can_start_batch(sim_state, rid, "PSC", "L1")
                feas_l2 = engine.can_start_batch(sim_state, rid, "PSC", "L2")
                mask[0] = feas_l1 or feas_l2
            else:
                line = PSC_OUTPUT_LINE.get(rid)
                if line is not None:
                    mask[0] = engine.can_start_batch(sim_state, rid, "PSC", line)

        # Action 1: NDG — only R1/R2
        if "NDG" in ROASTER_SKU_ELIGIBLE.get(rid, set()):
            mask[1] = engine.can_start_batch(sim_state, rid, "NDG", None)

        # Action 2: BUSTA — only R2
        if "BUSTA" in ROASTER_SKU_ELIGIBLE.get(rid, set()):
            mask[2] = engine.can_start_batch(sim_state, rid, "BUSTA", None)

        # Action 3: WAIT — always feasible at a decision point
        mask[3] = True
        # Restock actions (4-7) infeasible in roaster context

    elif kind == "restock":
        # Roaster actions (0-2) infeasible in restock context
        mask[3] = True  # WAIT is always available
        for action_id, (line, sku) in RESTOCK_ACTION_MAP.items():
            mask[action_id] = engine.can_start_restock(sim_state, line, sku)

    return mask


# ---------------------------------------------------------------------------
# Strategy adapter (mirrors q_learning.QStrategy / rl_hh.RLHHStrategy interface)
# ---------------------------------------------------------------------------

class PaengStrategy:
    """Plug-in strategy for ``SimulationEngine.run(strategy, ups)``.

    Holds the agent + maintains rolling (prev_state, prev_action, prev_profit)
    so we can compute per-decision rewards = Δ(net_profit) and store
    transitions during training.
    """

    def __init__(self, agent: PaengAgent, data, training: bool = False):
        from env.simulation_engine import SimulationEngine
        from dispatch.dispatching_heuristic import DispatchingHeuristic

        self.agent = agent
        self.data = data
        self.training = training

        # Engine handle for can_start_batch / can_start_restock (cheap clone OK
        # since SimulationEngine is mostly stateless wrt mask checks).
        params = data.to_env_params() if hasattr(data, "to_env_params") else data
        self._engine = SimulationEngine(params)
        # Cycle 41 (2026-04-29): delegate restock decisions to dispatching heuristic
        # (matches Q-learning's q_learning_train.py pattern). Reduces effective action
        # space from 8 -> 4 and decision count by ~50%, mirroring tabular Q-learning.
        self._restock_heuristic = DispatchingHeuristic(params)

        # Rolling state for reward computation
        self._prev_state: np.ndarray | None = None
        self._prev_auxin: np.ndarray | None = None
        self._prev_action: int = -1
        self._prev_mask: np.ndarray | None = None
        self._prev_profit: float = 0.0
        self._prev_reward_norm: float = 0.0
        self._prev_gc_low: bool = False  # Cycle 36: tracks if previous decision had GC silos below threshold
        # Cycle 16: curriculum — penalty only after N episodes. Train.py increments via reset_episode().
        self._episode_idx: int = 0
        # Cycle 36: cache GC capacities for low-threshold check.
        params = data.to_env_params() if hasattr(data, "to_env_params") else data
        self._gc_capacity = params.get("gc_capacity", {})
        # KPI handle is set externally if available; else we approximate from state.completed_batches
        self.kpi_ref = None

        # Action distribution diagnostics (maps directly to the 8 action IDs)
        self.action_counts: dict[int, int] = {i: 0 for i in range(8)}
        # Backwards-compat with rl_hh/test_rl_hh evaluators that look for tool_counts
        self.tool_counts = self.action_counts

    # -- engine-facing decision methods ------------------------------------

    def decide(self, sim_state, roaster_id: str) -> tuple:
        return self._step(sim_state, context={"kind": "roaster", "roaster_id": roaster_id})

    def decide_restock(self, sim_state) -> tuple:
        # Cycle 41/45: optionally delegate restock to dispatching heuristic.
        if self.agent.cfg.delegate_restock:
            return self._restock_heuristic.decide_restock(sim_state)
        return self._step(sim_state, context={"kind": "restock", "roaster_id": None})

    # -- internals ---------------------------------------------------------

    def _step(self, sim_state, context: dict) -> tuple:
        """Build state, query agent, store transition (if training), return env tuple."""
        context_full = dict(context)
        context_full["last_action_id"] = self._prev_action
        context_full["last_reward_norm"] = self._prev_reward_norm

        state_arr, auxin_arr = build_paeng_state(self.data, sim_state, context_full)
        mask = compute_feasibility_mask(self._engine, sim_state, context_full)

        # Cycle 36: compute current GC-low flag (avg fill across all silos < threshold).
        gc_low_now = False
        if self.agent.cfg.stockout_alarm_penalty > 0.0:
            total_fill, total_cap = 0.0, 0.0
            for (line, sku), cap in self._gc_capacity.items():
                if cap > 0:
                    total_fill += float(sim_state.gc_stock.get((line, sku), 0))
                    total_cap += float(cap)
            if total_cap > 0:
                gc_low_now = (total_fill / total_cap) < self.agent.cfg.gc_low_threshold

        # Cycle 25 (2026-04-28): force productive — with probability p, mask out WAIT
        # whenever any productive action (0,1,2) is feasible. Training-only.
        # Forces productive experiences into replay buffer.
        if (
            self.training
            and self.agent.cfg.force_productive_prob > 0.0
            and (mask[0] or mask[1] or mask[2])
            and random.random() < self.agent.cfg.force_productive_prob
        ):
            mask = mask.copy()
            mask[3] = False

        action_id = self.agent.select_action(state_arr, auxin_arr, mask, training=self.training)
        self.action_counts[action_id] = self.action_counts.get(action_id, 0) + 1

        # If training: store the transition from the PREVIOUS step (s, a, r, s')
        # where s' is the current build, r is profit-delta since previous decision,
        # divided by reward_scale (Cycle 1 fix — Paeng's rnorm analog).
        if self.training and self._prev_state is not None:
            current_profit = self._compute_profit(sim_state)
            raw_reward = current_profit - self._prev_profit
            scaled_reward = raw_reward / self.agent.cfg.reward_scale
            # Cycle 4 (2026-04-28): penalize WAIT chosen while productive was feasible.
            # Cycle 8 reverted: restock-WAIT extension regressed in 100-seed mean.
            # Cycle 16 (2026-04-28): curriculum — only apply penalty after warmup episodes.
            if (
                self._prev_action == 3
                and self._prev_mask is not None
                and (bool(self._prev_mask[0]) or bool(self._prev_mask[1]) or bool(self._prev_mask[2]))
                and self._episode_idx >= self.agent.cfg.curriculum_warmup_episodes
            ):
                scaled_reward -= self.agent.cfg.idle_penalty
            # Cycle 29 (2026-04-29): productive-action bonus (symmetric to idle penalty).
            if (
                self._prev_action in (0, 1, 2)
                and self.agent.cfg.productive_bonus > 0.0
                and self._episode_idx >= self.agent.cfg.curriculum_warmup_episodes
            ):
                scaled_reward += self.agent.cfg.productive_bonus
            # Cycle 36 (2026-04-29): stockout-prevention. If avg GC fill < threshold and agent
            # didn't restock (action 4-7), apply stockout_alarm_penalty.
            if (
                self.agent.cfg.stockout_alarm_penalty > 0.0
                and self._prev_action not in (4, 5, 6, 7)
                and self._episode_idx >= self.agent.cfg.curriculum_warmup_episodes
                and self._prev_gc_low
            ):
                scaled_reward -= self.agent.cfg.stockout_alarm_penalty
            # Cycle 22 (2026-04-28): reward clipping for Q-stability (Atari DQN style).
            if self.agent.cfg.reward_clip > 0:
                scaled_reward = float(np.clip(scaled_reward, -self.agent.cfg.reward_clip, self.agent.cfg.reward_clip))
            # Reward normalization for auxin (cap at ±10k per decision raw)
            self._prev_reward_norm = float(np.clip(raw_reward / 10_000.0, -1.0, 1.0))
            self.agent.store_transition(
                self._prev_state, self._prev_auxin, self._prev_action,
                scaled_reward, False,  # terminal=False inside an episode
                state_arr, auxin_arr, mask,
            )
            self.agent.train_step()
            self._prev_profit = current_profit
        elif self.training:
            # First decision of episode — establish baseline profit
            self._prev_profit = self._compute_profit(sim_state)

        # Roll prev pointers
        self._prev_state = state_arr.copy()
        self._prev_auxin = auxin_arr.copy()
        self._prev_action = action_id
        self._prev_mask = mask.copy()
        self._prev_gc_low = gc_low_now  # Cycle 36

        return self._action_to_env_tuple(action_id, context, sim_state)

    def end_episode(self, _sim_state, final_profit: float) -> None:
        """Called by train.py at episode end to push the terminal transition.

        sim_state is taken for API symmetry with `decide` even though we don't
        currently use it inside (could be used to build a true terminal s').
        """
        if not self.training or self._prev_state is None:
            return
        raw_reward = final_profit - self._prev_profit
        reward = raw_reward / self.agent.cfg.reward_scale
        if self.agent.cfg.reward_clip > 0:
            reward = float(np.clip(reward, -self.agent.cfg.reward_clip, self.agent.cfg.reward_clip))
        # Cycle 4: terminal-step idle penalty for symmetry with _step (reverted to productive-only after cycle 8 regression).
        # Cycle 16: curriculum guard.
        if (
            self._prev_action == 3
            and self._prev_mask is not None
            and (bool(self._prev_mask[0]) or bool(self._prev_mask[1]) or bool(self._prev_mask[2]))
            and self._episode_idx >= self.agent.cfg.curriculum_warmup_episodes
        ):
            reward -= self.agent.cfg.idle_penalty
        # Terminal next-state: zeros (mask all-WAIT-only)
        zeros = np.zeros_like(self._prev_state)
        zeros_aux = np.zeros_like(self._prev_auxin)
        terminal_mask = np.array([False, False, False, True, False, False, False, False])
        self.agent.store_transition(
            self._prev_state, self._prev_auxin, self._prev_action,
            reward, True, zeros, zeros_aux, terminal_mask,
        )
        self.agent.train_step()

        # Reset rolling state for next episode
        self._prev_state = None
        self._prev_auxin = None
        self._prev_action = -1
        self._prev_profit = 0.0
        self._prev_reward_norm = 0.0

    def reset_episode(self) -> None:
        """Called by train.py at episode start to clear rolling state."""
        self._prev_state = None
        self._prev_auxin = None
        self._prev_action = -1
        self._prev_profit = 0.0
        self._prev_reward_norm = 0.0
        self._prev_gc_low = False
        self._episode_idx += 1
        # Don't reset action_counts — it's per-evaluation-batch by convention,
        # caller can clear manually for diagnostics.

    # -- action-id → engine tuple ------------------------------------------

    def _action_to_env_tuple(
        self,
        action_id: int,
        context: dict,
        sim_state,
    ) -> tuple:
        if action_id == 3:
            return ("WAIT",)

        if context.get("kind") == "restock":
            if action_id in RESTOCK_ACTION_MAP:
                line, sku = RESTOCK_ACTION_MAP[action_id]
                return ("START_RESTOCK", line, sku)
            return ("WAIT",)

        # Roaster context: action 0=PSC, 1=NDG, 2=BUSTA
        rid = context.get("roaster_id")
        if action_id == 0:
            if rid == "R3":
                line = self._r3_route(sim_state)
                return ("PSC", line)
            line = PSC_OUTPUT_LINE.get(rid, "L1")
            return ("PSC", line)
        if action_id == 1:
            return ("NDG",)
        if action_id == 2:
            return ("BUSTA",)
        return ("WAIT",)

    def _r3_route(self, sim_state) -> str:
        """R3 cross-line routing — argmax(min(rc_space, gc_psc)).

        Identical to ``rl_hh.tools.ToolKit._psc_throughput`` R3 branch
        (decision D2). Tie goes to L2 to match cycle-10 result.
        """
        max_rc = float(self.data.max_rc)
        rc_l1 = float(sim_state.rc_stock.get("L1", 0))
        rc_l2 = float(sim_state.rc_stock.get("L2", 0))
        gc_l1 = float(sim_state.gc_stock.get(("L1", "PSC"), 0))
        gc_l2 = float(sim_state.gc_stock.get(("L2", "PSC"), 0))
        score_l1 = min(max_rc - rc_l1, gc_l1)
        score_l2 = min(max_rc - rc_l2, gc_l2)
        return "L1" if score_l1 > score_l2 else "L2"

    # -- profit reward source ----------------------------------------------

    def _compute_profit(self, sim_state) -> float:
        """Per-decision reward source.

        When ``cfg.use_kpi_for_reward=True`` (and ``kpi_ref`` is wired by the
        engine), returns the live ``kpi.net_profit()`` — the semantically
        correct profit signal including idle/setup/stockout/tardiness costs.

        When ``cfg.use_kpi_for_reward=False`` (default after cycle 44), returns
        the revenue-only approximation from completed_batches. This matches the
        signal cycle 13 trained under (the kpi_ref bug pre-fix) and produced
        the +$45k checkpoint. The terminal transition in ``end_episode`` still
        uses the true ``final_profit`` so the agent eventually sees costs.
        """
        if self.agent.cfg.use_kpi_for_reward and self.kpi_ref is not None:
            return float(self.kpi_ref.net_profit())
        revenue_per_sku = {"PSC": 4000.0, "NDG": 7000.0, "BUSTA": 7000.0}
        rev = sum(revenue_per_sku.get(b.sku, 0.0) for b in sim_state.completed_batches)
        return rev

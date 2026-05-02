"""Paeng v2 strategy — period-based dispatch with faithful Table 1 (3,25) state.

State layout per Paeng 2021 IEEE Access, Table 1 (NM-independent):
    Row f ∈ {PSC=0, NDG=1, BUSTA=2}, total cols = 25:
      [0:6]   Sw_count   — waiting-job slack histogram (Hw=6 buckets), normalized count
      [6:12]  Sw_slack   — same buckets, normalized slack-sum
      [12:17] Sp         — in-progress remaining-time histogram (Hp=5 buckets)
      [17:20] Ss         — setup-status to/from dominant setup (3 cols)
      [20:23] Su         — machine util fractions (proc, setup, idle) per family
      [23:25] Sa         — last action one-hot (f_last col 23, g_last col 24)
    Auxin vec(Sf) (3,)   — normalized expected processing time per family

Domain-specific adaptations (see PORT_NOTES_v2.md):
    PSC waiting "slack" derived from upcoming consume_events vs RC stock
    NDG/BUSTA waiting slack derived from job_due[jid] - state.t for active MTO jobs
    Setup is uniform σ (no S_ij matrix) → Ss collapses to 0/1 indicator
    Roaster eligibility: R1{PSC,NDG}, R2{PSC,NDG,BUSTA}, R3/R4/R5{PSC}
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paeng_ddqn_v2.agent_v2 import PaengAgentV2

# ---------------------------------------------------------------------------
# Domain constants (mirroring paeng_ddqn/strategy.py)
# ---------------------------------------------------------------------------

SKUS: tuple[str, ...] = ("PSC", "NDG", "BUSTA")
SKU_INDEX: dict[str, int] = {s: i for i, s in enumerate(SKUS)}

ROASTERS: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")

ROASTER_SKU_ELIGIBLE: dict[str, set[str]] = {
    "R1": {"PSC", "NDG"},
    "R2": {"PSC", "NDG", "BUSTA"},
    "R3": {"PSC"},
    "R4": {"PSC"},
    "R5": {"PSC"},
}

PSC_OUTPUT_LINE: dict[str, str | None] = {
    "R1": "L1", "R2": "L1", "R3": None, "R4": "L2", "R5": "L2",
}

# Slack bucket boundaries (Hw=6): [-∞, -T, 0, T, 2T, 3T, +∞]
HW_BUCKETS = 6
HP_BUCKETS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_params(data, params=None) -> dict:
    """Return params dict from either explicit `params` or `data.to_env_params()`."""
    if params is not None:
        return params
    if hasattr(data, "to_env_params"):
        return data.to_env_params()
    return data if isinstance(data, dict) else {}


def _bucket_index_slack(slack: float, T: int) -> int:
    """Bin a slack value into one of HW_BUCKETS=6: [-∞,-T) [-T,0) [0,T) [T,2T) [2T,3T) [3T,+∞)."""
    if slack < -T:
        return 0
    if slack < 0:
        return 1
    if slack < T:
        return 2
    if slack < 2 * T:
        return 3
    if slack < 3 * T:
        return 4
    return 5


def _roaster_can_produce(roaster_id: str, sku: str) -> bool:
    return sku in ROASTER_SKU_ELIGIBLE.get(roaster_id, set())


def _has_waiting_demand(family: str, sim_state, params: dict, period_T: int) -> bool:
    """True if there is current demand for `family` that the agent could service.

    PSC: continuous demand throughout shift (consume_events keep firing). Demand True
        iff ANY future consume_event exists for some line AND that line's PSC GC silo
        has room (so dispatched PSC has somewhere to push output). This prevents the
        Cycle 9 bug where R3/R4/R5 idled at shift start because RC was full and GC was
        slightly above the 30% threshold, making greedy fallback return WAIT.
    NDG/BUSTA: demand iff any MTO job of that sku still has remaining > 0.
    """
    del period_T  # unused (kept for API compatibility)
    if family == "PSC":
        consume_events = params.get("consume_events", {"L1": [], "L2": []})
        gc_capacity = params.get("gc_capacity", {})
        for line in ("L1", "L2"):
            has_future_consume = any(t > sim_state.t for t in consume_events.get(line, []))
            if not has_future_consume:
                continue
            cap = gc_capacity.get((line, "PSC"), 0)
            gc = sim_state.gc_stock.get((line, "PSC"), 0)
            # If we know capacity, only "demand" when room remains; if capacity unknown,
            # default to True (let engine handle silo limits).
            if cap == 0 or gc < cap:
                return True
        return False
    # NDG / BUSTA
    job_sku = params.get("job_sku", {})
    for jid, remaining in sim_state.mto_remaining.items():
        if remaining > 0 and job_sku.get(jid) == family:
            return True
    return False


# ---------------------------------------------------------------------------
# State builder — Table 1 (3, 25)
# ---------------------------------------------------------------------------

def build_state_v2(
    data,
    sim_state,
    params: dict | None = None,
    last_action_id: int | None = None,
    period_T: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the (3, 35) state + (3,) Sf auxin.

    Cols 0-24: paper-faithful Paeng Table 1 (Sw 12 + Sp 5 + Ss 3 + Su 3 + Sa 2).
    Cols 25-34: domain-specific features (RC/GC/pipeline/time/UPS) — domain
        adaptation since our env has pipeline blocking + inventory mgmt that pure UPMSP
        does not. See PORT_NOTES_v2.md for justification.
    """
    params = _resolve_params(data, params)
    SL = float(params.get("SL", 480))
    max_rc = float(params.get("max_rc", 40))
    gc_capacity = params.get("gc_capacity", {})
    roast_time_by_sku = (
        getattr(data, "roast_time_by_sku", None)
        or params.get("roast_time_by_sku", {"PSC": 15, "NDG": 17, "BUSTA": 18})
    )
    max_proc_time = float(max(roast_time_by_sku.values())) if roast_time_by_sku else 1.0
    consume_events = params.get("consume_events", {"L1": [], "L2": []})
    job_sku = params.get("job_sku", {})
    job_due = params.get("job_due", {})

    NM = max(1, len(ROASTERS))

    state = np.zeros((3, 35), dtype=np.float32)
    sf = np.zeros(3, dtype=np.float32)

    # --- Compute dominant setup once ---
    last_skus = [sim_state.last_sku.get(r) for r in ROASTERS if sim_state.last_sku.get(r)]
    dominant_setup = Counter(last_skus).most_common(1)[0][0] if last_skus else "PSC"

    # --- Build per-family ---
    for f_idx, family in enumerate(SKUS):

        # ===== Sw (3 × 12): waiting-slack histogram =====
        slack_bucket_count = [0] * HW_BUCKETS
        slack_bucket_sum = [0.0] * HW_BUCKETS

        if family == "PSC":
            # Project RC depletion: each upcoming consume_event decrements RC by 1
            for line in ("L1", "L2"):
                rc = sim_state.rc_stock.get(line, 0)
                upcoming = sorted(t for t in consume_events.get(line, []) if t > sim_state.t)
                running_rc = rc
                for ce in upcoming:
                    running_rc -= 1
                    if running_rc < 0:
                        slack = ce - sim_state.t
                        bidx = _bucket_index_slack(slack, period_T)
                        slack_bucket_count[bidx] += 1
                        slack_bucket_sum[bidx] += slack
        else:
            # NDG / BUSTA: pull from active MTO jobs of this family
            for jid, remaining in sim_state.mto_remaining.items():
                if remaining <= 0 or job_sku.get(jid) != family:
                    continue
                due = float(job_due.get(jid, SL))
                slack = due - float(sim_state.t)
                bidx = _bucket_index_slack(slack, period_T)
                slack_bucket_count[bidx] += int(remaining)  # qty-weighted
                slack_bucket_sum[bidx] += slack * int(remaining)

        # Normalize: count by 10, slack-sum by SL, both clipped
        for b in range(HW_BUCKETS):
            state[f_idx, b] = float(np.clip(slack_bucket_count[b] / 10.0, 0.0, 1.0))
            state[f_idx, HW_BUCKETS + b] = float(
                np.clip(slack_bucket_sum[b] / SL, -1.0, 1.0)
            )

        # ===== Sp (3 × 5): in-progress remaining-time buckets =====
        proc_f = float(roast_time_by_sku.get(family, max_proc_time))
        bucket_size_p = max(1.0, proc_f / HP_BUCKETS)
        sp_buckets = [0] * HP_BUCKETS
        for r in ROASTERS:
            if sim_state.status.get(r) != "RUNNING":
                continue
            cb = sim_state.current_batch.get(r)
            if cb is None or cb.sku != family:
                continue
            rem = float(sim_state.remaining.get(r, 0))
            bidx = min(HP_BUCKETS - 1, int(rem // bucket_size_p))
            sp_buckets[bidx] += 1
        for b in range(HP_BUCKETS):
            state[f_idx, 12 + b] = float(sp_buckets[b]) / NM

        # ===== Ss (3 × 3): setup status to/from dominant =====
        # Uniform σ → indicator-only
        is_dominant = 1.0 if family == dominant_setup else 0.0
        state[f_idx, 17] = 1.0 - is_dominant   # σ_to_f / σ ∈ {0, 1}
        state[f_idx, 18] = 1.0 - is_dominant   # σ_from_f / σ ∈ {0, 1}
        state[f_idx, 19] = is_dominant         # has_dominant_match

        # ===== Su (3 × 3): machine utilization per family =====
        n_proc = sum(
            1 for r in ROASTERS
            if sim_state.status.get(r) == "RUNNING"
            and sim_state.current_batch.get(r) is not None
            and sim_state.current_batch.get(r).sku == family
        )
        n_setup = sum(
            1 for r in ROASTERS
            if sim_state.status.get(r) == "SETUP"
            and sim_state.setup_target_sku.get(r) == family
        )
        n_idle = sum(
            1 for r in ROASTERS
            if sim_state.status.get(r) == "IDLE"
            and family in ROASTER_SKU_ELIGIBLE.get(r, set())
        )
        state[f_idx, 20] = float(n_proc) / NM
        state[f_idx, 21] = float(n_setup) / NM
        state[f_idx, 22] = float(n_idle) / NM

        # ===== Sa (3 × 2): last action one-hot — defer to outside loop =====

    # ===== Sa (last action): cols 23 (from_setup), 24 (to_dispatch) =====
    # Paeng convention: action_id = from_setup * F + to_dispatch
    if last_action_id is not None and 0 <= int(last_action_id) < 9:
        from_setup_last = int(last_action_id) // 3   # which family was the FROM-setup
        to_dispatch_last = int(last_action_id) % 3    # which family was DISPATCHED
        state[from_setup_last, 23] = 1.0
        state[to_dispatch_last, 24] = 1.0

    # ===== Domain extension cols 25-34: RC/GC/pipeline/time/UPS =====
    # Same value across rows where appropriate; per-family where it differs.
    time_progress = float(np.clip(sim_state.t / SL, 0.0, 1.0))
    rc_l1_norm = float(np.clip(sim_state.rc_stock.get("L1", 0) / max(max_rc, 1.0), 0.0, 1.0))
    rc_l2_norm = float(np.clip(sim_state.rc_stock.get("L2", 0) / max(max_rc, 1.0), 0.0, 1.0))
    pipe_l1 = float(np.clip(sim_state.pipeline_busy.get("L1", 0) / 15.0, 0.0, 1.0))
    pipe_l2 = float(np.clip(sim_state.pipeline_busy.get("L2", 0) / 15.0, 0.0, 1.0))

    for f_idx, family in enumerate(SKUS):
        # Cols 25-26: GC stock per (line, family) normalized
        cap_l1 = float(gc_capacity.get(("L1", family), 0))
        cap_l2 = float(gc_capacity.get(("L2", family), 0))
        gc_l1 = float(sim_state.gc_stock.get(("L1", family), 0))
        gc_l2 = float(sim_state.gc_stock.get(("L2", family), 0))
        state[f_idx, 25] = float(np.clip(gc_l1 / cap_l1, 0.0, 1.0)) if cap_l1 > 0 else 0.0
        state[f_idx, 26] = float(np.clip(gc_l2 / cap_l2, 0.0, 1.0)) if cap_l2 > 0 else 0.0
        # Col 27: RC norm averaged (only meaningful for PSC; zero for NDG/BUSTA)
        state[f_idx, 27] = (rc_l1_norm + rc_l2_norm) * 0.5 if family == "PSC" else 0.0
        # Cols 28-29: pipeline_busy per line (same across families — feeds for fusion)
        state[f_idx, 28] = pipe_l1
        state[f_idx, 29] = pipe_l2
        # Col 30: time_progress (replicated, same across rows)
        state[f_idx, 30] = time_progress
        # Col 31: n_idle_eligible / NM (per family)
        n_idle_elig = sum(
            1 for r in ROASTERS
            if sim_state.status.get(r) == "IDLE"
            and family in ROASTER_SKU_ELIGIBLE.get(r, set())
        )
        state[f_idx, 31] = float(n_idle_elig) / NM
        # Col 32: n_running_this_family / NM
        n_run = sum(
            1 for r in ROASTERS
            if sim_state.status.get(r) == "RUNNING"
            and sim_state.current_batch.get(r) is not None
            and sim_state.current_batch.get(r).sku == family
        )
        state[f_idx, 32] = float(n_run) / NM
        # Col 33: n_DOWN_eligible (UPS-affected) / NM
        n_down_elig = sum(
            1 for r in ROASTERS
            if sim_state.status.get(r) == "DOWN"
            and family in ROASTER_SKU_ELIGIBLE.get(r, set())
        )
        state[f_idx, 33] = float(n_down_elig) / NM
        # Col 34: setup-in-progress to this family (any roaster setting up to f)
        any_setup_to_family = any(
            sim_state.status.get(r) == "SETUP"
            and sim_state.setup_target_sku.get(r) == family
            for r in ROASTERS
        )
        state[f_idx, 34] = 1.0 if any_setup_to_family else 0.0

    # ===== vec(Sf) (3,): expected proc time per family, normalized =====
    for f_idx, family in enumerate(SKUS):
        sf[f_idx] = float(roast_time_by_sku.get(family, max_proc_time)) / max(max_proc_time, 1.0)

    return state, sf


# ---------------------------------------------------------------------------
# Feasibility mask — 9 actions, conditioned on calling roaster
# ---------------------------------------------------------------------------

def compute_feasibility_mask_v2(
    data,
    sim_state,
    params: dict | None = None,
    calling_roaster_id: str | None = None,
    period_T: int = 11,
) -> np.ndarray:
    """Mask 9 actions a = from_setup*F + to_dispatch (Paeng convention).

    Action (from, to) feasible iff:
        - to-family has waiting demand
        - SOME roaster currently has last_sku == from AND is eligible to produce to
          (because the action's "transition" must be matchable somewhere)

    `calling_roaster_id` is unused at mask level (period decision applies across
    all roasters via greedy fallback for non-matching ones).
    """
    del calling_roaster_id  # mask is period-level, not per-roaster
    params = _resolve_params(data, params)
    mask = np.zeros(9, dtype=bool)
    for from_idx, from_sku in enumerate(SKUS):
        for to_idx, to_sku in enumerate(SKUS):
            a = from_idx * 3 + to_idx
            if not _has_waiting_demand(to_sku, sim_state, params, period_T):
                continue
            # Some roaster must match (last_sku == from_sku AND eligible for to_sku)
            for r in ROASTERS:
                if (sim_state.last_sku.get(r) == from_sku
                        and to_sku in ROASTER_SKU_ELIGIBLE.get(r, set())):
                    mask[a] = True
                    break
    # Failsafe: if every action is masked False, allow all (engine WAITs handle the rest)
    if not mask.any():
        mask[:] = True
    return mask


# ---------------------------------------------------------------------------
# PaengStrategyV2 — period-based dispatch with per-period tardiness reward
# ---------------------------------------------------------------------------

class PaengStrategyV2:
    """Period-based strategy adapter. Decisions every `period_length` minutes.

    Per-period reward = -(kpi.tard_cost_now - kpi.tard_cost_period_start)
    via live `self.kpi_ref` set by SimulationEngine.run.
    """

    def __init__(
        self,
        agent: PaengAgentV2,
        data,
        training: bool = False,
        params: dict | None = None,
    ):
        self.agent = agent
        self.data = data
        self.training = training
        self.params = _resolve_params(data, params)

        # Period tracking
        self._period_boundary: int = 0
        self._period_action: int | None = None
        self._period_state: np.ndarray | None = None
        self._period_sf: np.ndarray | None = None
        self._period_feas: np.ndarray | None = None
        self._period_roaster_id: str | None = None
        self._prev_cost: float = 0.0   # -Δ(tard+idle+setup+stockout) reward
        self._prev_revenue: float = 0.0  # +Δrevenue * REVENUE_WEIGHT bonus
        self._last_action_id: int | None = None

        # Live KPI reference (set by SimulationEngine.run via hasattr check)
        self.kpi_ref = None

        # Diagnostics
        self.action_counts: dict[int, int] = {i: 0 for i in range(9)}
        self.reward_history: list[float] = []   # cleared each episode

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _kpi_total_cost(self, fallback: float = 0.0) -> float:
        """Sum of operational cost terms from live KPI: tard + idle + setup + stockout."""
        if self.kpi_ref is None:
            return fallback
        return float(
            self.kpi_ref.tard_cost
            + self.kpi_ref.idle_cost
            + self.kpi_ref.setup_cost
            + self.kpi_ref.stockout_cost
        )

    def _kpi_revenue(self) -> float:
        """Cumulative revenue from completed batches (positive completion signal)."""
        if self.kpi_ref is None:
            return 0.0
        return float(self.kpi_ref.total_revenue)

    def _compute_reward(self, current_cost: float, current_revenue: float) -> float:
        """Per-period reward = -Δcost + REVENUE_WEIGHT * Δrevenue.

        REVENUE_WEIGHT=0.1 gives a small but non-zero positive signal when batches
        complete, so the agent learns dispatching → completion → revenue is good
        (not just "less idle is good", which biases toward always-PSC collapse).
        """
        REVENUE_WEIGHT = 0.1
        cost_delta = current_cost - getattr(self, "_prev_cost", 0.0)
        revenue_delta = current_revenue - getattr(self, "_prev_revenue", 0.0)
        return -cost_delta + REVENUE_WEIGHT * revenue_delta

    def reset_episode(self) -> None:
        self._period_boundary = 0
        self._period_action = None
        self._period_state = None
        self._period_sf = None
        self._period_feas = None
        self._period_roaster_id = None
        self._prev_cost = 0.0
        self._prev_revenue = 0.0
        self._last_action_id = None
        self.action_counts = {i: 0 for i in range(9)}
        self.reward_history = []

    def end_episode(
        self,
        sim_state,
        final_profit: float,
        tardiness_cost: float = 0.0,
    ) -> None:
        """Store terminal transition with -Δcost reward and final train_step.

        `final_profit` and `tardiness_cost` are accepted for API compatibility but
        the actual reward is computed from `self.kpi_ref` (cost delta over the
        last period). If kpi_ref is unavailable, falls back to tardiness_cost.
        """
        del final_profit  # unused (kept in signature for API compat)
        if not self.training or self._period_state is None:
            return
        new_state, new_sf = build_state_v2(
            self.data, sim_state, self.params, self._period_action,
            period_T=self.agent.config.period_length,
        )
        new_feas = compute_feasibility_mask_v2(
            self.data, sim_state, self.params, self._period_roaster_id,
            period_T=self.agent.config.period_length,
        )
        current_cost = self._kpi_total_cost(fallback=float(tardiness_cost))
        current_revenue = self._kpi_revenue()
        reward = self._compute_reward(current_cost, current_revenue)
        self.reward_history.append(reward)

        self.agent.store_transition(
            self._period_state,
            self._period_sf,
            self._period_action if self._period_action is not None else 0,
            float(reward),
            True,                  # terminal
            new_state,
            new_sf,
            new_feas,
        )
        if len(self.agent.replay) >= self.agent.config.batch_size:
            self.agent.train_step()

    # ------------------------------------------------------------------
    # Per-decision API
    # ------------------------------------------------------------------

    def decide(self, state, roaster_id: str) -> tuple:
        period_T = self.agent.config.period_length
        current_time = int(state.t)

        # Lazy init at first call of episode
        if self._period_state is None:
            self._period_state, self._period_sf = build_state_v2(
                self.data, state, self.params, None, period_T=period_T
            )
            self._period_feas = compute_feasibility_mask_v2(
                self.data, state, self.params, roaster_id, period_T=period_T
            )
            self._period_action = self.agent.select_action(
                self._period_state,
                self._period_sf,
                self._period_feas,
                training=self.training,
            )
            self._period_roaster_id = roaster_id
            self._period_boundary = current_time + period_T
            self._prev_cost = self._kpi_total_cost()
            self._prev_revenue = self._kpi_revenue()
            self._last_action_id = self._period_action
            self.action_counts[self._period_action] += 1

        # Period boundary crossed → store transition + train + new action
        elif current_time >= self._period_boundary:
            new_state, new_sf = build_state_v2(
                self.data, state, self.params, self._period_action, period_T=period_T
            )
            new_feas = compute_feasibility_mask_v2(
                self.data, state, self.params, roaster_id, period_T=period_T
            )
            current_cost = self._kpi_total_cost()
            current_revenue = self._kpi_revenue()
            reward = self._compute_reward(current_cost, current_revenue)
            self.reward_history.append(reward)

            if self.training:
                self.agent.store_transition(
                    self._period_state,
                    self._period_sf,
                    self._period_action if self._period_action is not None else 0,
                    float(reward),
                    False,                # not terminal
                    new_state,
                    new_sf,
                    new_feas,
                )
                if len(self.agent.replay) >= self.agent.config.batch_size:
                    self.agent.train_step()

            new_action = self.agent.select_action(
                new_state, new_sf, new_feas, training=self.training
            )
            self._period_state = new_state
            self._period_sf = new_sf
            self._period_feas = new_feas
            self._period_action = new_action
            self._period_roaster_id = roaster_id
            self._period_boundary = current_time + period_T
            self._prev_cost = current_cost
            self._prev_revenue = current_revenue
            self._last_action_id = new_action
            self.action_counts[new_action] += 1

        # Convert held period action to engine tuple for THIS roaster
        return self._action_to_env_tuple(self._period_action, roaster_id, state)

    def decide_restock(self, state) -> tuple:
        """Delegate restock to DispatchingHeuristic (paper has no restock layer)."""
        try:
            from dispatch.dispatching_heuristic import DispatchingHeuristic
            dispatcher = DispatchingHeuristic(self.params)
            return dispatcher.decide_restock(state)
        except Exception:
            return ("WAIT",)

    # ------------------------------------------------------------------
    # Action → engine tuple (Algorithm 1 with greedy fallback)
    # ------------------------------------------------------------------

    def _action_to_env_tuple(
        self,
        action_id: int | None,
        roaster_id: str,
        state,
    ) -> tuple:
        """Faithful Paeng Algorithm 1 dispatch:
            action a = from_setup * F + to_dispatch
            For each roaster R:
              - If R.last_sku == SKUS[from_setup] AND R can produce SKUS[to_dispatch]:
                  dispatch SKUS[to_dispatch]   (forces the (from, to) transition)
              - Else: SSU greedy fallback — stay on last_sku if it has demand, else
                  any eligible family with demand.
        """
        if action_id is None:
            return ("WAIT",)
        from_setup = int(action_id) // 3
        to_dispatch = int(action_id) % 3
        sku_from = SKUS[from_setup]
        sku_to = SKUS[to_dispatch]
        period_T = self.agent.config.period_length

        # Action match: roaster's current setup is sku_from AND it can produce sku_to
        if (state.last_sku.get(roaster_id) == sku_from
                and _roaster_can_produce(roaster_id, sku_to)):
            return self._dispatch_tuple(sku_to, roaster_id, state)

        # SSU greedy fallback for non-matching roasters
        last = state.last_sku.get(roaster_id)
        # Prefer no-setup continuation (last_sku) if it has waiting demand
        if (last
                and _roaster_can_produce(roaster_id, last)
                and _has_waiting_demand(last, state, self.params, period_T)):
            return self._dispatch_tuple(last, roaster_id, state)
        # Else any eligible family with demand
        for sku in SKUS:
            if (_roaster_can_produce(roaster_id, sku)
                    and _has_waiting_demand(sku, state, self.params, period_T)):
                return self._dispatch_tuple(sku, roaster_id, state)
        return ("WAIT",)

    def _dispatch_tuple(self, sku: str, roaster_id: str, state) -> tuple:
        if sku == "PSC":
            line = PSC_OUTPUT_LINE.get(roaster_id)
            if line is None:
                # R3: pick line with greater (rc_space, gc_psc) headroom
                max_rc = float(self.params.get("max_rc", 40))
                best_line, best_score = "L1", -1.0
                for ln in ("L1", "L2"):
                    rc_space = max_rc - state.rc_stock.get(ln, 0)
                    gc_psc = state.gc_stock.get((ln, "PSC"), 0)
                    score = min(rc_space, float(gc_psc))
                    if score > best_score:
                        best_line, best_score = ln, score
                line = best_line
            return ("PSC", line)
        return (sku,)

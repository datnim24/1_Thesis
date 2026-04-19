"""Five deterministic heuristic tools for the RL-HH meta-agent.

Each tool inspects the simulation state and returns an env action_id (0-20)
or None if no feasible action exists.  Feasibility is derived from the same
engine mask logic used by the Gymnasium wrapper, guaranteeing consistency.

Tool 0 - PSC_THROUGHPUT : start PSC (R3 balances RC across lines)
Tool 1 - GC_RESTOCK     : restock the most critical GC silo
Tool 2 - MTO_DEADLINE   : start MTO batch with highest remaining count
Tool 3 - SETUP_AVOID    : continue same SKU to skip setup cost
Tool 4 - WAIT           : always returns action 20
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PPOmask.Engine.action_spec import (
    ACTION_BY_ID,
    RESTOCK_ACTION_IDS,
    ROASTER_ACTION_IDS,
    WAIT_ACTION_ID,
)

# ---------------------------------------------------------------------------
# Roaster -> PSC action_id mapping
# ---------------------------------------------------------------------------
_PSC_ACTIONS: dict[str, list[int]] = {
    "R1": [0],
    "R2": [1],
    "R3": [2, 3],   # 2 -> L1, 3 -> L2
    "R4": [4],
    "R5": [5],
}

# Roaster -> {sku: action_id} for MTO actions
_MTO_ACTIONS: dict[str, list[tuple[int, str]]] = {
    "R1": [(6, "NDG")],
    "R2": [(7, "NDG"), (8, "BUSTA")],
}


class ToolKit:
    """Holds engine/params references and exposes the five tool functions.

    Usage::

        tk = ToolKit(engine, params)
        outputs, mask = tk.compute_all(state, roaster_id)
        # outputs: list[int | None]  (length 5)
        # mask:    list[bool]         (length 5)
    """

    def __init__(self, engine, params: dict):
        self.engine = engine
        self.params = params

    # ------------------------------------------------------------------
    # Public entry point — runs all tools with a single mask computation
    # ------------------------------------------------------------------

    def compute_all(
        self, state, roaster_id: str | None,
    ) -> tuple[list[int | None], list[bool]]:
        """Run all 5 tools, return (tool_outputs, tool_mask)."""
        feasible = self._feasible_actions(state, roaster_id)
        outputs: list[int | None] = [
            self._psc_throughput(state, roaster_id, feasible),
            self._gc_restock(state, roaster_id, feasible),
            self._mto_deadline(state, roaster_id, feasible),
            self._setup_avoid(state, roaster_id, feasible),
            WAIT_ACTION_ID,  # Tool 4: WAIT — always feasible
        ]
        mask = [o is not None for o in outputs]
        mask[-1] = True  # WAIT always valid
        return outputs, mask

    # ------------------------------------------------------------------
    # Internal: single mask computation shared across tools
    # ------------------------------------------------------------------

    def _feasible_actions(self, state, roaster_id: str | None) -> set[int]:
        """Return set of feasible action_ids for the current decision context."""
        feasible: set[int] = set()
        if roaster_id is not None:
            env_mask = self.engine._compute_action_mask(state, roaster_id)
            for aid in ROASTER_ACTION_IDS.get(roaster_id, ()):
                if env_mask.get(ACTION_BY_ID[aid].env_action, False):
                    feasible.add(aid)
        else:
            for aid in RESTOCK_ACTION_IDS:
                ad = ACTION_BY_ID[aid]
                if self.engine.can_start_restock(state, ad.line_id, ad.sku):
                    feasible.add(aid)
        return feasible

    # ------------------------------------------------------------------
    # Tool 0 — PSC_THROUGHPUT
    # ------------------------------------------------------------------

    def _psc_throughput(
        self, state, roaster_id: str | None, feasible: set[int],
    ) -> int | None:
        """Start PSC immediately.  R3 routes to the line with lower RC."""
        if roaster_id is None:
            return None
        candidates = [a for a in _PSC_ACTIONS.get(roaster_id, []) if a in feasible]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # R3: both L1 (action 2) and L2 (action 3) feasible — balance RC
        if state.rc_stock.get("L1", 0) <= state.rc_stock.get("L2", 0):
            return 2  # route to L1 (lower stock)
        return 3      # route to L2

    # ------------------------------------------------------------------
    # Tool 1 — GC_RESTOCK
    # ------------------------------------------------------------------

    def _gc_restock(
        self, state, roaster_id: str | None, feasible: set[int],
    ) -> int | None:
        """Restock the GC silo with the lowest stock/capacity ratio."""
        if roaster_id is not None:
            return None  # only valid in the restock layer
        best_action: int | None = None
        best_ratio = float("inf")
        for aid in RESTOCK_ACTION_IDS:
            if aid not in feasible:
                continue
            ad = ACTION_BY_ID[aid]
            pair = (ad.line_id, ad.sku)
            cap = self.params["gc_capacity"].get(pair, 1)
            stock = state.gc_stock.get(pair, 0)
            ratio = stock / max(1, cap)
            if ratio < best_ratio:
                best_ratio = ratio
                best_action = aid
        return best_action

    # ------------------------------------------------------------------
    # Tool 2 — MTO_DEADLINE
    # ------------------------------------------------------------------

    def _mto_deadline(
        self, state, roaster_id: str | None, feasible: set[int],
    ) -> int | None:
        """Start the MTO SKU with the most remaining batches.
        Tie-break: BUSTA > NDG (more constrained).
        """
        if roaster_id is None:
            return None
        options = _MTO_ACTIONS.get(roaster_id)
        if not options:
            return None

        sku_priority = {"BUSTA": 1, "NDG": 0}
        candidates: list[tuple[int, int, int]] = []  # (remaining, priority, aid)
        for aid, sku in options:
            if aid not in feasible:
                continue
            remaining = sum(
                state.mto_remaining.get(jid, 0)
                for jid, jsku in self.params["job_sku"].items()
                if jsku == sku
            )
            if remaining <= 0:
                continue
            candidates.append((remaining, sku_priority.get(sku, 0), aid))

        if not candidates:
            return None
        candidates.sort(reverse=True)  # most remaining first, then BUSTA > NDG
        return candidates[0][2]

    # ------------------------------------------------------------------
    # Tool 3 — SETUP_AVOID
    # ------------------------------------------------------------------

    def _setup_avoid(
        self, state, roaster_id: str | None, feasible: set[int],
    ) -> int | None:
        """Continue same SKU as last batch to avoid $800 setup + 5 min."""
        if roaster_id is None:
            return None

        last = state.last_sku.get(roaster_id, "PSC")

        # PSC or unknown → delegate to PSC_THROUGHPUT (PSC→PSC = no setup)
        if last in ("PSC", None):
            return self._psc_throughput(state, roaster_id, feasible)

        # last is NDG or BUSTA — try to continue same MTO SKU
        sku_actions = dict(_MTO_ACTIONS.get(roaster_id, []))
        if last not in sku_actions:
            return None

        aid = sku_actions[last]
        if aid not in feasible:
            return None

        # Check remaining batches for this SKU
        remaining = sum(
            state.mto_remaining.get(jid, 0)
            for jid, jsku in self.params["job_sku"].items()
            if jsku == last
        )
        return aid if remaining > 0 else None

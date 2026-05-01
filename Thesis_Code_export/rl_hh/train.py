"""RL Hyper-Heuristic (Dueling DDQN over a hyper-heuristic action space).

This single file consolidates the canonical RL-HH training pipeline from
``test_rl_hh/`` into one module: configs (as namespace ``C``), action toolkit,
masking, replay buffer, dueling DDQN network (PyTorch + numpy inference), agent,
fast-loop episode runner, env-bridge strategy, cycle utilities, and the
training entry points (single-shot and cycle-based).

Outputs land in ``Results/<YYYYMMDD_HHMMSS>_RLHH_<RunName>/`` via
``evaluation.result_schema.make_run_dir``.

Usage:
    python -m rl_hh.train --name SmokeTest --max-steps 200 --seed 42
    python -m rl_hh.train cycle --name CycleSmoke --total-steps 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Project root on sys.path so env.* + evaluation.* import cleanly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.action_spec import (
    ACTION_BY_ID,
    RESTOCK_ACTION_IDS,
    ROASTER_ACTION_IDS,
    WAIT_ACTION_ID,
)
from env.observation_spec import (
    GC_FEATURE_ORDER,
    ObservationContext,
    PIPELINE_MODE_ENCODING,
    ROASTER_ORDER,
    SKU_ENCODING,
    STATUS_ENCODING,
    build_observation,
)
from env.data_loader import load_data
from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events

# ============================================================================
# === configs.py (inlined as namespace `C`)
# ============================================================================
class _ConfigsNS:


    # ---------------------------------------------------------------------------
    # Network architecture
    # ---------------------------------------------------------------------------
    INPUT_DIM = 33
    N_TOOLS = 5

    # Grouped feature slices (indices into the 33-dim observation)
    ROASTER_FEATURES = slice(1, 16)       # 15: 5 status + 5 timer + 5 last_sku
    INVENTORY_FEATURES = slice(16, 27)    # 11: RC, mto, pipeline, GC, restock
    CONTEXT_FEATURES_IDX = [0] + list(range(27, 33))  # time + 6 one-hot = 7

    ROASTER_DIM = 15
    INVENTORY_DIM = 11
    CONTEXT_DIM = 7

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    LR = 5e-4
    GAMMA = 0.99
    BATCH_SIZE = 128
    BUFFER_SIZE = 50_000
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_FRAC = 0.30
    TAU = 0.005            # soft target update rate
    NUM_EPISODES = 300_000
    TRAIN_EVERY = 4        # train once every N decisions (inside fast_loop)
    TRAINS_PER_EP = 4      # gradient steps per episode (keep low for speed)
    GRAD_CLIP = 10.0       # max gradient norm

    # ---------------------------------------------------------------------------
    # UPS — always read from Input_data/shift_parameters.csv at runtime.
    # These are ONLY used as fallbacks if data loading somehow fails.
    # ---------------------------------------------------------------------------
    UPS_LAMBDA_FALLBACK = 3
    UPS_MU_FALLBACK = 20

    # ---------------------------------------------------------------------------
    # Logging / checkpoints
    # ---------------------------------------------------------------------------
    LOG_INTERVAL = 1_000
    CHECKPOINT_INTERVAL = 50_000

    # ---------------------------------------------------------------------------
    # Tool names (for logging / display)
    # ---------------------------------------------------------------------------
    TOOL_NAMES = ["PSC_THROUGHPUT", "GC_RESTOCK", "MTO_DEADLINE", "SETUP_AVOID", "WAIT"]



C = _ConfigsNS


# ============================================================================
# === tools.py
# ============================================================================





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
        # R3: both L1 (action 2) and L2 (action 3) feasible.
        # Cycle 5: Route to the line with more headroom in the TIGHTEST dimension
        # (RC buffer space vs upstream GC supply). This avoids sending R3 to a
        # line whose GC_PSC is nearly depleted, which would block future R3/R4/R5
        # batches there once GC hits 0.
        max_rc = self.params.get("max_rc", 40)
        rc_l1 = state.rc_stock.get("L1", 0)
        rc_l2 = state.rc_stock.get("L2", 0)
        gc_l1 = state.gc_stock.get(("L1", "PSC"), 0)
        gc_l2 = state.gc_stock.get(("L2", "PSC"), 0)
        score_l1 = min(max_rc - rc_l1, gc_l1)
        score_l2 = min(max_rc - rc_l2, gc_l2)
        # Cycle 14: UPS-aware — if R4 or R5 is DOWN, L2 is understaffed; bias R3
        # toward L2 to maintain throughput on the weaker line.
        l2_down = (
            state.status.get("R4") == "DOWN"
            or state.status.get("R5") == "DOWN"
        )
        l1_down = state.status.get("R1") == "DOWN" or state.status.get("R2") == "DOWN"
        if l2_down and not l1_down:
            return 3  # force L2 support
        if l1_down and not l2_down:
            return 2  # force L1 support
        if score_l1 > score_l2:
            return 2  # route to L1
        return 3      # route to L2 (tie goes here per cycle 10)

    # ------------------------------------------------------------------
    # Tool 1 — GC_RESTOCK
    # ------------------------------------------------------------------

    def _gc_restock(
        self, state, roaster_id: str | None, feasible: set[int],
    ) -> int | None:
        """Restock only when a silo is genuinely running low.

        Urgency rule (cycle 1): a silo qualifies if stock/cap < 0.5 AND
        stock <= 12 absolute (PSC silos capacity=40 -> safe above 20;
        NDG/BUSTA cap=10 -> safe above 5). Tool masked (returns None)
        if no silo qualifies, freeing the agent to pick WAIT instead
        of triggering wasteful restocks.
        """
        if roaster_id is not None:
            return None  # only valid in the restock layer
        # Cycle 16: count idle-waiting roasters per line — boosts urgency
        # when more roasters are stalled waiting for this line's supply.
        idle_per_line = {
            "L1": sum(1 for r in ("R1", "R2") if state.status.get(r) == "IDLE"),
            "L2": sum(1 for r in ("R4", "R5") if state.status.get(r) == "IDLE"),
        }
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
            # Boost priority by subtracting small amount per idle roaster
            ratio -= 0.05 * idle_per_line.get(ad.line_id, 0)
            if cap >= 20:
                if ratio >= 0.35 or stock > 6:
                    continue
            else:
                if ratio >= 0.4 or stock > 3:
                    continue
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
        """Continue same SKU as last batch to avoid $800 setup + 5 min.

        Cycle 4: On R1/R2 when last=PSC and MTO still remaining, interpret
        'avoid setup' as 'start MTO now so we only do ONE PSC->MTO transition
        for the entire shift' instead of delegating to PSC (which creates a
        future PSC->MTO + MTO->PSC double setup). Net: 1 setup per roaster
        instead of 2. Exploits the agent's learned high Q-value for this tool.
        """
        if roaster_id is None:
            return None

        last = state.last_sku.get(roaster_id, "PSC")

        # PSC or unknown -> delegate to MTO (R1/R2 with MTO pending) or PSC
        if last in ("PSC", None):
            if roaster_id in ("R1", "R2"):
                mto = self._mto_deadline(state, roaster_id, feasible)
                if mto is not None:
                    return mto
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



# ============================================================================
# === tool_mask.py
# ============================================================================




def compute_tool_mask(
    toolkit: ToolKit, state, roaster_id: str | None,
) -> tuple[list[int | None], list[bool]]:
    """Compute tool outputs and feasibility mask in a single pass.

    Returns
    -------
    tool_outputs : list[int | None]
        Action_id each tool would produce (None = infeasible).
    tool_mask : list[bool]
        True where the corresponding tool returned a valid action.
        Last entry (WAIT) is always True.
    """
    return toolkit.compute_all(state, roaster_id)



# ============================================================================
# === replay_buffer.py
# ============================================================================





class ReplayBuffer:
    def __init__(
        self,
        capacity: int = C.BUFFER_SIZE,
        state_dim: int = C.INPUT_DIM,
        n_tools: int = C.N_TOOLS,
    ):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.tool_masks = np.zeros((capacity, n_tools), dtype=bool)
        self.next_tool_masks = np.zeros((capacity, n_tools), dtype=bool)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def store(
        self,
        state: np.ndarray,
        tool_id: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        tool_mask: list[bool],
        next_tool_mask: list[bool],
    ) -> None:
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = tool_id
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.tool_masks[idx] = tool_mask
        self.next_tool_masks[idx] = next_tool_mask
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int = C.BATCH_SIZE,
    ) -> tuple[np.ndarray, ...]:
        """Uniform random batch.  Returns numpy arrays."""
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.tool_masks[indices],
            self.next_tool_masks[indices],
        )



# ============================================================================
# === network.py
# ============================================================================





class DuelingDDQN(nn.Module):
    def __init__(
        self,
        n_tools: int = C.N_TOOLS,
        roaster_dim: int = C.ROASTER_DIM,
        inventory_dim: int = C.INVENTORY_DIM,
        context_dim: int = C.CONTEXT_DIM,
    ):
        super().__init__()

        # Group A — Roaster block
        self.roaster_net = nn.Sequential(
            nn.Linear(roaster_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Group B — Inventory block
        self.inventory_net = nn.Sequential(
            nn.Linear(inventory_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Group C — Context + time block
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
        )

        # Merge: cat(32, 32, 16) = 80
        self.merge = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
        )

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_tools),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  x: (batch, 33) -> Q: (batch, n_tools)."""
        roaster_in = x[:, C.ROASTER_FEATURES]          # (B, 15)
        inventory_in = x[:, C.INVENTORY_FEATURES]       # (B, 11)
        context_in = x[:, C.CONTEXT_FEATURES_IDX]       # (B, 7)

        r = self.roaster_net(roaster_in)
        i = self.inventory_net(inventory_in)
        c = self.context_net(context_in)

        merged = self.merge(torch.cat([r, i, c], dim=1))

        v = self.value_stream(merged)          # (B, 1)
        a = self.advantage_stream(merged)      # (B, n_tools)

        # Q = V + (A - mean(A))
        q = v + a - a.mean(dim=1, keepdim=True)
        return q



# ============================================================================
# === numpy_net.py
# ============================================================================




def _extract_linear(mod: nn.Linear) -> tuple[np.ndarray, np.ndarray]:
    """Extract weight and bias from a Linear layer as numpy arrays."""
    w = mod.weight.detach().cpu().numpy().copy()
    b = mod.bias.detach().cpu().numpy().copy()
    return w, b


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, out=x)


class NumpyDuelingDDQN:
    """Pure-numpy forward pass for the Dueling DDQN."""

    __slots__ = (
        "r_w1", "r_b1", "r_w2", "r_b2",
        "i_w1", "i_b1", "i_w2", "i_b2",
        "c_w1", "c_b1",
        "m_w1", "m_b1",
        "v_w1", "v_b1", "v_w2", "v_b2",
        "a_w1", "a_b1", "a_w2", "a_b2",
        "_ctx_idx", "_bufs",
    )

    def __init__(self, torch_net):
        # Group A — Roaster
        self.r_w1, self.r_b1 = _extract_linear(torch_net.roaster_net[0])
        self.r_w2, self.r_b2 = _extract_linear(torch_net.roaster_net[2])
        # Group B — Inventory
        self.i_w1, self.i_b1 = _extract_linear(torch_net.inventory_net[0])
        self.i_w2, self.i_b2 = _extract_linear(torch_net.inventory_net[2])
        # Group C — Context
        self.c_w1, self.c_b1 = _extract_linear(torch_net.context_net[0])
        # Merge
        self.m_w1, self.m_b1 = _extract_linear(torch_net.merge[0])
        # Value stream
        self.v_w1, self.v_b1 = _extract_linear(torch_net.value_stream[0])
        self.v_w2, self.v_b2 = _extract_linear(torch_net.value_stream[2])
        # Advantage stream
        self.a_w1, self.a_b1 = _extract_linear(torch_net.advantage_stream[0])
        self.a_w2, self.a_b2 = _extract_linear(torch_net.advantage_stream[2])

        # Context feature indices: [0, 27, 28, 29, 30, 31, 32]
        self._ctx_idx = np.array(C.CONTEXT_FEATURES_IDX, dtype=np.intp)

        # Pre-allocated buffers for forward_single (zero-alloc inference)
        self._bufs = {
            "r1": np.empty(64, dtype=np.float32),
            "r2": np.empty(32, dtype=np.float32),
            "i1": np.empty(64, dtype=np.float32),
            "i2": np.empty(32, dtype=np.float32),
            "c1": np.empty(16, dtype=np.float32),
            "cat": np.empty(80, dtype=np.float32),
            "m1": np.empty(128, dtype=np.float32),
            "vh": np.empty(64, dtype=np.float32),
            "v": np.empty(1, dtype=np.float32),
            "ah": np.empty(64, dtype=np.float32),
            "a": np.empty(5, dtype=np.float32),
            "q": np.empty(5, dtype=np.float32),
        }

    def sync(self, torch_net) -> None:
        """Re-extract weights after training step."""
        self.r_w1, self.r_b1 = _extract_linear(torch_net.roaster_net[0])
        self.r_w2, self.r_b2 = _extract_linear(torch_net.roaster_net[2])
        self.i_w1, self.i_b1 = _extract_linear(torch_net.inventory_net[0])
        self.i_w2, self.i_b2 = _extract_linear(torch_net.inventory_net[2])
        self.c_w1, self.c_b1 = _extract_linear(torch_net.context_net[0])
        self.m_w1, self.m_b1 = _extract_linear(torch_net.merge[0])
        self.v_w1, self.v_b1 = _extract_linear(torch_net.value_stream[0])
        self.v_w2, self.v_b2 = _extract_linear(torch_net.value_stream[2])
        self.a_w1, self.a_b1 = _extract_linear(torch_net.advantage_stream[0])
        self.a_w2, self.a_b2 = _extract_linear(torch_net.advantage_stream[2])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, 33) -> Q: (N, 5).  All float32."""
        # Group A: x[:, 1:16] -> (N, 32)
        r = x[:, 1:16] @ self.r_w1.T + self.r_b1
        _relu(r)
        r = r @ self.r_w2.T + self.r_b2
        _relu(r)

        # Group B: x[:, 16:27] -> (N, 32)
        inv = x[:, 16:27] @ self.i_w1.T + self.i_b1
        _relu(inv)
        inv = inv @ self.i_w2.T + self.i_b2
        _relu(inv)

        # Group C: x[:, ctx_idx] -> (N, 16)
        ctx = x[:, self._ctx_idx] @ self.c_w1.T + self.c_b1
        _relu(ctx)

        # Merge
        merged = np.concatenate([r, inv, ctx], axis=1) @ self.m_w1.T + self.m_b1
        _relu(merged)

        # Value
        v = merged @ self.v_w1.T + self.v_b1
        _relu(v)
        v = v @ self.v_w2.T + self.v_b2  # (N, 1)

        # Advantage
        a = merged @ self.a_w1.T + self.a_b1
        _relu(a)
        a = a @ self.a_w2.T + self.a_b2  # (N, 5)

        # Q = V + A - mean(A)
        return v + a - a.mean(axis=1, keepdims=True)

    def forward_single(self, x: np.ndarray) -> np.ndarray:
        """x: (33,) -> Q: (5,).  Pre-allocated buffers for zero-alloc inference."""
        b = self._bufs
        # Group A
        np.dot(self.r_w1, x[1:16], out=b["r1"])
        b["r1"] += self.r_b1
        np.maximum(b["r1"], 0.0, out=b["r1"])
        np.dot(self.r_w2, b["r1"], out=b["r2"])
        b["r2"] += self.r_b2
        np.maximum(b["r2"], 0.0, out=b["r2"])

        # Group B
        np.dot(self.i_w1, x[16:27], out=b["i1"])
        b["i1"] += self.i_b1
        np.maximum(b["i1"], 0.0, out=b["i1"])
        np.dot(self.i_w2, b["i1"], out=b["i2"])
        b["i2"] += self.i_b2
        np.maximum(b["i2"], 0.0, out=b["i2"])

        # Group C
        np.dot(self.c_w1, x[self._ctx_idx], out=b["c1"])
        b["c1"] += self.c_b1
        np.maximum(b["c1"], 0.0, out=b["c1"])

        # Merge: cat(32, 32, 16) = 80 -> 128
        cat = b["cat"]
        cat[:32] = b["r2"]
        cat[32:64] = b["i2"]
        cat[64:] = b["c1"]
        np.dot(self.m_w1, cat, out=b["m1"])
        b["m1"] += self.m_b1
        np.maximum(b["m1"], 0.0, out=b["m1"])

        # V stream
        np.dot(self.v_w1, b["m1"], out=b["vh"])
        b["vh"] += self.v_b1
        np.maximum(b["vh"], 0.0, out=b["vh"])
        np.dot(self.v_w2, b["vh"], out=b["v"])
        b["v"] += self.v_b2

        # A stream
        np.dot(self.a_w1, b["m1"], out=b["ah"])
        b["ah"] += self.a_b1
        np.maximum(b["ah"], 0.0, out=b["ah"])
        np.dot(self.a_w2, b["ah"], out=b["a"])
        b["a"] += self.a_b2

        # Q = V + A - mean(A)
        q = b["q"]
        q[:] = b["v"][0] + b["a"] - b["a"].mean()
        return q



# ============================================================================
# === meta_agent.py
# ============================================================================






class DuelingDDQNAgent:
    def __init__(
        self,
        lr: float = C.LR,
        gamma: float = C.GAMMA,
        buffer_size: int = C.BUFFER_SIZE,
        batch_size: int = C.BATCH_SIZE,
        epsilon_start: float = C.EPSILON_START,
        epsilon_end: float = C.EPSILON_END,
        epsilon_decay_frac: float = C.EPSILON_DECAY_FRAC,
        tau: float = C.TAU,
        grad_clip: float = C.GRAD_CLIP,
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.grad_clip = grad_clip

        self.online_net = DuelingDDQN()
        self.target_net = DuelingDDQN()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.epsilon = epsilon_start
        self._eps_start = epsilon_start
        self._eps_end = epsilon_end
        self._eps_decay_frac = epsilon_decay_frac

        self.step_count = 0
        self._last_loss: float = 0.0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_tool(
        self,
        state_vec: np.ndarray,
        tool_mask: list[bool],
        training: bool = True,
    ) -> int:
        """Epsilon-greedy selection over masked Q-values."""
        valid_tools = [i for i, m in enumerate(tool_mask) if m]
        if not valid_tools:
            return C.N_TOOLS - 1  # WAIT fallback

        if training and random.random() < self.epsilon:
            return random.choice(valid_tools)

        with torch.no_grad():
            x = torch.as_tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            q_values = self.online_net(x).squeeze(0)

        mask_tensor = torch.tensor(tool_mask, dtype=torch.bool)
        q_values[~mask_tensor] = -1e9
        return int(q_values.argmax().item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> float:
        """One gradient step using DDQN target with masked action selection."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        (states, actions, rewards, next_states,
         dones, masks, next_masks) = self.replay_buffer.sample(self.batch_size)

        states_t = torch.as_tensor(states)
        actions_t = torch.as_tensor(actions, dtype=torch.long)
        rewards_t = torch.as_tensor(rewards)
        next_states_t = torch.as_tensor(next_states)
        dones_t = torch.as_tensor(dones)
        next_masks_t = torch.as_tensor(next_masks, dtype=torch.bool)

        # Current Q-values for chosen tools
        q_current = self.online_net(states_t).gather(
            1, actions_t.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            # DDQN: online net selects, target net evaluates
            q_online_next = self.online_net(next_states_t)
            q_online_next[~next_masks_t] = -1e9
            best_tools = q_online_next.argmax(dim=1)

            q_target_next = self.target_net(next_states_t)
            q_next_val = q_target_next.gather(
                1, best_tools.unsqueeze(1),
            ).squeeze(1)

            targets = rewards_t + self.gamma * q_next_val * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_current, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft update target network
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

        self.step_count += 1
        self._last_loss = float(loss.item())
        return self._last_loss

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    def decay_epsilon(self, episode: int, total_episodes: int) -> None:
        """Linear decay from eps_start to eps_end over decay_frac of training."""
        decay_episodes = int(total_episodes * self._eps_decay_frac)
        if episode >= decay_episodes:
            self.epsilon = self._eps_end
        else:
            frac = episode / max(1, decay_episodes)
            self.epsilon = self._eps_start + frac * (self._eps_end - self._eps_start)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = ckpt.get("epsilon", self._eps_end)
        self.step_count = ckpt.get("step_count", 0)



# ============================================================================
# === fast_loop.py
# ============================================================================







# Pre-computed context one-hot vectors
_CONTEXT_VECS: dict[str, np.ndarray] = {}
for _i, _key in enumerate(("RESTOCK", "R1", "R2", "R3", "R4", "R5")):
    _v = np.zeros(6, dtype=np.float32)
    _v[_i] = 1.0
    _v.flags.writeable = False
    _CONTEXT_VECS[_key] = _v


def _build_base_obs_fast(data, state, buf: np.ndarray) -> None:
    """Write 27-dim base observation into buf[0:27]."""
    buf[0] = float(state.t) / float(max(1, data.shift_length))

    roaster_timer_norm = max(
        data.shift_length,
        max(data.roast_time_by_sku.values()),
        data.setup_time,
    )
    inv_timer_norm = 1.0 / float(roaster_timer_norm)

    status = state.status
    remaining = state.remaining
    last_sku = state.last_sku

    for offset, rid in enumerate(ROASTER_ORDER, start=1):
        buf[offset] = STATUS_ENCODING[status[rid]]
    for offset, rid in enumerate(ROASTER_ORDER, start=6):
        v = remaining[rid]
        buf[offset] = float(max(0, min(v, roaster_timer_norm))) * inv_timer_norm
    for offset, rid in enumerate(ROASTER_ORDER, start=11):
        buf[offset] = SKU_ENCODING[last_sku[rid]]

    inv_max_rc = 1.0 / float(data.max_rc) if data.max_rc > 0 else 0.0
    buf[16] = float(state.rc_stock["L1"]) * inv_max_rc
    buf[17] = float(state.rc_stock["L2"]) * inv_max_rc

    total_mto_initial = max(1, sum(data.job_batches.values()))
    buf[18] = float(sum(state.mto_remaining.values())) / float(total_mto_initial)

    buf[19] = PIPELINE_MODE_ENCODING[state.pipeline_mode["L1"]]
    buf[20] = PIPELINE_MODE_ENCODING[state.pipeline_mode["L2"]]

    gc_stock = state.gc_stock
    gc_cap = data.gc_capacity
    for index, pair in enumerate(GC_FEATURE_ORDER, start=21):
        buf[index] = float(gc_stock[pair]) / float(max(1, gc_cap[pair]))

    buf[25] = 1.0 if state.restock_busy > 0 else 0.0
    rst_dur = data.restock_duration
    rb = state.restock_busy
    buf[26] = float(max(0, min(rb, rst_dur))) / float(rst_dur) if rst_dur > 0 else 0.0


def run_fast_episode(
    engine,
    agent: DuelingDDQNAgent,
    toolkit: ToolKit,
    data,
    replay_buffer: ReplayBuffer,
    ups_events: list,
    *,
    training: bool = True,
    np_net: NumpyDuelingDDQN | None = None,
):
    """Run one episode.  Returns (kpi, n_decisions, tool_counts).

    Pass np_net for numpy-only inference (much faster on CPU).
    """

    state = engine._initialize_state()
    kpi = engine._make_kpi_tracker()

    ups_by_time: dict[int, list] = defaultdict(list)
    for ev in ups_events:
        ups_by_time[ev.t].append(ev)

    prev_obs: np.ndarray | None = None
    prev_tid: int = 0
    prev_mask: list[bool] = []
    prev_profit: float = 0.0
    n_decisions = 0
    tool_counts = [0] * C.N_TOOLS

    SL = engine.params["SL"]
    base_buf = np.empty(27, dtype=np.float32)
    epsilon = agent.epsilon

    for slot in range(SL):
        state.t = slot

        # ---- Phases 1-5 ----
        for ev in ups_by_time.get(slot, []):
            engine._process_ups(state, ev, None, kpi)
        engine._step_roaster_timers(state, kpi)
        engine._step_pipeline_and_restock_timers(state, kpi)
        engine._process_consumption_events(state, kpi)
        engine._track_stockout_duration(state, kpi)
        engine._accrue_idle_penalties(state, kpi)

        # ---- Collect decisions ----
        pending: list[tuple] = []

        if state.restock_busy == 0:
            to, tm = toolkit.compute_all(state, None)
            if any(tm[:-1]):
                pending.append(("RESTOCK", None, to, tm))

        for rid in engine.roasters:
            if state.status[rid] != "IDLE" or not state.needs_decision[rid]:
                continue
            to, tm = toolkit.compute_all(state, rid)
            if not any(tm[:-1]):
                engine._apply_action(state, rid, ("WAIT",), kpi)
                continue
            pending.append((rid, rid, to, tm))

        if not pending:
            continue

        # ---- Build observations ----
        _build_base_obs_fast(data, state, base_buf)
        n_pending = len(pending)
        obs_array = np.empty((n_pending, C.INPUT_DIM), dtype=np.float32)
        for i, (ctx_key, _, _, _) in enumerate(pending):
            obs_array[i, :27] = base_buf
            obs_array[i, 27:] = _CONTEXT_VECS[ctx_key]

        # ---- Select tools ----
        tool_ids: list[int] = [0] * n_pending

        greedy_indices: list[int] = []
        for i in range(n_pending):
            valid = [j for j, m in enumerate(pending[i][3]) if m]
            if not valid:
                tool_ids[i] = C.N_TOOLS - 1
            elif training and random.random() < epsilon:
                tool_ids[i] = random.choice(valid)
            else:
                greedy_indices.append(i)

        if greedy_indices:
            ng = len(greedy_indices)
            if np_net is not None:
                # Fast numpy inference
                if ng == 1:
                    idx = greedy_indices[0]
                    q = np_net.forward_single(obs_array[idx])
                    mask = pending[idx][3]
                    for j in range(C.N_TOOLS):
                        if not mask[j]:
                            q[j] = -1e9
                    tool_ids[idx] = int(np.argmax(q))
                else:
                    batch = np.stack([obs_array[i] for i in greedy_indices])
                    q_batch = np_net.forward(batch)
                    for k, idx in enumerate(greedy_indices):
                        q = q_batch[k]
                        mask = pending[idx][3]
                        for j in range(C.N_TOOLS):
                            if not mask[j]:
                                q[j] = -1e9
                        tool_ids[idx] = int(np.argmax(q))
            else:
                # PyTorch fallback
                batch_t = torch.as_tensor(
                    np.stack([obs_array[i] for i in greedy_indices]),
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    q_batch = agent.online_net(batch_t).numpy()
                for k, idx in enumerate(greedy_indices):
                    q = q_batch[k]
                    mask = pending[idx][3]
                    for j in range(C.N_TOOLS):
                        if not mask[j]:
                            q[j] = -1e9
                    tool_ids[idx] = int(np.argmax(q))

        # ---- Apply decisions ----
        cur_profit = float(kpi.net_profit())
        for i in range(n_pending):
            ctx_key, rid, tool_outputs, tool_mask = pending[i]
            obs = obs_array[i]
            tid = tool_ids[i]
            tool_counts[tid] += 1
            n_decisions += 1

            if training and prev_obs is not None:
                replay_buffer.store(
                    prev_obs, prev_tid, cur_profit - prev_profit,
                    obs, False, prev_mask, tool_mask,
                )

            aid = tool_outputs[tid]
            if aid is None:
                aid = WAIT_ACTION_ID

            if rid is None:
                ad = ACTION_BY_ID[aid]
                if ad.env_action[0] == "START_RESTOCK":
                    _, lid, sku = ad.env_action
                    if engine.can_start_restock(state, lid, sku):
                        engine._start_restock(state, lid, sku, kpi)
            else:
                engine._apply_action(state, rid, ACTION_BY_ID[aid].env_action, kpi)

            prev_obs = obs.copy()
            prev_tid = tid
            prev_mask = tool_mask
            prev_profit = cur_profit
            cur_profit = float(kpi.net_profit())

    # ---- End-of-shift ----
    engine._penalize_skipped_mto(state, kpi)

    if training and prev_obs is not None:
        terminal_obs = np.zeros(C.INPUT_DIM, dtype=np.float32)
        terminal_mask = [False] * (C.N_TOOLS - 1) + [True]
        reward = float(kpi.net_profit()) - prev_profit
        replay_buffer.store(
            prev_obs, prev_tid, reward,
            terminal_obs, True, prev_mask, terminal_mask,
        )

    return kpi, n_decisions, tool_counts



# ============================================================================
# === rl_hh_strategy.py
# ============================================================================








class RLHHStrategy:
    """Plugs into SimulationEngine via decide(state, roaster) -> env_action tuple."""

    def __init__(
        self,
        agent: DuelingDDQNAgent,
        data,
        training: bool = False,
    ):
        self.agent = agent
        self.data = data
        self.params = data.to_env_params() if hasattr(data, "to_env_params") else data
        self._engine: SimulationEngine | None = None
        self._toolkit: ToolKit | None = None
        self.training = training

        # Tool selection stats (for evaluation diagnostics)
        self.tool_counts: dict[int, int] = {i: 0 for i in range(5)}

    def _get_engine(self) -> SimulationEngine:
        if self._engine is None:
            params = self.data.to_env_params() if hasattr(self.data, "to_env_params") else self.data
            self._engine = SimulationEngine(params)
        return self._engine

    def _get_toolkit(self) -> ToolKit:
        if self._toolkit is None:
            params = self.data.to_env_params() if hasattr(self.data, "to_env_params") else self.data
            self._toolkit = ToolKit(self._get_engine(), params)
        return self._toolkit

    # ------------------------------------------------------------------
    # SimulationEngine strategy interface
    # ------------------------------------------------------------------

    def decide(self, state, roaster_id: str) -> tuple:
        """Per-roaster decision: select tool via DDQN, return env_action tuple."""
        context = ObservationContext(kind="ROASTER", roaster_id=roaster_id)
        obs = build_observation(self.data, state, context)

        toolkit = self._get_toolkit()
        tool_outputs, tool_mask = toolkit.compute_all(state, roaster_id)

        tool_id = self.agent.select_tool(obs, tool_mask, training=self.training)
        self.tool_counts[tool_id] = self.tool_counts.get(tool_id, 0) + 1

        action_id = tool_outputs[tool_id]
        if action_id is None:
            action_id = WAIT_ACTION_ID
        return ACTION_BY_ID[action_id].env_action

    def decide_restock(self, state) -> tuple:
        """Global restock decision: select tool via DDQN, return env_action tuple."""
        context = ObservationContext(kind="RESTOCK")
        obs = build_observation(self.data, state, context)

        toolkit = self._get_toolkit()
        tool_outputs, tool_mask = toolkit.compute_all(state, None)

        tool_id = self.agent.select_tool(obs, tool_mask, training=self.training)
        self.tool_counts[tool_id] = self.tool_counts.get(tool_id, 0) + 1

        action_id = tool_outputs[tool_id]
        if action_id is None:
            return ("WAIT",)

        env_action = ACTION_BY_ID[action_id].env_action
        if env_action[0] == "START_RESTOCK":
            return env_action
        return ("WAIT",)



# ============================================================================
# === cycle_utils.py
# ============================================================================




def load_json(p: str | Path) -> dict:
    with open(p) as f:
        return json.load(f)


def fmt_delta(new: float, old: float, lower_is_better: bool = False) -> str:
    d = new - old
    sign = "+" if d >= 0 else ""
    arrow = ""
    if lower_is_better:
        if d < -1: arrow = " ✓"
        elif d > 1: arrow = " ✗"
    else:
        if d > 1: arrow = " ✓"
        elif d < -1: arrow = " ✗"
    return f"{sign}{d:,.1f}{arrow}"


def cycle_markdown_block(cycle: int, result: dict, baseline: dict, train_summary: dict, changes: str, hypothesis: str, diagnosis: str | None = None, action: str | None = None) -> str:
    """Produce a markdown block for one cycle."""
    p_mean = result["profit_mean"]
    p_std = result["profit_std"]
    p_med = result["profit_median"]
    p_min = result["profit_min"]
    p_max = result["profit_max"]
    vs_base = p_mean - baseline["profit_mean"]
    beat = "BEAT ✓" if p_mean > baseline["profit_mean"] else "LOSE ✗"

    tool_dist = result["tool_distribution"]
    tool_str = ", ".join(f"{k}={v*100:.1f}%" for k, v in tool_dist.items())

    lines = [
        f"## Cycle {cycle} — {hypothesis}",
        "",
        f"**Result vs baseline**: {beat} (mean=${p_mean:,.0f}, Δ=${vs_base:+,.0f} vs ${baseline['profit_mean']:,.0f})",
        "",
        "### Training",
        f"- Episodes: {train_summary['episodes']:,}",
        f"- Wall: {train_summary['wall_sec']:.0f}s ({train_summary['wall_sec']/60:.1f}min)",
        f"- Best training profit: ${train_summary['best_training_profit']:,.0f} (ep {train_summary['best_training_episode']})",
        f"- Final ε: {train_summary['final_epsilon']}",
        f"- Best ckpt: `{train_summary['best_ckpt']}`",
        "",
        "### Code Changes",
        changes,
        "",
        "### Eval (100 seeds, base_seed=900000, λ=5 μ=20)",
        "| Metric | Cycle {c} | Baseline | Δ |".format(c=cycle),
        "|--------|-----------|----------|---|",
        f"| Profit mean | ${p_mean:,.0f} | ${baseline['profit_mean']:,.0f} | **{fmt_delta(p_mean, baseline['profit_mean'])}** |",
        f"| Profit std | ${p_std:,.0f} | ${baseline['profit_std']:,.0f} | {fmt_delta(p_std, baseline['profit_std'], lower_is_better=True)} |",
        f"| Profit median | ${p_med:,.0f} | ${baseline['profit_median']:,.0f} | {fmt_delta(p_med, baseline['profit_median'])} |",
        f"| Profit min | ${p_min:,.0f} | ${baseline['profit_min']:,.0f} | {fmt_delta(p_min, baseline['profit_min'])} |",
        f"| Profit max | ${p_max:,.0f} | ${baseline['profit_max']:,.0f} | {fmt_delta(p_max, baseline['profit_max'])} |",
        f"| Mean idle min | {result['mean_idle_min']:.0f} | {baseline['mean_idle_min']:.0f} | {fmt_delta(result['mean_idle_min'], baseline['mean_idle_min'], lower_is_better=True)} |",
        f"| Mean setups | {result['mean_setup_events']:.1f} | {baseline['mean_setup_events']:.1f} | {fmt_delta(result['mean_setup_events'], baseline['mean_setup_events'], lower_is_better=True)} |",
        f"| Mean restocks | {result['mean_restock_count']:.1f} | {baseline['mean_restock_count']:.1f} | {fmt_delta(result['mean_restock_count'], baseline['mean_restock_count'], lower_is_better=True)} |",
        f"| Mean PSC | {result['mean_psc']:.1f} | {baseline['mean_psc']:.1f} | {fmt_delta(result['mean_psc'], baseline['mean_psc'])} |",
        f"| Mean tard cost | ${result['mean_tard_cost']:,.0f} | ${baseline['mean_tard_cost']:,.0f} | {fmt_delta(result['mean_tard_cost'], baseline['mean_tard_cost'], lower_is_better=True)} |",
        "",
        f"**Tool distribution**: {tool_str}",
        "",
    ]
    if diagnosis:
        lines.append("### Diagnosis")
        lines.append(diagnosis)
        lines.append("")
    if action:
        lines.append("### Next action (cycle %d)" % (cycle + 1))
        lines.append(action)
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)



# ============================================================================
# === train_rl_hh.py
# ============================================================================







def train(
    num_episodes: int = C.NUM_EPISODES,
    ups_lambda: float | None = None,
    ups_mu: float | None = None,
    output_dir: str | Path = "test_rl_hh/outputs",
    input_dir: str | Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    data = load_data(input_dir)
    params = data.to_env_params()
    if ups_lambda is None:
        ups_lambda = data.ups_lambda
    if ups_mu is None:
        ups_mu = data.ups_mu

    engine = SimulationEngine(params)
    toolkit = ToolKit(engine, params)
    agent = DuelingDDQNAgent()
    np_net = NumpyDuelingDDQN(agent.online_net)

    shift_length = int(params["SL"])
    roasters = list(params["roasters"])

    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "net_profit", "epsilon", "buffer_size",
            "loss", "decisions", "wall_sec",
        ])

    t0 = time.perf_counter()
    best_profit = -float("inf")
    total_decisions = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for episode in range(num_episodes):
        ups = generate_ups_events(ups_lambda, ups_mu, episode, shift_length, roasters)

        kpi, n_dec, tool_counts = run_fast_episode(
            engine, agent, toolkit, data,
            agent.replay_buffer, ups, training=True,
            np_net=np_net,
        )
        total_decisions += n_dec

        # Train: fixed small number of gradient steps per episode
        if len(agent.replay_buffer) >= C.BATCH_SIZE:
            for _ in range(C.TRAINS_PER_EP):
                agent.train_step()
            np_net.sync(agent.online_net)

        agent.decay_epsilon(episode, num_episodes)

        net_profit = float(kpi.net_profit())

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if episode % C.LOG_INTERVAL == 0:
            elapsed = time.perf_counter() - t0
            eps_per_sec = max(1, episode) / elapsed if elapsed > 0 else 0
            eta_h = (num_episodes - episode) / max(1, eps_per_sec) / 3600
            print(
                f"[{episode:>7d}/{num_episodes}]  "
                f"profit={net_profit:>10.1f}  "
                f"eps={agent.epsilon:.3f}  "
                f"buf={len(agent.replay_buffer):>6d}  "
                f"loss={agent._last_loss:.4f}  "
                f"dec={n_dec:>4d}  "
                f"tools={tool_counts}  "
                f"wall={elapsed:.0f}s  "
                f"({eps_per_sec:.0f} ep/s, ETA {eta_h:.1f}h)"
            )
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode, round(net_profit, 2), round(agent.epsilon, 4),
                    len(agent.replay_buffer), round(agent._last_loss, 6),
                    n_dec, round(elapsed, 1),
                ])

        # Checkpoints
        if episode > 0 and episode % C.CHECKPOINT_INTERVAL == 0:
            ckpt_path = output_dir / f"rlhh_ep{episode}.pt"
            agent.save_checkpoint(ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}")

        if net_profit > best_profit:
            best_profit = net_profit
            agent.save_checkpoint(output_dir / "rlhh_best.pt")

    # Final checkpoint
    agent.save_checkpoint(output_dir / "rlhh_final.pt")
    elapsed = time.perf_counter() - t0
    print(f"\nTraining complete.  {num_episodes} episodes in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Best profit: {best_profit:.1f}  |  Total decisions: {total_decisions}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _main_train_single(args) -> None:
    train(
        num_episodes=args.episodes,
        ups_lambda=args.ups_lambda,
        ups_mu=args.ups_mu,
        output_dir=str(args.output_dir),
        input_dir=args.input_dir,
    )



# ============================================================================
# === train_cycle.py
# ============================================================================







def train_one_cycle(
    cycle: int,
    time_sec: float,
    output_dir: str | Path = "test_rl_hh/outputs",
    episode_budget_for_eps: int = 300_000,
    warm_start: str | Path | None = None,
    epsilon_start: float | None = None,
    lr: float | None = None,
) -> dict:
    """Train one cycle within wall-clock budget.

    warm_start: optional checkpoint path to load as starting weights (fine-tune mode).
    epsilon_start: override initial ε (useful for fine-tune with low exploration).
    episode_budget_for_eps is used only for the ε-decay schedule (should match
    the expected episode count so ε lands at ε_end by the end).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    params = data.to_env_params()
    ups_lambda = data.ups_lambda
    ups_mu = data.ups_mu
    SL = int(params["SL"])
    roasters = list(params["roasters"])

    engine = SimulationEngine(params)
    toolkit = ToolKit(engine, params)
    agent = DuelingDDQNAgent(lr=lr if lr is not None else C.LR)
    if lr is not None:
        print(f"  Override LR = {lr}")
    if warm_start:
        print(f"  Warm-starting from {warm_start}")
        agent.load_checkpoint(warm_start)
        # load_checkpoint restores optimizer state with OLD lr; reset lr explicitly
        if lr is not None:
            for g in agent.optimizer.param_groups:
                g["lr"] = lr
    if epsilon_start is not None:
        agent.epsilon = epsilon_start
        agent._eps_start = epsilon_start
        print(f"  Override initial eps = {epsilon_start}")
    np_net = NumpyDuelingDDQN(agent.online_net)

    log_path = output_dir / f"cycle{cycle}_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "net_profit", "best_profit",
            "epsilon", "buffer_size", "wall_sec", "tool_counts",
        ])

    t0 = time.perf_counter()
    episode = 0
    best_profit = -float("inf")
    best_ep = 0

    best_ckpt = output_dir / f"cycle{cycle}_best.pt"
    final_ckpt = output_dir / f"cycle{cycle}_final.pt"

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= time_sec:
            break

        ups = generate_ups_events(ups_lambda, ups_mu, episode + cycle * 1_000_000, SL, roasters)
        kpi, n_dec, tc = run_fast_episode(
            engine, agent, toolkit, data,
            agent.replay_buffer, ups, training=True, np_net=np_net,
        )
        if len(agent.replay_buffer) >= C.BATCH_SIZE:
            for _ in range(C.TRAINS_PER_EP):
                agent.train_step()
            np_net.sync(agent.online_net)

        agent.decay_epsilon(episode, episode_budget_for_eps)
        episode += 1

        ep_profit = float(kpi.net_profit())
        if ep_profit > best_profit:
            best_profit = ep_profit
            best_ep = episode
            agent.save_checkpoint(best_ckpt)

        if episode % C.LOG_INTERVAL == 0:
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            print(
                f"  [C{cycle} ep={episode:>7d}] "
                f"profit={ep_profit:>10.1f}  "
                f"best={best_profit:>10.1f}  "
                f"eps={agent.epsilon:.3f}  "
                f"buf={len(agent.replay_buffer):>6d}  "
                f"tools={tc}  "
                f"wall={elapsed:.0f}s  "
                f"({eps_per_sec:.0f} ep/s)"
            )
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode, round(ep_profit, 2), round(best_profit, 2),
                    round(agent.epsilon, 4), len(agent.replay_buffer),
                    round(elapsed, 1), json.dumps(tc),
                ])

    agent.save_checkpoint(final_ckpt)
    elapsed = time.perf_counter() - t0
    summary = {
        "cycle": cycle,
        "episodes": episode,
        "wall_sec": round(elapsed, 1),
        "best_training_profit": round(best_profit, 2),
        "best_training_episode": best_ep,
        "final_epsilon": round(agent.epsilon, 4),
        "best_ckpt": str(best_ckpt),
        "final_ckpt": str(final_ckpt),
    }
    print(f"\n  Cycle {cycle} training complete: {episode} episodes in {elapsed:.0f}s")
    print(f"  Best training profit: {best_profit:.1f} at ep {best_ep}")
    print(f"  Saved: {best_ckpt} and {final_ckpt}")

    with open(output_dir / f"cycle{cycle}_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _main_train_cycle(args) -> None:
    train_one_cycle(
        args.cycle, args.time_sec, str(args.output_dir), args.eps_budget,
        warm_start=args.warm_start, epsilon_start=args.epsilon_start,
        lr=args.lr,
    )


# ============================================================================
# Unified CLI dispatcher
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    """Unified CLI for rl_hh.train.

    Subcommands:
      single  — single-shot training (uses NUM_EPISODES). Default if none given.
      cycle   — time-budgeted single-cycle training (warm-start friendly).

    Output goes to ``Results/<ts>_RLHH_<RunName>/``.
    """
    from evaluation.result_schema import make_run_dir

    parser = argparse.ArgumentParser(description="Train RL-HH (Dueling DDQN hyper-heuristic).")
    parser.add_argument("--name", required=True, help="Run name (used in Results/ folder)")
    parser.add_argument("--input-dir", default=None, help="Override Input_data path")

    sub = parser.add_subparsers(dest="cmd")

    # single subcommand
    p_single = sub.add_parser("single", help="Single-shot training")
    p_single.add_argument("--episodes", type=int, default=C.NUM_EPISODES)
    p_single.add_argument("--ups-lambda", type=float, default=None)
    p_single.add_argument("--ups-mu", type=float, default=None)

    # cycle subcommand
    p_cycle = sub.add_parser("cycle", help="Time-budgeted cycle training")
    p_cycle.add_argument("--cycle", type=int, default=1)
    p_cycle.add_argument("--time-sec", type=float, default=600.0)
    p_cycle.add_argument("--eps-budget", type=int, default=300_000)
    p_cycle.add_argument("--warm-start", default=None)
    p_cycle.add_argument("--epsilon-start", type=float, default=None)
    p_cycle.add_argument("--lr", type=float, default=None)

    args = parser.parse_args(argv)
    args.output_dir = make_run_dir("RLHH", args.name)
    print(f"[rl_hh] Output -> {args.output_dir}")

    if args.cmd is None or args.cmd == "single":
        # Use defaults if 'single' subcommand not specified
        if not hasattr(args, "episodes"):
            args.episodes = C.NUM_EPISODES
            args.ups_lambda = None
            args.ups_mu = None
        _main_train_single(args)
    elif args.cmd == "cycle":
        _main_train_cycle(args)
    else:
        parser.error(f"unknown subcommand: {args.cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    main()


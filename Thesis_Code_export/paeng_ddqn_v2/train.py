"""Paeng DDQN v2 — period-based parameter-sharing Dueling DDQN (Paeng 2021 port).

Single-file consolidation of the v2 training pipeline: agent (config + network +
replay buffer + agent), strategy (state builder + feasibility + decision adapter),
and the time-budgeted training loop.

Outputs land in ``Results/<YYYYMMDD_HHMMSS>_PaengDDQNv2_<RunName>/`` via
``evaluation.result_schema.make_run_dir``.

Usage:
    python -m paeng_ddqn_v2.train --name SmokeV2 --time-sec 60
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
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Project root on sys.path so env.* + evaluation.* import cleanly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.data_loader import load_data
from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events

# ============================================================================
# === agent_v2.py
# ============================================================================





@dataclass
class PaengConfigV2:
    """Hyperparameters faithful to Paeng et al. (2021) Table II."""

    action_dim: int = 9
    state_rows: int = 3
    state_cols: int = 35   # 25 paper-faithful + 10 domain features (RC/GC/pipeline/time/UPS)
    sf_dim: int = 3

    lr: float = 0.0025
    gamma: float = 1.0
    batch_size: int = 64
    buffer_size: int = 100_000
    warmup_timesteps: int = 2_000
    freq_target_episodes: int = 50

    eps_start: float = 0.2
    eps_end: float = 0.0
    eps_ratio: float = 0.8

    is_double: bool = True
    is_duel: bool = True
    tau: float = 1.0
    huber_delta: float = 1.0

    target_episode_estimate: int = 100_000
    period_length: int = 11


class ParameterSharingDQNV2(nn.Module):
    """Parameter-sharing DQN per Paeng paper.

    Input: state (B, 3, 25) + vec(Sf) auxin (B, 3).
    Shared encoder per family row, then fusion and dueling heads.
    """

    def __init__(self, state_rows: int, state_cols: int, sf_dim: int, action_dim: int, is_duel: bool = True):
        super().__init__()
        self.state_rows = state_rows
        self.state_cols = state_cols
        self.sf_dim = sf_dim
        self.action_dim = action_dim
        self.is_duel = is_duel

        # Shared encoder: applies to each row independently
        self.encoder = nn.Sequential(
            nn.Linear(state_cols, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Fusion layer after row concatenation + sf auxin
        fusion_input_dim = state_rows * 16 + sf_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 32),
            nn.ReLU(),
        )

        if is_duel:
            # Dueling heads
            self.v_head = nn.Linear(32, 1)
            self.a_head = nn.Linear(32, action_dim)
        else:
            # Standard Q-head
            self.q_head = nn.Linear(32, action_dim)

    def forward(self, state: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, 3, 25)
            sf: (B, 3)
        Returns:
            q_values: (B, 9)
        """
        B = state.size(0)

        # Encode each row
        encoded_rows = []
        for i in range(self.state_rows):
            row_i = state[:, i, :]  # (B, 25)
            encoded_i = self.encoder(row_i)  # (B, 16)
            encoded_rows.append(encoded_i)

        # Concatenate rows + sf auxin
        encoded_concat = torch.cat(encoded_rows + [sf], dim=1)  # (B, 48+3) = (B, 51)
        fused = self.fusion(encoded_concat)  # (B, 32)

        if self.is_duel:
            v = self.v_head(fused)  # (B, 1)
            a = self.a_head(fused)  # (B, 9)
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q
        else:
            return self.q_head(fused)


class ReplayBufferV2:
    """Circular replay buffer for experience tuples."""

    def __init__(self, buffer_size: int, state_shape: tuple, sf_shape: tuple, action_dim: int):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.sf_shape = sf_shape
        self.action_dim = action_dim
        self.idx = 0
        self.size = 0

        # Pre-allocate numpy arrays
        self.state = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.sf = np.zeros((buffer_size, *sf_shape), dtype=np.float32)
        self.action = np.zeros((buffer_size,), dtype=np.int64)
        self.reward = np.zeros((buffer_size,), dtype=np.float32)
        self.terminal = np.zeros((buffer_size,), dtype=bool)
        self.next_state = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.next_sf = np.zeros((buffer_size, *sf_shape), dtype=np.float32)
        self.next_feas = np.zeros((buffer_size, action_dim), dtype=bool)

    def add(self, state, sf, action, reward, terminal, next_state, next_sf, next_feas):
        self.state[self.idx] = state
        self.sf[self.idx] = sf
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.terminal[self.idx] = terminal
        self.next_state[self.idx] = next_state
        self.next_sf[self.idx] = next_sf
        self.next_feas[self.idx] = next_feas

        self.idx = (self.idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample_batch(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.from_numpy(self.state[indices]),
            torch.from_numpy(self.sf[indices]),
            torch.from_numpy(self.action[indices]),
            torch.from_numpy(self.reward[indices]),
            torch.from_numpy(self.terminal[indices]),
            torch.from_numpy(self.next_state[indices]),
            torch.from_numpy(self.next_sf[indices]),
            torch.from_numpy(self.next_feas[indices]),
        )

    def __len__(self):
        return self.size


class PaengAgentV2:
    """Paeng DDQN agent with period-based decisions."""

    def __init__(self, config: PaengConfigV2, device: str = "cpu"):
        self.config = config
        self.device = device

        # Networks
        self.online = ParameterSharingDQNV2(
            config.state_rows, config.state_cols, config.sf_dim,
            config.action_dim, is_duel=config.is_duel
        ).to(device)
        self.target = ParameterSharingDQNV2(
            config.state_rows, config.state_cols, config.sf_dim,
            config.action_dim, is_duel=config.is_duel
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        # Optimizer (paper doesn't specify, use Adam as standard)
        self.optimizer = optim.Adam(self.online.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss(beta=config.huber_delta)

        # Replay buffer
        self.replay = ReplayBufferV2(
            config.buffer_size,
            state_shape=(config.state_rows, config.state_cols),
            sf_shape=(config.sf_dim,),
            action_dim=config.action_dim,
        )

        # Training state
        self.epsilon = config.eps_start
        self.timestep = 0
        self.episode = 0

    def select_action(
        self,
        state: np.ndarray,
        sf: np.ndarray,
        feas_mask: np.ndarray,
        training: bool = True,
    ) -> int:
        """ε-greedy action selection with feasibility mask.

        Args:
            state: (3, 25)
            sf: (3,)
            feas_mask: (9,) boolean array
            training: if True, apply ε-greedy; else greedy

        Returns:
            action_id: int in [0, 8]
        """
        if training and np.random.random() < self.epsilon:
            # Random from feasible
            feasible_actions = np.where(feas_mask)[0]
            if len(feasible_actions) == 0:
                feasible_actions = np.arange(self.config.action_dim)
            return int(np.random.choice(feasible_actions))
        else:
            # Greedy: argmax over feasible
            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 3, 25)
                sf_t = torch.from_numpy(sf).unsqueeze(0).to(self.device)  # (1, 3)
                q = self.online(state_t, sf_t)  # (1, 9)
                q_np = q.cpu().numpy()[0]  # (9,)

            # Mask infeasible actions with -inf
            q_masked = q_np.copy()
            q_masked[~feas_mask] = -np.inf

            # Argmax over masked Q-values
            action = int(np.argmax(q_masked))
            return action

    def store_transition(
        self,
        state,
        sf,
        action,
        reward,
        terminal,
        next_state,
        next_sf,
        next_feas,
    ):
        """Store transition in replay buffer."""
        self.replay.add(state, sf, action, reward, terminal, next_state, next_sf, next_feas)
        self.timestep += 1

    def train_step(self) -> float:
        """One SGD step with Double-Q target.

        Returns:
            loss value (float)
        """
        if len(self.replay) < self.config.batch_size or self.timestep < self.config.warmup_timesteps:
            return 0.0

        # Sample batch
        state, sf, action, reward, terminal, next_state, next_sf, next_feas = self.replay.sample_batch(
            self.config.batch_size
        )

        state = state.to(self.device)
        sf = sf.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        terminal = terminal.to(self.device)
        next_state = next_state.to(self.device)
        next_sf = next_sf.to(self.device)
        next_feas = next_feas.to(self.device)

        # Double DQN: online selects, target evaluates
        with torch.no_grad():
            next_q_online = self.online(next_state, next_sf)  # (B, 9)

            # Mask infeasible actions
            next_q_online_masked = next_q_online.clone()
            next_q_online_masked[~next_feas] = -1e9

            next_action = torch.argmax(next_q_online_masked, dim=1)  # (B,)

            next_q_target = self.target(next_state, next_sf)  # (B, 9)
            next_q_max = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze(1)  # (B,)

            # Compute target
            target_q = reward + self.config.gamma * next_q_max * (~terminal).float()  # (B,)

        # Compute online Q and loss
        q_online = self.online(state, sf)  # (B, 9)
        q_selected = q_online.gather(1, action.unsqueeze(1)).squeeze(1)  # (B,)

        loss = self.loss_fn(q_selected, target_q)

        # SGD
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Hard copy: target := online (τ=1.0)."""
        self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon(self, episode_idx: int, max_episodes: int):
        """Linear decay from eps_start to eps_end over eps_ratio fraction of episodes."""
        decay_episodes = int(self.config.eps_ratio * max_episodes)
        if episode_idx < decay_episodes:
            self.epsilon = self.config.eps_start - (
                self.config.eps_start - self.config.eps_end
            ) * (episode_idx / decay_episodes)
        else:
            self.epsilon = self.config.eps_end

    def save_checkpoint(self, path: Path | str):
        """Save agent state."""
        path = Path(path)
        torch.save(
            {
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "epsilon": self.epsilon,
                "timestep": self.timestep,
                "episode": self.episode,
            },
            path,
        )

    def load_checkpoint(self, path: Path | str):
        """Load agent state."""
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = ckpt.get("epsilon", self.config.eps_start)
        self.timestep = ckpt.get("timestep", 0)
        self.episode = ckpt.get("episode", 0)

    @classmethod
    def from_checkpoint(cls, path: Path | str, device: str = "cpu") -> PaengAgentV2:
        """Load agent from checkpoint."""
        path = Path(path)
        ckpt = torch.load(path, map_location=device)
        config_dict = ckpt["config"]
        config = PaengConfigV2(**config_dict)
        agent = cls(config, device=device)
        agent.load_checkpoint(path)
        return agent


# ============================================================================
# === strategy_v2.py
# ============================================================================






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


# ============================================================================
# === train_v2.py
# ============================================================================







def train(
    time_sec: float,
    output_dir: Path,
    seed_base: int = 42,
    use_ups: bool = True,
    target_episode_estimate: int = 100_000,
    snapshot_every: int = 0,
    rolling_window: int = 0,
    restore_drop_threshold: float = 50_000.0,
    load_ckpt: str | None = None,
    initial_epsilon: float | None = None,
    run_seed: int | None = None,
) -> dict:
    """Time-budgeted training loop for Paeng DDQN v2."""

    if run_seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(run_seed)
        _np.random.seed(run_seed)
        try:
            import torch as _torch
            _torch.manual_seed(run_seed)
        except ImportError:
            pass

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    params = data.to_env_params()
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    ups_lambda = float(data.ups_lambda)
    ups_mu = float(data.ups_mu)

    cfg = PaengConfigV2()
    cfg.target_episode_estimate = target_episode_estimate
    agent = PaengAgentV2(cfg)

    if load_ckpt is not None:
        agent.load_checkpoint(load_ckpt)
        print(f"[paeng-train-v2] warm-start: loaded {load_ckpt}; epsilon={agent.epsilon:.3f}")

    if initial_epsilon is not None:
        agent.epsilon = float(initial_epsilon)
        print(f"[paeng-train-v2] initial epsilon override: {agent.epsilon:.3f}")

    strategy = PaengStrategyV2(agent, data, training=True, params=params)
    engine = SimulationEngine(params)

    log_path = output_dir / "training_log.csv"
    summary_path = output_dir / "training_summary.json"
    best_ckpt = output_dir / "paeng_v2_best.pt"
    final_ckpt = output_dir / "paeng_v2_final.pt"
    best_rolling_ckpt = output_dir / "paeng_v2_best_rolling.pt"
    snapshot_dir = output_dir / "snapshots"
    if snapshot_every > 0:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths: list[str] = []

    rolling: deque[float] = deque(maxlen=rolling_window) if rolling_window > 0 else deque()
    best_rolling_mean = -float("inf")
    best_rolling_episode = -1
    n_restores = 0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "episode", "profit", "tardiness_cost", "period_decisions", "epsilon",
            "buffer_size", "wall_sec", "loss_avg", "action_dist",
        ])

    t0 = time.perf_counter()
    episode = 0
    best_profit = -float("inf")
    best_episode = 0
    losses_window: list[float] = []
    period_decisions_per_ep = 0

    print(f"[paeng-train-v2] starting: budget={time_sec:.0f}s, target_episodes={target_episode_estimate}, "
          f"ups={use_ups}, lambda={ups_lambda}, mu={ups_mu}, seed_base={seed_base}")

    try:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= time_sec or episode >= target_episode_estimate:
                break

            seed = seed_base + episode
            ups = generate_ups_events(ups_lambda, ups_mu, seed, SL, roasters) if use_ups else []

            strategy.reset_episode()
            strategy.action_counts = {i: 0 for i in range(9)}

            kpi, sim_state = engine.run(strategy, ups)
            final_profit = float(kpi.net_profit())
            tardiness_cost = float(kpi.tard_cost)
            period_decisions_per_ep = sum(strategy.action_counts.values())

            strategy.end_episode(sim_state, final_profit, tardiness_cost)

            if initial_epsilon is None:
                agent.decay_epsilon(episode, cfg.target_episode_estimate)

            if episode % cfg.freq_target_episodes == 0:
                agent.update_target_network()

            if len(agent.replay) >= cfg.batch_size and agent.timestep > cfg.warmup_timesteps:
                losses_window.append(agent.train_step())
                if len(losses_window) > 100:
                    losses_window.pop(0)
            loss_avg = sum(losses_window) / len(losses_window) if losses_window else 0.0

            if final_profit > best_profit:
                best_profit = final_profit
                best_episode = episode
                agent.save_checkpoint(best_ckpt)

            # Rolling-mean collapse-restore
            if rolling_window > 0:
                rolling.append(final_profit)
                if len(rolling) >= rolling_window:
                    rolling_mean = sum(rolling) / len(rolling)
                    if rolling_mean > best_rolling_mean:
                        best_rolling_mean = rolling_mean
                        best_rolling_episode = episode
                        agent.save_checkpoint(best_rolling_ckpt)
                    # Restore if rolling mean drops too far
                    if (
                        agent.epsilon <= cfg.eps_end + 1e-9
                        and best_rolling_mean > -1e9
                        and rolling_mean < best_rolling_mean - restore_drop_threshold
                        and best_rolling_ckpt.exists()
                    ):
                        agent.load_checkpoint(best_rolling_ckpt)
                        rolling.clear()
                        n_restores += 1
                        print(f"  [restore #{n_restores}] ep{episode}  rolling=${rolling_mean:>10,.0f}  "
                              f"best=${best_rolling_mean:>10,.0f} (ep{best_rolling_episode})")

            # Snapshots
            if snapshot_every > 0 and agent.epsilon <= cfg.eps_end + 1e-9 and episode % snapshot_every == 0 and episode > 0:
                snap_path = snapshot_dir / f"ckpt_ep{episode}.pt"
                agent.save_checkpoint(snap_path)
                snapshot_paths.append(str(snap_path))

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    episode,
                    round(final_profit, 2),
                    round(tardiness_cost, 2),
                    period_decisions_per_ep,
                    round(agent.epsilon, 4),
                    len(agent.replay),
                    round(elapsed, 1),
                    round(loss_avg, 6),
                    json.dumps(dict(strategy.action_counts)),
                ])

            if episode % 50 == 0 or elapsed > time_sec - 5:
                eps_per_sec = (episode + 1) / max(elapsed, 1e-3)
                print(f"  ep {episode:>5d}  profit=${final_profit:>10,.0f}  "
                      f"best=${best_profit:>10,.0f} (ep {best_episode})  "
                      f"eps={agent.epsilon:.3f}  buf={len(agent.replay):>6d}  "
                      f"loss={loss_avg:.4f}  wall={elapsed:.0f}s  ({eps_per_sec:.1f} ep/s)")

            episode += 1

    except KeyboardInterrupt:
        print("[paeng-train-v2] interrupted by user; saving final checkpoint anyway")

    agent.save_checkpoint(final_ckpt)
    elapsed = time.perf_counter() - t0
    summary = {
        "episodes": episode,
        "wall_sec": round(elapsed, 1),
        "best_profit": round(best_profit, 2),
        "best_episode": best_episode,
        "final_epsilon": round(agent.epsilon, 4),
        "final_buffer_size": len(agent.replay),
        "best_ckpt": str(best_ckpt),
        "final_ckpt": str(final_ckpt),
        "config": {
            "action_dim": cfg.action_dim,
            "state_rows": cfg.state_rows,
            "state_cols": cfg.state_cols,
            "sf_dim": cfg.sf_dim,
            "lr": cfg.lr,
            "gamma": cfg.gamma,
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.buffer_size,
            "eps_start": cfg.eps_start,
            "eps_end": cfg.eps_end,
            "eps_ratio": cfg.eps_ratio,
            "is_double": cfg.is_double,
            "is_duel": cfg.is_duel,
            "tau": cfg.tau,
            "huber_delta": cfg.huber_delta,
        },
        "use_ups": use_ups,
        "ups_lambda": ups_lambda,
        "ups_mu": ups_mu,
        "seed_base": seed_base,
        "best_rolling_mean": round(best_rolling_mean, 2) if best_rolling_mean > -1e9 else None,
        "best_rolling_episode": best_rolling_episode if best_rolling_episode >= 0 else None,
        "best_rolling_ckpt": str(best_rolling_ckpt) if rolling_window > 0 else None,
        "n_restores": n_restores,
        "snapshot_every": snapshot_every,
        "snapshot_paths": snapshot_paths,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[paeng-train-v2] done.  episodes={episode}  best=${best_profit:,.0f} "
          f"(ep {best_episode})  wall={elapsed:.0f}s")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Paeng DDQN v2 (period-based DDQN, Paeng 2021 port).")
    parser.add_argument("--name", required=True, help="Run name (used in Results/ folder)")
    parser.add_argument("--time-sec", type=float, default=600.0, help="Wall-clock budget (seconds)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output dir (default: Results/<ts>_PaengDDQNv2_<name>/)")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--no-ups", action="store_true")
    parser.add_argument("--target-episodes", type=int, default=100_000)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--rolling-window", type=int, default=0)
    parser.add_argument("--restore-drop-threshold", type=float, default=50_000.0)
    parser.add_argument("--load-ckpt", type=str, default=None)
    parser.add_argument("--initial-epsilon", type=float, default=None)
    parser.add_argument("--run-seed", type=int, default=None)
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip post-training single-seed eval (default: run it).")
    parser.add_argument("--eval-seed", type=int, default=42,
                        help="Seed for the post-training eval (default: 42).")
    args = parser.parse_args(argv)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = _PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        from evaluation.result_schema import make_run_dir
        out_dir = make_run_dir("PaengDDQNv2", args.name)
    print(f"[paeng_v2] Output -> {out_dir}")

    train(
        time_sec=args.time_sec,
        output_dir=out_dir,
        seed_base=args.seed_base,
        use_ups=not args.no_ups,
        target_episode_estimate=args.target_episodes,
        snapshot_every=args.snapshot_every,
        rolling_window=args.rolling_window,
        restore_drop_threshold=args.restore_drop_threshold,
        load_ckpt=args.load_ckpt,
        initial_epsilon=args.initial_epsilon,
        run_seed=args.run_seed,
    )

    # Post-training single-seed eval -> result.json + report.html for plot/master_eval.
    if not args.no_eval:
        best_ckpt = out_dir / "paeng_v2_best.pt"
        if best_ckpt.exists():
            print(f"\n{'=' * 72}\nAUTO-EVALUATION: greedy single-seed eval (seed={args.eval_seed})\n{'=' * 72}")
            from paeng_ddqn_v2.evaluate import main as eval_main
            eval_main([
                "--checkpoint", str(best_ckpt),
                "--name", f"{args.name}_AutoEval",
                "--seed", str(args.eval_seed),
            ])
            # Also copy result.json into the training run dir so master_eval finds it.
            from evaluation.result_schema import DEFAULT_RESULTS_DIR  # noqa: F401 just to ensure import
            results_root = _PROJECT_ROOT / "Results"
            eval_dirs = sorted(
                [d for d in results_root.iterdir()
                 if d.is_dir() and "_Eval_PaengDDQNv2_" in d.name and d.name.endswith(f"{args.name}_AutoEval")],
                key=lambda d: d.stat().st_mtime, reverse=True,
            )
            if eval_dirs:
                src_json = eval_dirs[0] / "result.json"
                if src_json.exists():
                    import shutil
                    shutil.copy2(src_json, out_dir / "result.json")
                    src_html = eval_dirs[0] / "report.html"
                    if src_html.exists():
                        shutil.copy2(src_html, out_dir / "report.html")
                    print(f"[paeng_v2] Copied result.json + report.html into {out_dir}")
        else:
            print(f"[paeng_v2] No best checkpoint at {best_ckpt} - skipping auto-eval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


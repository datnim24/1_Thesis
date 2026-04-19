"""Fast episode runner using SimulationEngine internals directly.

Bypasses the Gymnasium wrapper.  Uses the same simulation phases as
engine.run() but collects DDQN transitions inline with incremental
KPI profit as reward.

Key optimisations:
  - No decision queue / info-dict / violation-check overhead
  - Base observation built once per slot, context appended cheaply
  - Inference via pure-numpy forward pass (no PyTorch call overhead)
  - Random exploration skips network entirely
"""

from __future__ import annotations

import random as _random
from collections import defaultdict

import numpy as np

from PPOmask.Engine.action_spec import ACTION_BY_ID, WAIT_ACTION_ID
from PPOmask.Engine.observation_spec import (
    ROASTER_ORDER,
    GC_FEATURE_ORDER,
    STATUS_ENCODING,
    SKU_ENCODING,
    PIPELINE_MODE_ENCODING,
)

from . import configs as C
from .meta_agent import DuelingDDQNAgent
from .numpy_net import NumpyDuelingDDQN
from .replay_buffer import ReplayBuffer
from .tools import ToolKit


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
            elif training and _random.random() < epsilon:
                tool_ids[i] = _random.choice(valid)
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
                import torch
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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ROASTER_ORDER = ("R1", "R2", "R3", "R4", "R5")
GC_FEATURE_ORDER = (
    ("L1", "PSC"),
    ("L1", "NDG"),
    ("L1", "BUSTA"),
    ("L2", "PSC"),
)
CONTEXT_ORDER = ("RESTOCK", "R1", "R2", "R3", "R4", "R5")
OBS_BASE_DIM = 27
OBS_CONTEXT_DIM = len(CONTEXT_ORDER)
OBS_TOTAL_DIM = OBS_BASE_DIM + OBS_CONTEXT_DIM


STATUS_ENCODING = {
    "IDLE": 0.0,
    "RUNNING": 1.0 / 3.0,
    "SETUP": 2.0 / 3.0,
    "DOWN": 1.0,
}
SKU_ENCODING = {
    "PSC": 0.0,
    "NDG": 0.5,
    "BUSTA": 1.0,
}
PIPELINE_MODE_ENCODING = {
    "FREE": 0.0,
    "CONSUME": 0.5,
    "RESTOCK": 1.0,
}


@dataclass(frozen=True)
class ObservationContext:
    kind: str
    roaster_id: str | None = None

    @property
    def key(self) -> str:
        return self.roaster_id if self.kind == "ROASTER" and self.roaster_id else "RESTOCK"


def _normalized_timer(value: int, normalizer: int) -> float:
    if normalizer <= 0:
        return 0.0
    return float(max(0, min(value, normalizer))) / float(normalizer)


def build_base_observation(data, state) -> np.ndarray:
    base = np.zeros(OBS_BASE_DIM, dtype=np.float32)
    base[0] = float(state.t) / float(max(1, data.shift_length))

    roaster_timer_norm = max(
        data.shift_length,
        max(data.roast_time_by_sku.values()),
        data.setup_time,
    )
    for offset, roaster_id in enumerate(ROASTER_ORDER, start=1):
        base[offset] = STATUS_ENCODING[state.status[roaster_id]]
    for offset, roaster_id in enumerate(ROASTER_ORDER, start=6):
        base[offset] = _normalized_timer(state.remaining[roaster_id], roaster_timer_norm)
    for offset, roaster_id in enumerate(ROASTER_ORDER, start=11):
        base[offset] = SKU_ENCODING[state.last_sku[roaster_id]]

    base[16] = float(state.rc_stock["L1"]) / float(data.max_rc)
    base[17] = float(state.rc_stock["L2"]) / float(data.max_rc)
    total_mto_initial = max(1, sum(data.job_batches.values()))
    base[18] = float(sum(state.mto_remaining.values())) / float(total_mto_initial)
    base[19] = PIPELINE_MODE_ENCODING[state.pipeline_mode["L1"]]
    base[20] = PIPELINE_MODE_ENCODING[state.pipeline_mode["L2"]]

    for index, pair in enumerate(GC_FEATURE_ORDER, start=21):
        capacity = data.gc_capacity[pair]
        base[index] = float(state.gc_stock[pair]) / float(max(1, capacity))

    base[25] = 1.0 if state.restock_busy > 0 else 0.0
    base[26] = _normalized_timer(state.restock_busy, data.restock_duration)
    return base


def build_observation(data, state, context: ObservationContext) -> np.ndarray:
    base = build_base_observation(data, state)
    context_vec = np.zeros(OBS_CONTEXT_DIM, dtype=np.float32)
    context_vec[CONTEXT_ORDER.index(context.key)] = 1.0
    return np.concatenate([base, context_vec], dtype=np.float32)

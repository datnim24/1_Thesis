from __future__ import annotations

import numpy as np

from .action_spec import (
    ACTION_COUNT,
    ACTION_BY_ID,
    RESERVED_ACTION_IDS,
    RESTOCK_ACTION_IDS,
    ROASTER_ACTION_IDS,
    WAIT_ACTION_ID,
)


def compute_action_mask(data, engine, state, context) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT, dtype=bool)
    mask[WAIT_ACTION_ID] = True
    for action_id in RESERVED_ACTION_IDS:
        mask[action_id] = False

    if context.kind == "RESTOCK":
        for action_id in RESTOCK_ACTION_IDS:
            action = ACTION_BY_ID[action_id]
            assert action.line_id is not None and action.sku is not None
            mask[action_id] = engine.can_start_restock(state, action.line_id, action.sku)
        return mask

    roaster_id = context.roaster_id
    if roaster_id is None:
        raise ValueError("ROASTER context requires roaster_id")

    env_mask = engine._compute_action_mask(state, roaster_id)
    for action_id in ROASTER_ACTION_IDS[roaster_id]:
        mask[action_id] = env_mask.get(ACTION_BY_ID[action_id].env_action, False)
    return mask

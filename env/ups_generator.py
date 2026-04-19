"""Poisson UPS event generation for paired simulation experiments."""

from __future__ import annotations

import math
import random

from .simulation_state import UPSEvent


def generate_ups_events(
    lambda_rate: float,
    mu_mean: float,
    seed: int,
    shift_length: int = 480,
    roasters: list[str] | None = None,
) -> list[UPSEvent]:
    """Generate reproducible UPS events for one simulated shift.

    If *roasters* is not provided, defaults to ["R1","R2","R3","R4","R5"].
    """
    if lambda_rate == 0:
        return []

    if roasters is None:
        from .data_bridge import get_sim_params
        roasters = list(get_sim_params()["roasters"])

    rng = random.Random(seed)
    events: list[UPSEvent] = []
    current_time = 0.0

    while True:
        inter_arrival = rng.expovariate(lambda_rate / float(shift_length))
        current_time += inter_arrival
        if current_time >= shift_length:
            break
        slot = int(math.floor(current_time))
        raw_duration = max(1, int(round(rng.expovariate(1.0 / float(mu_mean)))))
        max_duration = int(3 * float(mu_mean))
        shift_remaining = shift_length - slot
        duration = min(raw_duration, max_duration, shift_remaining)
        duration = max(1, duration)
        roaster_id = rng.choice(roasters)
        events.append(UPSEvent(t=slot, roaster_id=roaster_id, duration=duration))

    return sorted(events, key=lambda e: (e.t, e.roaster_id))


def generate_experiment_seeds(n_reps: int, base_seed: int = 0) -> list[int]:
    """Generate distinct seeds for paired strategy comparisons."""
    rng = random.Random(base_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n_reps)]

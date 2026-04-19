"""Data loader: wraps env.data_bridge.get_sim_params() and adds PPO-specific parameters."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT_DIR = _ROOT / "Input_data"


@dataclass
class PPOData:
    """Immutable container for all simulation + PPO training parameters."""

    params: dict[str, Any]
    input_dir: Path

    # Convenience accessors (duplicated from params for fast attribute access)
    shift_length: int = 480
    max_rc: int = 40
    safety_stock: int = 20
    setup_time: int = 5
    roasters: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")
    lines: tuple[str, ...] = ("L1", "L2")
    roast_time_by_sku: dict[str, int] = field(default_factory=dict)
    gc_capacity: dict[tuple[str, str], int] = field(default_factory=dict)
    gc_init: dict[tuple[str, str], int] = field(default_factory=dict)
    restock_duration: int = 15
    restock_qty: int = 5
    ups_lambda: float = 0.0
    ups_mu: float = 0.0
    job_batches: dict[str, int] = field(default_factory=dict)
    consume_events: dict[str, list[int]] = field(default_factory=dict)

    # PPO-specific
    violation_penalty: float = 50000.0
    episode_termination_on_violation: bool = True
    rc_maintenance_bonus: float = 0.0
    completion_bonus: float = 0.0

    def to_env_params(self) -> dict[str, Any]:
        return self.params


def _read_param_map(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("parameter") or "").strip()
            val = row.get("value")
            if key:
                values[key] = "" if val is None else val.strip()
    return values


def load_data(input_dir: str | Path | None = None) -> PPOData:
    """Load simulation params via canonical data_bridge + PPO-specific extras."""
    import sys
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from env.data_bridge import get_sim_params

    resolved_dir = Path(input_dir) if input_dir else _DEFAULT_INPUT_DIR
    if not resolved_dir.is_absolute():
        resolved_dir = _ROOT / resolved_dir

    params = get_sim_params(str(resolved_dir))

    # Validate critical fields
    assert "roasters" in params, "Missing 'roasters' in params"
    assert "lines" in params, "Missing 'lines' in params"
    assert "gc_capacity" in params, "Missing 'gc_capacity' in params"
    assert "SL" in params, "Missing 'SL' (shift length) in params"
    assert "max_rc" in params, "Missing 'max_rc' in params"
    assert params["roasters"] == ["R1", "R2", "R3", "R4", "R5"], (
        f"Expected roasters [R1..R5], got {params['roasters']}"
    )
    assert sorted(params["lines"]) == ["L1", "L2"], (
        f"Expected lines [L1,L2], got {params['lines']}"
    )

    # Read PPO-specific params from shift_parameters.csv
    sp = _read_param_map(resolved_dir / "shift_parameters.csv")
    violation_penalty = float(sp.get("violation_penalty", "50000"))
    episode_term = sp.get("episode_termination_on_violation", "1").strip()
    episode_termination_on_violation = episode_term in ("1", "true", "True")
    rc_maintenance_bonus = float(sp.get("rc_maintenance_bonus", "0.0"))

    return PPOData(
        params=params,
        input_dir=resolved_dir,
        shift_length=int(params["SL"]),
        max_rc=int(params["max_rc"]),
        safety_stock=int(params["safety_stock"]),
        setup_time=int(params["sigma"]),
        roasters=tuple(params["roasters"]),
        lines=tuple(params["lines"]),
        roast_time_by_sku=dict(params["roast_time_by_sku"]),
        gc_capacity=dict(params["gc_capacity"]),
        gc_init=dict(params["gc_init"]),
        restock_duration=int(params["restock_duration"]),
        restock_qty=int(params["restock_qty"]),
        ups_lambda=float(params["ups_lambda"]),
        ups_mu=float(params["ups_mu"]),
        job_batches=dict(params["job_batches"]),
        consume_events=dict(params["consume_events"]),
        violation_penalty=violation_penalty,
        episode_termination_on_violation=episode_termination_on_violation,
        rc_maintenance_bonus=rc_maintenance_bonus,
    )

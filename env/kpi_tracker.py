"""Profit and KPI accumulator for simulation runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KPITracker:
    """Accumulate objective terms and report schema-compatible results."""

    psc_completed: int = 0
    ndg_completed: int = 0
    busta_completed: int = 0
    revenue_psc: float = 0.0
    revenue_ndg: float = 0.0
    revenue_busta: float = 0.0
    total_revenue: float = 0.0
    tardiness_min: dict = field(default_factory=dict)
    tard_cost: float = 0.0
    setup_events: int = 0
    setup_cost: float = 0.0
    stockout_events: dict = field(default_factory=dict)
    stockout_duration: dict = field(default_factory=dict)
    stockout_cost: float = 0.0
    idle_min_per_roaster: dict = field(default_factory=dict)
    idle_cost: float = 0.0
    over_min_per_roaster: dict = field(default_factory=dict)
    over_cost: float = 0.0
    total_compute_ms: float = 0.0
    num_resolves: int = 0
    restock_count: int = 0
    mto_skipped: int = 0

    def net_profit(self) -> float:
        return (
            self.total_revenue
            - self.tard_cost
            - self.setup_cost
            - self.stockout_cost
            - self.idle_cost
            - self.over_cost
        )

    def to_dict(self) -> dict:
        return {
            "net_profit": round(self.net_profit(), 2),
            "total_revenue": round(self.total_revenue, 2),
            "total_costs": round(
                self.tard_cost
                + self.setup_cost
                + self.stockout_cost
                + self.idle_cost
                + self.over_cost,
                2,
            ),
            "psc_count": self.psc_completed,
            "ndg_count": self.ndg_completed,
            "busta_count": self.busta_completed,
            "revenue_psc": round(self.revenue_psc, 2),
            "revenue_ndg": round(self.revenue_ndg, 2),
            "revenue_busta": round(self.revenue_busta, 2),
            "tardiness_min": {
                job_id: round(float(value), 2)
                for job_id, value in self.tardiness_min.items()
            },
            "tard_cost": round(self.tard_cost, 2),
            "setup_events": int(self.setup_events),
            "setup_cost": round(self.setup_cost, 2),
            "stockout_events": dict(self.stockout_events),
            "stockout_duration": dict(self.stockout_duration),
            "stockout_cost": round(self.stockout_cost, 2),
            "idle_min": round(sum(self.idle_min_per_roaster.values()), 2),
            "idle_cost": round(self.idle_cost, 2),
            "over_min": round(sum(self.over_min_per_roaster.values()), 2),
            "over_cost": round(self.over_cost, 2),
            "total_compute_ms": round(self.total_compute_ms, 3),
            "num_resolves": int(self.num_resolves),
            "restock_count": int(self.restock_count),
        }

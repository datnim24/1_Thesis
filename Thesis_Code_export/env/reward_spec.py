from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardBreakdown:
    total_revenue: float
    tard_cost: float
    setup_cost: float
    stockout_cost: float
    idle_cost: float
    over_cost: float
    net_profit: float


def reward_breakdown_from_kpi(kpi) -> RewardBreakdown:
    return RewardBreakdown(
        total_revenue=float(kpi.total_revenue),
        tard_cost=float(kpi.tard_cost),
        setup_cost=float(kpi.setup_cost),
        stockout_cost=float(kpi.stockout_cost),
        idle_cost=float(kpi.idle_cost),
        over_cost=float(kpi.over_cost),
        net_profit=float(kpi.net_profit()),
    )


def incremental_profit(previous_profit: float, current_profit: float) -> float:
    return float(current_profit) - float(previous_profit)


def violation_reward(violation_penalty: float) -> float:
    """Return the penalty reward applied on hard constraint violation."""
    return -abs(violation_penalty)

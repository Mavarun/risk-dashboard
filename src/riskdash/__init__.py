"""riskdash: small portfolio risk metrics library (research slice)."""

from riskdash.metrics import (
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from riskdash.rolling_beta import rolling_beta
from riskdash.costs import apply_costs, equity_curve, net_of_cost_returns

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "hit_rate",
    "rolling_beta",
    "apply_costs",
    "equity_curve",
    "net_of_cost_returns",
]

__version__ = "0.1.0"

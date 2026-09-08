"""Transaction-cost helpers for gross vs net-of-cost equity diagnostics.

Costs are a research proxy (flat bps on turnover), not exchange fees or live PnL.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, list]


def apply_costs(
    returns: ArrayLike,
    positions: ArrayLike | None = None,
    cost_bps: float = 5.0,
) -> np.ndarray:
    """Subtract turnover costs from a gross return series.

    If positions is None, treats each period as a full turnover of 1.0
    (conservative constant rebalance). Otherwise cost = cost_bps/1e4 * |dpos|.
    """
    r = np.asarray(returns, dtype=float).ravel()
    if positions is None:
        turnover = np.ones_like(r)
    else:
        pos = np.asarray(positions, dtype=float).ravel()
        if pos.size != r.size:
            raise ValueError("positions length must match returns")
        prev = np.concatenate([[0.0], pos[:-1]])
        turnover = np.abs(pos - prev)
    cost = (cost_bps / 1e4) * turnover
    return r - cost


def net_of_cost_returns(
    returns: ArrayLike,
    positions: ArrayLike | None = None,
    cost_bps: float = 5.0,
) -> np.ndarray:
    """Alias for apply_costs — net returns after flat bps costs."""
    return apply_costs(returns, positions=positions, cost_bps=cost_bps)


def equity_curve(returns: ArrayLike, start: float = 1.0) -> np.ndarray:
    """Cumulative product equity path from a return series."""
    r = np.asarray(returns, dtype=float).ravel()
    if r.size == 0:
        return np.asarray([start], dtype=float)
    return start * np.cumprod(1.0 + r)


def cost_drag_summary(
    gross_returns: ArrayLike,
    positions: ArrayLike | None = None,
    cost_bps: float = 5.0,
) -> dict[str, float]:
    """Compare max drawdown and terminal equity: gross vs net-of-cost."""
    from riskdash.metrics import max_drawdown

    gross = np.asarray(gross_returns, dtype=float).ravel()
    net = apply_costs(gross, positions=positions, cost_bps=cost_bps)
    eq_g = equity_curve(gross)
    eq_n = equity_curve(net)
    return {
        "cost_bps": float(cost_bps),
        "gross_max_drawdown": max_drawdown(gross),
        "net_max_drawdown": max_drawdown(net),
        "drawdown_widening": float(max_drawdown(net) - max_drawdown(gross)),
        "gross_terminal_equity": float(eq_g[-1]),
        "net_terminal_equity": float(eq_n[-1]),
        "terminal_equity_gap": float(eq_g[-1] - eq_n[-1]),
    }

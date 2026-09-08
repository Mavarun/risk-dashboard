"""Report builder: assemble a metrics table for the research slice."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from riskdash.costs import apply_costs, cost_drag_summary
from riskdash.metrics import summarize_returns
from riskdash.rolling_beta import rolling_beta_last


def _strategy_returns(portfolio: np.ndarray, positions: np.ndarray | None) -> np.ndarray:
    """Gross strategy returns = position * asset return (research overlay)."""
    if positions is None:
        return portfolio
    return positions * portfolio


def build_report(
    panel: pd.DataFrame,
    window: int = 60,
    cost_bps: float = 5.0,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Compute gross/net risk metrics and trailing beta from a return panel."""
    port = panel["portfolio"].to_numpy(dtype=float)
    mkt = panel["market"].to_numpy(dtype=float)
    pos = (
        panel["position"].to_numpy(dtype=float)
        if "position" in panel.columns
        else None
    )

    gross_r = _strategy_returns(port, pos)
    net_r = apply_costs(gross_r, positions=pos, cost_bps=cost_bps)

    gross = summarize_returns(
        gross_r, periods_per_year=periods_per_year, label="gross_portfolio"
    )
    net = summarize_returns(
        net_r, periods_per_year=periods_per_year, label="net_portfolio"
    )
    mkt_sum = summarize_returns(
        mkt, periods_per_year=periods_per_year, label="market"
    )
    beta = rolling_beta_last(port, mkt, window=window)
    drag = cost_drag_summary(gross_r, positions=pos, cost_bps=cost_bps)

    return {
        "n_obs": int(len(panel)),
        "rolling_window": window,
        "cost_bps": cost_bps,
        "trailing_beta": beta,
        "gross": gross,
        "net": net,
        "market": mkt_sum,
        "cost_drag": drag,
    }


def metrics_table(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten report into a small comparison DataFrame."""
    rows = []
    for key in ("gross", "net", "market"):
        s = report[key]
        rows.append(
            {
                "series": s["label"],
                "n": int(s["n"]),
                "mean": s["mean"],
                "std": s["std"],
                "sharpe": s["sharpe"],
                "sortino": s["sortino"],
                "max_dd": s["max_drawdown"],
                "hit_rate": s["hit_rate"],
            }
        )
    return pd.DataFrame(rows)


def format_metrics_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Pretty-print a metrics table as markdown-ish aligned text."""
    cols = list(df.columns)
    header = " | ".join(f"{c:>14}" for c in cols)
    sep = "-+-".join("-" * 14 for _ in cols)
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{float_fmt.format(v):>14}")
            else:
                cells.append(f"{str(v):>14}")
        lines.append(" | ".join(cells))
    return "\n".join(lines)

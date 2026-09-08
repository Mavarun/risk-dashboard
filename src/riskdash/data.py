"""Synthetic factor/market return generators; optional yfinance fallback.

Default path is fully synthetic and reproducible (seeded). Live market data
via yfinance is optional and never required for tests or the demo script.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def make_synthetic_factor_portfolio(
    n: int = 756,
    seed: int = 42,
    market_vol: float = 0.012,
    idio_vol: float = 0.008,
    true_beta: float = 1.15,
    alpha_daily: float = 0.00015,
    start: str = "2020-01-02",
) -> pd.DataFrame:
    """Generate aligned market and factor-portfolio daily returns.

    Portfolio return = alpha + true_beta * market + idiosyncratic noise.
    Positions follow a lagged 5-day market-sign rule (research toy only) so
    turnover is nonzero and cost drag is measurable — not a trading claim.
    """
    rng = np.random.default_rng(seed)
    market = rng.normal(loc=0.0003, scale=market_vol, size=n)
    idio = rng.normal(loc=0.0, scale=idio_vol, size=n)
    portfolio = alpha_daily + true_beta * market + idio

    position = np.ones(n, dtype=float)
    for t in range(n):
        if t < 5:
            position[t] = 1.0
        else:
            trail = market[t - 5 : t].sum()
            position[t] = 1.0 if trail >= 0 else 0.0

    dates = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "market": market,
            "portfolio": portfolio,
            "position": position,
        }
    )


def load_returns(
    source: str = "synthetic",
    n: int = 756,
    seed: int = 42,
    ticker: str = "SPY",
    benchmark: str = "SPY",
    start: str = "2022-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load return panel from synthetic generator or optional yfinance.

    Prefer synthetic. If source='yfinance' and the package/network fails,
    falls back to synthetic with a printed note.
    """
    if source == "synthetic":
        return make_synthetic_factor_portfolio(n=n, seed=seed)

    if source != "yfinance":
        raise ValueError("source must be 'synthetic' or 'yfinance'")

    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        print("yfinance not installed; falling back to synthetic returns")
        return make_synthetic_factor_portfolio(n=n, seed=seed)

    try:
        asset = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        mkt = yf.download(benchmark, start=start, end=end, progress=False, auto_adjust=True)
        if asset.empty or mkt.empty:
            raise RuntimeError("empty download")
        a = asset["Close"].pct_change().dropna()
        m = mkt["Close"].pct_change().dropna()
        if isinstance(a, pd.DataFrame):
            a = a.iloc[:, 0]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[:, 0]
        panel = pd.concat([m.rename("market"), a.rename("portfolio")], axis=1).dropna()
        panel = panel.reset_index()
        panel.columns = ["date", "market", "portfolio"]
        panel["position"] = 1.0
        print(
            f"Loaded yfinance {ticker} vs {benchmark} "
            f"({len(panel)} rows). Research use only — not live PnL."
        )
        return panel
    except Exception as exc:  # noqa: BLE001
        print(f"yfinance fetch failed ({exc}); falling back to synthetic returns")
        return make_synthetic_factor_portfolio(n=n, seed=seed)

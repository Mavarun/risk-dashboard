# risk-dashboard

Research slice: a small library that computes **Sharpe**, **Sortino**, **max drawdown**,
**rolling beta**, and **hit rate** from a return series **without look-ahead**, plus
gross vs net-of-cost equity diagnostics. Prefer synthetic data. **Not live PnL. Not alpha.**

## Hypothesis

1. A small risk library can compute Sharpe, Sortino, max drawdown, rolling beta, and hit rate from a return series without look-ahead.
2. On a synthetic factor portfolio vs a market series, rolling beta and drawdown diagnostics match hand-checked formulas within tolerance.
3. Reporting net-of-cost equity vs gross shows costs widen drawdowns — measure it honestly; not live PnL.

## Method

- **Data (default):** seeded synthetic daily market + factor portfolio
  (`n=756`, `seed=42`, planted `true_beta=1.15`). Portfolio =
  `alpha + beta * market + idio`. Positions follow a **lagged** 5-day market-sign
  rule (past bars only) so turnover is nonzero for cost demos.
- **Optional:** `--source yfinance` if installed/networked; otherwise falls back
  to synthetic with a clear note. No bundled market CSV required.
- **Metrics:** annualized Sharpe / Sortino (`periods_per_year=252`), max drawdown
  on cumulative equity, hit rate (fraction of strict positive periods), trailing
  OLS rolling beta (window=60, no look-ahead).
- **Costs:** flat `cost_bps` on `|Δposition|` each bar. Net equity vs gross —
  research proxy only.

## Metrics (local synthetic run)

Command: `python scripts/run_risk_slice.py --source synthetic`

| series | n | mean | std | sharpe | sortino | max_dd | hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gross_portfolio | 756 | -0.0005 | 0.0114 | -0.6535 | -0.9104 | -0.4198 | 0.2421 |
| net_portfolio | 756 | -0.0006 | 0.0114 | -0.8060 | -1.1152 | -0.4626 | 0.2407 |
| market | 756 | -0.0002 | 0.0119 | -0.2430 | -0.3402 | -0.3290 | 0.4960 |

- **trailing_beta** (window=60): **1.1539** (planted 1.15 — within tolerance).
- **cost drag** (`cost_bps=5`): gross max DD **-0.4198** → net **-0.4626**
  (widening **-0.0427**); terminal equity gap **0.0529**.

**Reading:** Hypothesis (2) holds — trailing beta recovers the planted factor
loading. Hypothesis (3) holds on this path — the same turnover that funds the
toy overlay **widens drawdowns** once flat bps costs are applied. Negative
Sharpes here are expected on this seeded path with a sparse long/flat rule;
they are diagnostics, not a performance claim.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt
pytest
python scripts/run_risk_slice.py
python scripts/run_risk_slice.py --json --out artifacts/metrics.json
# optional (requires yfinance + network):
# python scripts/run_risk_slice.py --source yfinance --ticker SPY --benchmark SPY
```

## Package layout

```
src/riskdash/   metrics, rolling_beta, costs, data, report
scripts/        run_risk_slice.py
tests/          known formulas, no look-ahead windows, cost drag
```

## Assumptions / limits

- Synthetic returns are **not** market history; beta strength is planted for a
  reproducible check.
- Flat bps costs ignore spreads that vary with size, latency, and adverse selection.
- Hit rate and Sharpe on a long/flat overlay are not comparable to always-invested
  benchmarks without care.
- Optional yfinance path is convenience only — still research diagnostics.
- **Never** interpret these numbers as live PnL, capacity, or alpha.

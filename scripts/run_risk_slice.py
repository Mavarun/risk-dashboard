#!/usr/bin/env python3
"""Run the risk-dashboard research slice and print a metrics table.

Prefer synthetic data (default). Never claims live PnL or alpha.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from riskdash.data import load_returns  # noqa: E402
from riskdash.report import build_report, format_metrics_table, metrics_table  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Risk dashboard research slice")
    p.add_argument("--source", choices=["synthetic", "yfinance"], default="synthetic")
    p.add_argument("--n", type=int, default=756, help="synthetic path length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=60, help="rolling beta window")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--json", action="store_true", help="also emit JSON blob")
    p.add_argument("--out", type=str, default="", help="optional JSON output path")
    args = p.parse_args()

    panel = load_returns(
        source=args.source,
        n=args.n,
        seed=args.seed,
        ticker=args.ticker,
        benchmark=args.benchmark,
    )
    report = build_report(panel, window=args.window, cost_bps=args.cost_bps)
    table = metrics_table(report)

    print("risk-dashboard research slice (not live PnL)")
    print(
        f"source={args.source}  n={report['n_obs']}  "
        f"window={report['rolling_window']}  cost_bps={report['cost_bps']}"
    )
    print(f"trailing_beta={report['trailing_beta']:.4f}")
    print()
    print(format_metrics_table(table))
    print()
    drag = report["cost_drag"]
    print(
        f"cost drag: gross_max_dd={drag['gross_max_drawdown']:.4f}  "
        f"net_max_dd={drag['net_max_drawdown']:.4f}  "
        f"widening={drag['drawdown_widening']:.4f}  "
        f"terminal_gap={drag['terminal_equity_gap']:.4f}"
    )

    payload = {
        "source": args.source,
        "trailing_beta": report["trailing_beta"],
        "cost_drag": report["cost_drag"],
        "table": table.to_dict(orient="records"),
        "disclaimer": "Synthetic/research diagnostics only. Not live PnL or alpha.",
    }
    if args.json or args.out:
        text = json.dumps(payload, indent=2, default=float)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text)
            print(f"\nwrote {out_path}")
        if args.json:
            print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproduces the claim in the README: a strategy built from pure random noise can
look convincing, and the multiple-testing check catches it.

    python examples/noise_test.py

No arguments, no data files, no network. It generates 200 random return series whose
true edge is exactly zero, keeps the luckiest one, and runs it through the validator.

The point is not that the best of 200 coin flips looks good - that is obvious once
stated. The point is that it looks good *by every conventional measure*, including
the one most people rely on, and only the multiple-testing correction rejects it.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_validator import check_deflated_sharpe, periods_per_year  # noqa: E402

TRIALS = 200
DAYS = 1260          # about five years of trading days
DAILY_VOL = 0.01
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)

    best_sharpe, best_returns = -np.inf, None
    for _ in range(TRIALS):
        # Mean of exactly zero. There is no edge here to find.
        r = rng.normal(0.0, DAILY_VOL, DAYS)
        sharpe = r.mean() / r.std(ddof=1)
        if sharpe > best_sharpe:
            best_sharpe, best_returns = sharpe, r

    equity = pd.Series(
        100_000 * np.cumprod(1 + best_returns),
        index=pd.bdate_range("2019-01-01", periods=DAYS),
    )

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    annual_sharpe = best_sharpe * math.sqrt(periods_per_year(equity.index))
    peak = equity.cummax()
    max_dd = abs(((equity - peak) / peak * 100).min())

    print(f"Generated {TRIALS} random strategies. True edge of every one: zero.\n")
    print("The luckiest of them looks like this:\n")
    print(f"    Total return      {total_return:+.1f}%")
    print(f"    Sharpe ratio      {annual_sharpe:.2f}")
    print(f"    Max drawdown      {max_dd:.1f}%")
    print(f"    Final balance     {equity.iloc[-1]:,.0f} from 100,000\n")
    print("Numbers most people would be happy with. Now the check:\n")

    result = check_deflated_sharpe(equity)
    print(f"    {result.verdict}: {result.headline}")
    print(f"    Probability the edge is real, if only one version was tried: "
          f"{result.evidence['psr_single_trial']:.1%}\n")

    breakeven = result.evidence["breakeven_trials"]
    print(f"Read that second line carefully. Judged as a single attempt, this looks")
    print(f"{result.evidence['psr_single_trial']:.1%} certain to be real. It is not real at all.\n")
    print(f"The check says the result only holds up if fewer than about {breakeven} versions")
    print(f"were tried. {TRIALS} were tried. That gap is the whole point.\n")
    print("If you adjusted settings until your backtest looked good, you ran this")
    print("experiment too - you just did not write down how many attempts it took.")


if __name__ == "__main__":
    main()

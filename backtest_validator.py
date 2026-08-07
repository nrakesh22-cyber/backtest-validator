#!/usr/bin/env python3
"""
backtest_validator — audit a trading backtest for the flaws that silently inflate it.

Single file. Requires only pandas and numpy. Your data never leaves your machine.

    python backtest_validator.py --equity curve.csv

    python backtest_validator.py --equity curve.csv --trades trades.csv \
        --benchmark-csv spy.csv --look-ahead no --plateau no

INPUT FORMATS (CSV)
    --equity        date,equity                    [required]
    --trades        entry_date,exit_date,pnl[,ticker]   [optional]
    --positions     date,in_market                 [optional]
    --benchmark-csv date,close                     [optional]

If --positions is omitted but --trades is supplied, market exposure is inferred
from entry/exit dates. If --benchmark-csv is omitted, the benchmark is fetched with
yfinance when installed (on *your* machine, under your own terms of use) and the
benchmark-dependent checks are skipped when it is not.

DESIGN PRINCIPLE
    Compute the breakeven, do not ask the question. A check that quotes your own
    answer back at you is a form, not a diagnosis. Nine of the eleven checks below
    are computed from your data. The two that are not are marked as such, and are
    ignored entirely in scoring unless you answer them.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PASS, WARN, FAIL, UNKNOWN = "PASS", "WARN", "FAIL", "UNKNOWN"

# How much each check counts toward the score out of 10.
WEIGHTS = {
    "benchmark": 2.0,
    "exposure_matched": 2.0,
    "deflated_sharpe": 2.0,
    "performance_decay": 2.0,
    "look_ahead": 2.0,
    "breakeven_cost": 1.5,
    "survivorship_exposure": 1.5,
    "out_of_sample": 1.5,
    "return_concentration": 1.0,
    "sample_size": 1.0,
    "drawdown_realism": 1.0,
    "parameter_sensitivity": 1.0,
}

EULER_GAMMA = 0.5772156649015329

# Fraction of US-listed companies leaving the listed universe each year through
# bankruptcy, delisting or acquisition. ~4-5%/yr is the commonly cited range and
# matches S&P 500 constituent turnover of roughly 20-25 names a year.
ANNUAL_ATTRITION = 0.045


# --------------------------------------------------------------------- statistics
def norm_cdf(x: float) -> float:
    """Standard normal CDF. Uses math.erf so scipy is not a dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.2e-9).

    Needed for the expected-maximum-Sharpe term in the deflated Sharpe ratio.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                 / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    q = p - 0.5
    r = q * q
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


def returns_of(s: pd.Series) -> pd.Series:
    """Simple period returns. Computed directly rather than via pct_change() so the
    result does not depend on which pandas version the user happens to have."""
    return (s / s.shift(1) - 1.0).dropna()


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer sampling frequency so annualisation is not hardcoded to daily. A weekly
    equity curve annualised at 252 would overstate Sharpe by about 2.2x.

    Counts observations per elapsed year rather than taking the median gap between
    them. A curve with one point per closed trade - which is what a TradingView
    export produces - has clustered exits, and the median gap there is far shorter
    than the average one. Using the median inflated the annualised Sharpe by roughly
    50% on exactly the input this tool is now meant to read.
    """
    if len(index) < 3:
        return 252.0
    years = (index[-1] - index[0]).days / 365.25
    if years <= 0:
        return 252.0
    return min(252.0, len(index) / years)


def max_drawdown_pct(s: pd.Series) -> float:
    peak = s.cummax()
    return float(abs(((s - peak) / peak * 100).min()))


# ------------------------------------------------------------------- result type
@dataclass
class CheckResult:
    key: str
    name: str
    verdict: str
    headline: str
    detail: str = ""
    evidence: dict = field(default_factory=dict)
    remedy: str = ""
    inferred: bool = True  # False => answered by the user, not computed

    @property
    def weight(self) -> float:
        return WEIGHTS.get(self.key, 1.0)


def _unknown(key: str, name: str, why: str) -> CheckResult:
    return CheckResult(key=key, name=name, verdict=UNKNOWN, headline=why)


# ------------------------------------------------------------------ curve checks
def check_benchmark(equity: pd.Series, benchmark: pd.Series) -> CheckResult:
    """The first question any strategy must answer: did it beat simply buying the
    benchmark? In this project's own history, this was asked far too late."""
    s_ret = float((equity.iloc[-1] / equity.iloc[0] - 1) * 100)
    b_ret = float((benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100)
    gap = s_ret - b_ret

    if gap > 0:
        verdict, head = PASS, f"Beat buy-and-hold by {gap:+.1f} pts"
    elif gap > -10:
        verdict, head = WARN, f"Trails buy-and-hold by {abs(gap):.1f} pts"
    else:
        verdict, head = FAIL, f"Badly trails buy-and-hold by {abs(gap):.1f} pts"

    return CheckResult(
        key="benchmark", name="Beats buy-and-hold", verdict=verdict, headline=head,
        detail=f"Strategy {s_ret:+.1f}% vs benchmark {b_ret:+.1f}% over the same window.",
        evidence={"strategy_return_pct": round(s_ret, 2), "benchmark_return_pct": round(b_ret, 2)},
        remedy="If the strategy trails a passive benchmark, complexity is not being paid for. "
               "No amount of parameter tuning fixes a negative gap.",
    )


def check_exposure_matched(equity: pd.Series, benchmark: pd.Series,
                           in_market: pd.Series) -> CheckResult:
    """The decisive test for stock-selection skill. Compares the strategy against
    holding the benchmark ONLY on the days the strategy held a position. A strategy
    that beats buy-and-hold purely by sitting in cash during bad periods will fail
    here — that is market timing, not selection."""
    bm_ret = returns_of(benchmark)
    flags = in_market.reindex(bm_ret.index).fillna(False).astype(bool)
    matched = float(((1 + bm_ret.where(flags, 0.0)).prod() - 1) * 100)
    s_ret = float((equity.iloc[-1] / equity.iloc[0] - 1) * 100)
    exposure = float(flags.mean() * 100)
    gap = s_ret - matched

    if gap > 0:
        verdict, head = PASS, f"Adds {gap:+.1f} pts over benchmark-when-invested"
    elif gap > -10:
        verdict, head = WARN, f"Trails benchmark-when-invested by {abs(gap):.1f} pts"
    else:
        verdict, head = FAIL, f"No selection skill: trails by {abs(gap):.1f} pts"

    return CheckResult(
        key="exposure_matched", name="Selection skill (exposure-matched)", verdict=verdict,
        headline=head,
        detail=f"Time in market {exposure:.1f}%. Holding the benchmark on exactly those days "
               f"returned {matched:+.1f}% vs the strategy's {s_ret:+.1f}%.",
        evidence={"exposure_pct": round(exposure, 1),
                  "matched_benchmark_pct": round(matched, 2),
                  "strategy_return_pct": round(s_ret, 2)},
        remedy="If the strategy loses here, its returns come from being out of the market at "
               "the right times, not from picking the right assets. A simple timing rule would "
               "capture the same benefit with none of the complexity. NOTE: a deliberately "
               "low-beta or defensive strategy can fail this check legitimately — read it "
               "alongside the drawdown figures before concluding there is no skill.",
    )


def check_drawdown_realism(equity: pd.Series, benchmark: pd.Series) -> CheckResult:
    """A strategy with far lower drawdown than its benchmark is usually holding cash,
    not managing risk. Worth surfacing so the ratio is not mistaken for skill."""
    s_dd, b_dd = max_drawdown_pct(equity), max_drawdown_pct(benchmark)
    s_ret = float((equity.iloc[-1] / equity.iloc[0] - 1) * 100)
    b_ret = float((benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100)
    s_ratio = s_ret / s_dd if s_dd else float("nan")
    b_ratio = b_ret / b_dd if b_dd else float("nan")

    verdict = PASS if s_ratio > b_ratio else WARN
    return CheckResult(
        key="drawdown_realism", name="Return per unit of drawdown", verdict=verdict,
        headline=f"Strategy {s_ratio:.2f} vs benchmark {b_ratio:.2f}",
        detail=f"Max drawdown {s_dd:.1f}% vs benchmark {b_dd:.1f}%.",
        evidence={"strategy_maxdd_pct": round(s_dd, 1), "benchmark_maxdd_pct": round(b_dd, 1)},
        remedy="A favourable ratio driven by low time-in-market is an artefact of absence, not "
               "risk management. Read this alongside the exposure-matched check.",
    )


def check_deflated_sharpe(equity: pd.Series, dispersion_annual: float = 0.5,
                          confidence: float = 0.95) -> CheckResult:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado).

    Two numbers, both computed, neither requiring an honest confession:

      1. PSR — the probability the true Sharpe exceeds zero, corrected for the
         non-normality (skew and fat tails) that makes naive Sharpe optimistic.
         This assumes a single trial and needs no inputs beyond the equity curve.

      2. Breakeven trial count — how many strategy variants you could have tested
         before this Sharpe becomes what the *best of a random search* would produce
         anyway. Nobody reports their true trial count honestly, so the check gives
         the threshold and lets the reader confront their own memory of it.
    """
    r = returns_of(equity)
    T = len(r)
    if T < 30:
        return _unknown("deflated_sharpe", "Deflated Sharpe (multiple testing)",
                        f"Only {T} return observations — too few")

    sd = float(r.std(ddof=1))
    if sd == 0:
        return _unknown("deflated_sharpe", "Deflated Sharpe (multiple testing)",
                        "Equity curve has zero variance")

    sr = float(r.mean()) / sd                       # per-period Sharpe
    ppy = periods_per_year(equity.index)
    sr_annual = sr * math.sqrt(ppy)
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0                # pandas gives excess kurtosis

    # Denominator of the PSR statistic: the standard error of the Sharpe estimate
    # once skew and kurtosis are accounted for.
    var_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if var_term <= 0:
        return _unknown("deflated_sharpe", "Deflated Sharpe (multiple testing)",
                        "Return distribution too extreme to estimate")
    denom = math.sqrt(var_term)

    def psr(threshold_sr: float) -> float:
        return norm_cdf(((sr - threshold_sr) * math.sqrt(T - 1)) / denom)

    psr0 = psr(0.0)

    # Expected maximum Sharpe across N independent trials, given how much the Sharpe
    # varies from one trial to the next.
    disp = dispersion_annual / math.sqrt(ppy)       # annual -> per-period

    def expected_max_sr(n: int) -> float:
        if n <= 1:
            return 0.0
        return disp * ((1 - EULER_GAMMA) * norm_ppf(1 - 1.0 / n)
                       + EULER_GAMMA * norm_ppf(1 - 1.0 / (n * math.e)))

    breakeven_n = 0
    if psr0 >= confidence:
        for n in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 250, 500, 1000, 5000):
            if psr(expected_max_sr(n)) >= confidence:
                breakeven_n = n
            else:
                break

    ev = {"sharpe_annualised": round(sr_annual, 2), "psr_single_trial": round(psr0, 3),
          "skew": round(skew, 2), "kurtosis": round(kurt, 2), "observations": T,
          "breakeven_trials": breakeven_n}

    if psr0 < confidence:
        return CheckResult(
            key="deflated_sharpe", name="Deflated Sharpe (multiple testing)", verdict=FAIL,
            headline=f"Sharpe {sr_annual:.2f} not significant even at 1 trial "
                     f"(PSR {psr0:.2%} < {confidence:.0%})",
            detail=f"Adjusted for skew {skew:.2f} and kurtosis {kurt:.2f} over {T} observations, "
                   f"the probability the true Sharpe exceeds zero is {psr0:.0%} — below the "
                   f"{confidence:.0%} bar before any correction for how many variants you tried.",
            evidence=ev,
            remedy="This result does not clear the lowest possible hurdle. Fat tails and negative "
                   "skew make the headline Sharpe flattering. Lengthen the sample or abandon it.",
        )

    if breakeven_n >= 50:
        verdict = PASS
        head = f"Sharpe {sr_annual:.2f} survives up to ~{breakeven_n} trials"
    elif breakeven_n >= 10:
        verdict = WARN
        head = f"Sharpe {sr_annual:.2f} survives only ~{breakeven_n} trials"
    else:
        verdict = FAIL
        head = f"Sharpe {sr_annual:.2f} survives only ~{breakeven_n} trials"

    return CheckResult(
        key="deflated_sharpe", name="Deflated Sharpe (multiple testing)", verdict=verdict,
        headline=head,
        detail=f"PSR at a single trial is {psr0:.0%}. But if you tested more than about "
               f"{breakeven_n} variants — parameter settings, universes, entry rules, anything "
               f"you ran and discarded — this Sharpe is what the best of that search would have "
               f"produced from noise alone. Assumed dispersion of trial Sharpes: "
               f"{dispersion_annual:.2f} annualised.",
        evidence=ev,
        remedy=f"Count honestly. Every abandoned variant is a trial, including the ones you "
               f"stopped early because they looked bad. If your true count exceeds {breakeven_n}, "
               f"treat this Sharpe as unproven and re-test the frozen rules on data you have "
               f"never examined.",
    )


def check_performance_decay(equity: pd.Series, benchmark: pd.Series | None) -> CheckResult:
    """Split the record in half and compare. Parameters fitted to the visible data
    tend to work in the first half and fade in the second.

    Measured as excess return over the benchmark in each half, not raw return, so a
    strategy is not condemned merely for having lived through a bad market in its
    later years. Without a benchmark the check falls back to raw return and says so.
    """
    n = len(equity)
    if n < 60:
        return _unknown("performance_decay", "Performance decay (first half vs second)",
                        f"Only {n} observations — too short to split")

    mid = n // 2
    h1, h2 = equity.iloc[:mid + 1], equity.iloc[mid:]

    def growth(s):
        return float((s.iloc[-1] / s.iloc[0] - 1) * 100)

    s1, s2 = growth(h1), growth(h2)
    if benchmark is not None:
        b1, b2 = growth(benchmark.iloc[:mid + 1]), growth(benchmark.iloc[mid:])
        e1, e2 = s1 - b1, s2 - b2
        basis = "excess return over benchmark"
    else:
        e1, e2 = s1, s2
        basis = "raw return (no benchmark supplied)"

    delta = e2 - e1
    split_date = equity.index[mid].date()

    if e1 > 0 and e2 < 0:
        verdict = FAIL
        head = f"Edge reverses after {split_date}: {e1:+.1f} -> {e2:+.1f} pts"
    elif delta < -20:
        verdict = FAIL
        head = f"Edge decays sharply: {e1:+.1f} -> {e2:+.1f} pts"
    elif delta < 0:
        verdict = WARN
        head = f"Edge weaker in second half: {e1:+.1f} -> {e2:+.1f} pts"
    else:
        verdict = PASS
        head = f"Holds up in second half: {e1:+.1f} -> {e2:+.1f} pts"

    return CheckResult(
        key="performance_decay", name="Performance decay (first half vs second)",
        verdict=verdict, headline=head,
        detail=f"Split at {split_date}, measured as {basis}. First half {e1:+.1f} pts, "
               f"second half {e2:+.1f} pts.",
        evidence={"first_half_excess_pct": round(e1, 2), "second_half_excess_pct": round(e2, 2),
                  "split_date": str(split_date), "vs_benchmark": benchmark is not None},
        remedy="Decay across halves is the signature of parameters fitted to the data you could "
               "see. This is weaker evidence than a true out-of-sample test — it cannot tell "
               "overfitting apart from an edge that genuinely stopped working — but a clean "
               "reversal is rarely innocent.",
    )


# ------------------------------------------------------------------ trade checks
def check_sample_size(trades: pd.DataFrame, years: float) -> CheckResult:
    n = len(trades)
    per_year = n / years if years else 0.0
    if n >= 200:
        verdict, head = PASS, f"{n} trades over {years:.1f}y — adequate"
    elif n >= 50:
        verdict, head = WARN, f"Only {n} trades over {years:.1f}y — thin"
    else:
        verdict, head = FAIL, f"Only {n} trades — not statistically meaningful"

    return CheckResult(
        key="sample_size", name="Statistical sample size", verdict=verdict, headline=head,
        detail=f"About {per_year:.1f} trades per year. With few trades, a handful of lucky "
               f"outcomes can account for the entire result.",
        evidence={"trades": n, "trades_per_year": round(per_year, 1)},
        remedy="Below ~200 trades, treat the result as a hypothesis rather than evidence. "
               "Test on more assets or a longer window.",
    )


def check_return_concentration(trades: pd.DataFrame) -> CheckResult:
    """If a tiny number of trades produce all the profit, the strategy is a lottery
    ticket wearing a system's clothes."""
    if "pnl" not in trades.columns or trades.empty:
        return _unknown("return_concentration", "Return concentration", "No pnl column supplied")

    pnl = trades["pnl"].astype(float).sort_values(ascending=False)
    total = float(pnl.sum())
    if total <= 0:
        return CheckResult(key="return_concentration", name="Return concentration",
                           verdict=FAIL, headline="Total P&L is not positive",
                           evidence={"total_pnl": round(total, 2)})

    k = max(1, int(len(pnl) * 0.05))
    share = float(pnl.head(k).sum()) / total * 100
    without_top = total - float(pnl.head(k).sum())

    if share < 50:
        verdict, head = PASS, f"Top 5% of trades = {share:.0f}% of profit"
    elif share < 90:
        verdict, head = WARN, f"Top 5% of trades = {share:.0f}% of profit"
    else:
        verdict, head = FAIL, f"Top 5% of trades = {share:.0f}% of profit"

    return CheckResult(
        key="return_concentration", name="Return concentration", verdict=verdict, headline=head,
        detail=f"Removing the best {k} trade(s) leaves {without_top:,.0f} of {total:,.0f} total "
               f"P&L. Highly concentrated profit means the edge may not repeat.",
        evidence={"top5pct_share_of_profit": round(share, 1), "pnl_excluding_top": round(without_top, 2),
                  "top_trade_count": k, "total_trades": len(pnl)},
        remedy="Re-run excluding the best few trades. If the edge vanishes, it was never there.",
    )


def check_breakeven_cost(trades: pd.DataFrame, assumed_cost: float) -> CheckResult:
    """Replaces the old 'did you model costs? yes/no' question with a number.

    Computes the round-trip cost per trade at which the entire profit disappears,
    then compares it to what a retail round trip actually costs. A strategy whose
    edge dies at $6 a trade has no edge; asking whether commission was 'modelled'
    would never have surfaced that.
    """
    if "pnl" not in trades.columns or trades.empty:
        return _unknown("breakeven_cost", "Breakeven trading cost", "No pnl column supplied")

    pnl = trades["pnl"].astype(float)
    n = len(pnl)
    total = float(pnl.sum())
    if total <= 0:
        return CheckResult(key="breakeven_cost", name="Breakeven trading cost", verdict=FAIL,
                           headline="Strategy is unprofitable before any costs",
                           evidence={"total_pnl": round(total, 2)})

    breakeven = total / n
    median_pnl = float(pnl.median())
    surviving = total - n * assumed_cost
    detail = (f"Total P&L {total:,.0f} across {n} round trips. Charging {breakeven:,.2f} per "
              f"round trip reduces the result to exactly zero. Median trade: {median_pnl:,.2f}.")

    if "entry_date" in trades.columns and "exit_date" in trades.columns:
        try:
            hold = (pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])).dt.days
            detail += f" Median holding period {float(hold.median()):.0f} day(s)."
        except Exception:
            pass

    # The mean breakeven is itself an outlier statistic. If the median trade loses
    # money, a comfortable-looking mean is being carried by a handful of winners and
    # says nothing about what the typical trade can absorb in costs.
    if median_pnl <= 0:
        verdict = FAIL
        head = f"Typical trade loses {abs(median_pnl):,.2f} before any costs"
    elif median_pnl < assumed_cost:
        verdict = FAIL
        head = f"Median trade {median_pnl:,.2f} < assumed cost {assumed_cost:,.2f}"
    elif breakeven < 10:
        verdict = FAIL
        head = f"Edge dies at {breakeven:,.2f}/trade in costs"
    elif breakeven < 25:
        verdict = WARN
        head = f"Edge dies at {breakeven:,.2f}/trade in costs"
    else:
        verdict = PASS
        head = f"Survives up to {breakeven:,.2f}/trade in costs"

    if median_pnl <= 0 < breakeven:
        detail += (f" Note the gap: the average trade appears to clear {breakeven:,.2f} in costs, "
                   f"but the median trade is negative. The mean is being carried by outliers, so "
                   f"the average figure describes no trade you are actually likely to make.")

    return CheckResult(
        key="breakeven_cost", name="Breakeven trading cost", verdict=verdict, headline=head,
        detail=detail + f" At the assumed {assumed_cost:,.2f} per round trip, {surviving:,.0f} "
                        f"of the P&L survives.",
        evidence={"breakeven_cost_per_trade": round(breakeven, 2), "trades": n,
                  "median_trade_pnl": round(median_pnl, 2),
                  "assumed_cost_per_trade": assumed_cost, "surviving_pnl": round(surviving, 2)},
        remedy="A retail round trip is rarely under a few dollars once commission on both legs "
               "and half the bid-ask spread are counted, and is far worse on illiquid names or "
               "at the open. Set --assumed-cost to your own realistic figure and re-read this "
               "line. Note this assumes equal position sizing; if your sizing varies a lot, "
               "convert to basis points against your actual notional.",
    )


def check_survivorship_exposure(trades: pd.DataFrame, years: float) -> CheckResult:
    """Replaces 'did you use a point-in-time universe? yes/no' with an expected count.

    Cannot prove survivorship bias without point-in-time constituent data — nothing
    free can. What it can do is compute how many companies a universe of this size
    should have lost over this period, which turns a yes/no question the user will
    answer carelessly into a number they have to look at.
    """
    if "ticker" not in trades.columns or trades.empty:
        return _unknown("survivorship_exposure", "Survivorship exposure",
                        "No ticker column in trade log")

    universe = int(trades["ticker"].nunique())
    if universe < 2 or years <= 0:
        return _unknown("survivorship_exposure", "Survivorship exposure",
                        "Universe too small to estimate")

    expected_dead = universe * (1 - (1 - ANNUAL_ATTRITION) ** years)
    pct = expected_dead / universe * 100

    if expected_dead < 1:
        verdict = PASS
        head = f"Universe of {universe} over {years:.1f}y — attrition negligible"
    elif pct < 15:
        verdict = WARN
        head = f"~{expected_dead:.0f} of {universe} names should have died"
    else:
        verdict = FAIL
        head = f"~{expected_dead:.0f} of {universe} names should have died ({pct:.0f}%)"

    return CheckResult(
        key="survivorship_exposure", name="Survivorship exposure", verdict=verdict,
        headline=head,
        detail=f"Your trade log touches {universe} distinct tickers over {years:.1f} years. At a "
               f"{ANNUAL_ATTRITION:.1%} annual rate of bankruptcy, delisting and acquisition, a "
               f"point-in-time universe of that size would have contained roughly "
               f"{expected_dead:.0f} companies that no longer exist. If your ticker list came "
               f"from an index's *current* membership, every one of those is missing from your "
               f"test, and each was excluded precisely because it did badly.",
        evidence={"distinct_tickers": universe, "years": round(years, 1),
                  "expected_delistings": round(expected_dead, 1)},
        remedy="Check how many tickers in your list are delisted today. If the answer is zero, "
               "your universe is survivorship-biased and the published estimates of that bias "
               "run to 1.5-4% per year of fake alpha. Source point-in-time constituents, or "
               "subtract that from your result before believing it.",
    )


# ------------------------------------------- methodology (cannot be inferred yet)
def check_look_ahead(fundamentals_as_of_date: bool | None) -> CheckResult:
    if fundamentals_as_of_date is None:
        r = _unknown("look_ahead", "Look-ahead bias", "Not answered (--look-ahead yes/no)")
        r.inferred = False
        return r
    if fundamentals_as_of_date:
        return CheckResult(key="look_ahead", name="Look-ahead bias", verdict=PASS,
                           headline="Screening data is as-of the decision date", inferred=False)
    return CheckResult(
        key="look_ahead", name="Look-ahead bias", verdict=FAIL, inferred=False,
        headline="Screening uses data unavailable at the decision date",
        detail="Selecting historical trades using today's fundamentals means choosing companies "
               "already known to have stayed healthy.",
        evidence={"fundamentals_as_of_date": False},
        remedy="Either source point-in-time fundamentals, or drop the fundamental screen and "
               "re-test on liquidity alone. In the project this tool came from, removing it cut "
               "results by roughly half.",
    )


def check_out_of_sample(oos_type: str | None) -> CheckResult:
    """Splitting by ticker over the same window is NOT out-of-sample — both halves
    saw the identical market regimes."""
    if oos_type is None:
        r = _unknown("out_of_sample", "Out-of-sample validation",
                     "Not answered (--oos temporal|cross_sectional|none)")
        r.inferred = False
        return r
    o = oos_type.strip().lower()
    if o in ("temporal", "walk_forward", "walkforward"):
        return CheckResult(key="out_of_sample", name="Out-of-sample validation", verdict=PASS,
                           headline="Temporal / walk-forward split used", inferred=False)
    if o in ("cross_sectional", "ticker", "cross-sectional"):
        return CheckResult(
            key="out_of_sample", name="Out-of-sample validation", verdict=WARN, inferred=False,
            headline="Cross-sectional only (different assets, same period)",
            detail="Both samples experienced the same regimes, so this does not test whether the "
                   "edge survives a different market environment.",
            remedy="Add a temporal split: fit on the earlier period, test untouched on the later one.",
        )
    return CheckResult(
        key="out_of_sample", name="Out-of-sample validation", verdict=FAIL, inferred=False,
        headline="No out-of-sample testing",
        detail="Parameters chosen while viewing the whole dataset are fitted to it by construction.",
        remedy="Hold out a period entirely. Do not look at it until parameters are frozen.",
    )


def check_parameter_sensitivity(is_plateau: bool | None) -> CheckResult:
    if is_plateau is None:
        r = _unknown("parameter_sensitivity", "Parameter robustness",
                     "Not answered (--plateau yes/no)")
        r.inferred = False
        return r
    if is_plateau:
        return CheckResult(key="parameter_sensitivity", name="Parameter robustness", verdict=PASS,
                           headline="Performance sits on a broad plateau", inferred=False)
    return CheckResult(
        key="parameter_sensitivity", name="Parameter robustness", verdict=FAIL, inferred=False,
        headline="Performance is a lone spike at the chosen parameters",
        detail="A result that only works at one setting is fitted to noise.",
        remedy="Sweep each parameter across a wide range. A real effect degrades gracefully; "
               "an overfit one collapses either side of the chosen value.",
    )


# ------------------------------------------------------------------- scoring
_POINTS = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}
_ICON = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", UNKNOWN: " -- "}


MIN_COVERAGE = 0.40


def score(results: list[CheckResult]) -> tuple[float, str, float]:
    """Weighted score out of 10, ignoring unanswered checks so a partial
    questionnaire is not silently punished — it is reported separately.

    Returns coverage alongside the grade. A tool that exists to catch overclaiming
    must not itself grade a strategy 0/10 off two of twelve checks: below
    MIN_COVERAGE the score is withheld rather than dressed up as a verdict.
    """
    scored = [r for r in results if r.verdict != UNKNOWN]
    if not scored:
        return 0.0, "UNSCORED", 0.0
    total_w = sum(r.weight for r in scored)
    possible_w = sum(r.weight for r in results)
    coverage = total_w / possible_w if possible_w else 0.0
    got = sum(_POINTS[r.verdict] * r.weight for r in scored)
    out_of_10 = got / total_w * 10

    if coverage < MIN_COVERAGE:
        return round(out_of_10, 1), "INSUFFICIENT DATA", coverage

    if any(r.verdict == FAIL and r.weight >= 2.0 for r in scored):
        grade = "NOT TRUSTWORTHY"
    elif out_of_10 >= 8:
        grade = "ROBUST"
    elif out_of_10 >= 6:
        grade = "PLAUSIBLE"
    elif out_of_10 >= 4:
        grade = "WEAK"
    else:
        grade = "NOT TRUSTWORTHY"
    return round(out_of_10, 1), grade, coverage


# Typographic characters render as mojibake on a legacy Windows console, which makes
# a distributed tool look broken on the first screen a stranger sees. Report text is
# forced to ASCII rather than relying on the user's code page.
_ASCII_MAP = {"—": "-", "–": "-", "‘": "'", "’": "'",
              "“": '"', "”": '"', "…": "...", " ": " ",
              "→": "->", "×": "x", "≥": ">=", "≤": "<="}


def to_ascii(text: str) -> str:
    for uni, plain in _ASCII_MAP.items():
        text = text.replace(uni, plain)
    return text.encode("ascii", "replace").decode("ascii")


# --------------------------------------------------------------- plain language
#
# Every word a non-technical reader sees lives in this section, so the tone can be
# rewritten without touching a single statistic.
#
# Two rules that are not stylistic:
#   1. Describe the EVIDENCE, never the person who produced it. "This test used only
#      companies that still exist" is a fact. "The seller is hiding losses" is an
#      accusation about a third party, and not one this tool can support.
#   2. Never tell the reader to buy or not buy anything. State what the numbers show
#      and stop. The moment this recommends a decision it stops being a diagnostic.

PLAIN_GRADE = {
    "ROBUST": ("LOOKS SOLID", "The obvious ways a backtest can flatter itself were checked "
                              "and this one avoided them."),
    "PLAUSIBLE": ("PROBABLY OK", "Nothing serious found, but some questions remain open."),
    "WEAK": ("BE CAREFUL", "Several findings suggest these results would not repeat in "
                           "real trading."),
    "NOT TRUSTWORTHY": ("HIGH RISK", "These results are very unlikely to repeat in real "
                                     "trading."),
    "INSUFFICIENT DATA": ("NOT ENOUGH INFORMATION", "Too little was supplied to judge this "
                                                    "backtest. The findings below still stand "
                                                    "on their own."),
    "UNSCORED": ("NOT ENOUGH INFORMATION", "Nothing could be checked."),
}


def _plain(r: CheckResult) -> tuple[str, str] | None:
    """One finding, rewritten for a reader who does not know what a Sharpe ratio is."""
    e, k = r.evidence, r.key

    if k == "benchmark":
        return ("It made less money than simply buying the market",
                f"This strategy gained {e['strategy_return_pct']:.0f}% over the period. Buying "
                f"the benchmark and doing nothing else gained {e['benchmark_return_pct']:.0f}% "
                f"over exactly the same period.")

    if k == "exposure_matched":
        return ("It did not actually pick good trades",
                f"It held positions {e['exposure_pct']:.0f}% of the time. On those exact days, "
                f"simply holding the benchmark would have returned "
                f"{e['matched_benchmark_pct']:.0f}% instead of {e['strategy_return_pct']:.0f}%. "
                f"Whatever gains it made came from being out of the market at the right moments, "
                f"not from choosing the right things to buy.")

    if k == "deflated_sharpe":
        n = e.get("breakeven_trials", 0)
        if n == 0:
            return ("The returns are too erratic to prove anything",
                    "Once the size and unevenness of the ups and downs are accounted for, this "
                    "result cannot be told apart from luck even if only one version was ever "
                    "tested.")
        if n == 1:
            return ("The result may be luck rather than skill",
                    "If even one other version of this strategy was tried before this one was "
                    "settled on - a different setting, a different market, a different rule - "
                    "then a result this good is what you would expect from picking the best of "
                    "those tries by chance alone. Every version that was tested and thrown away "
                    "counts.")
        return ("The result may be luck rather than skill",
                f"If more than about {n} versions of this strategy were tried before this one "
                f"was settled on - different settings, different markets, different rules - then "
                f"a result this good is what you would expect from the best of those tries by "
                f"chance alone. Every version that was tested and thrown away counts.")

    if k == "performance_decay":
        a, b = e["first_half_excess_pct"], e["second_half_excess_pct"]

        def side(v):
            return f"ahead by {v:.0f}" if v >= 0 else f"behind by {abs(v):.0f}"

        head = ("It worked early on and stopped working later" if a >= 0
                else "It got worse as time went on")
        if e.get("vs_benchmark", True):
            lead = f"Measured against the market, it was {side(a)} points in the first half"
        else:
            lead = (f"No benchmark was supplied, so this compares it against itself: it gained "
                    f"{a:.0f}% in the first half")
        return (head,
                f"{lead} of the period and {side(b) if e.get('vs_benchmark', True) else f'{b:.0f}%'} "
                f"in the second, split at {e['split_date']}. A strategy tuned until it fits the "
                f"past often behaves exactly like this.")

    if k == "breakeven_cost":
        med = e.get("median_trade_pnl", 0)
        if med <= 0:
            return ("The typical trade lost money",
                    f"Half of all trades lost more than {abs(med):,.0f}. The total only looks "
                    f"positive because a small number of large winners covered the rest. Trading "
                    f"fees and the spread come off every single trade, winners and losers alike.")
        return ("The profit is thin compared to trading costs",
                f"Costs of about {e['breakeven_cost_per_trade']:,.0f} per completed trade would "
                f"wipe out the entire profit. Real costs include commission on both the buy and "
                f"the sell, plus the gap between the buy and sell price.")

    if k == "return_concentration":
        return ("Nearly all the profit came from a handful of trades",
                f"{e['top_trade_count']} trades out of {e['total_trades']} produced "
                f"{e['top5pct_share_of_profit']:.0f}% of the profit. Remove those few and almost "
                f"nothing is left. A result that depends on a few lucky outcomes is not a system.")

    if k == "sample_size":
        return ("There are not enough trades to trust the result",
                f"{e['trades']} trades in total, about {e['trades_per_year']:.0f} a year. That is "
                f"a small sample. With this few trades, plain good luck and a genuine edge look "
                f"exactly the same.")

    if k == "survivorship_exposure":
        return ("The test only used companies that still exist today",
                f"It covers {e['distinct_tickers']} companies over {e['years']:.0f} years. Over "
                f"that long, roughly {e['expected_delistings']:.0f} of them would normally have "
                f"gone bankrupt, been delisted or been bought out. If none of those appear, the "
                f"test quietly skipped the companies that did worst.")

    if k == "drawdown_realism":
        return ("Its smoother ride may just mean it sat in cash",
                f"Its worst fall was {e['strategy_maxdd_pct']:.0f}% against the market's "
                f"{e['benchmark_maxdd_pct']:.0f}%. Staying out of the market avoids losses; that "
                f"is not the same as managing risk well.")

    if k == "look_ahead":
        return ("It used information that did not exist at the time",
                "The trades were picked using company information published later than the date "
                "the trade was supposedly made. In real trading nobody has tomorrow's news.")

    if k == "out_of_sample":
        # WARN and FAIL mean different things here and must not share wording: one
        # tested on other assets over the same years, the other never held anything back.
        if r.verdict == WARN:
            return ("It was tested on other assets, but never on other years",
                    "The check used different companies over the same stretch of history. Those "
                    "companies all lived through the same booms and crashes, so this does not "
                    "show the strategy would survive a different kind of market.")
        return ("It was never tested on a period it had not already seen",
                "The settings were chosen while looking at all the data. A strategy tuned on the "
                "same history it is then measured against will always look good.")

    if k == "parameter_sensitivity":
        return ("It only works on one exact set of settings",
                "Nudge the settings slightly and the profit disappears. A real effect weakens "
                "gradually; one that collapses was fitted to noise.")

    return None


def render_plain(results: list[CheckResult], title: str = "Backtest Health Check") -> str:
    import textwrap

    pts, grade, coverage = score(results)
    label, meaning = PLAIN_GRADE.get(grade, (grade, ""))
    problems = [r for r in results if r.verdict == FAIL]
    minor = [r for r in results if r.verdict == WARN]
    skipped = [r for r in results if r.verdict == UNKNOWN]

    def para(text: str, indent: str = "     ") -> list[str]:
        return textwrap.wrap(text, width=70, initial_indent=indent, subsequent_indent=indent)

    L = ["=" * 76, f"  {title.upper()}", "=" * 76, ""]
    if problems:
        L.append(f"  RESULT:  {label}  -  {len(problems)} serious problem"
                 f"{'s' if len(problems) != 1 else ''} found")
    else:
        L.append(f"  RESULT:  {label}")
    L.append("")
    L += para(meaning, indent="  ")
    L.append("")

    n = 0
    if problems:
        L += ["-" * 76, "  WHAT IS WRONG", "-" * 76, ""]
        for r in problems:
            plain = _plain(r)
            if not plain:
                continue
            n += 1
            head, body = plain
            L.append(f"  {n:>2}. {head}")
            L += para(body, indent="      ")
            L.append("")

    if minor:
        L += ["-" * 76, "  WORTH KNOWING", "-" * 76, ""]
        for r in minor:
            plain = _plain(r)
            if not plain:
                continue
            n += 1
            head, body = plain
            L.append(f"  {n:>2}. {head}")
            L += para(body, indent="      ")
            L.append("")

    if skipped:
        L += ["-" * 76, "  COULD NOT BE CHECKED", "-" * 76, ""]
        for r in skipped:
            L.append(f"  - {r.name}: {r.headline}")
        L.append("")

    L += ["=" * 76]
    L += para("A backtest shows what a strategy would have done on past data. It is a "
              "claim, not a track record. This report checks whether that claim was "
              "measured carefully - it cannot tell you what will happen next, and it "
              "is not advice about what to buy or sell.", indent="  ")
    L += ["=" * 76]
    return to_ascii("\n".join(L))


def render(results: list[CheckResult], title: str = "Backtest Validation Report") -> str:
    pts, grade, coverage = score(results)
    unanswered = [r for r in results if r.verdict == UNKNOWN]
    fails = [r for r in results if r.verdict == FAIL]
    warns = [r for r in results if r.verdict == WARN]
    computed = sum(1 for r in results if r.inferred and r.verdict != UNKNOWN)
    assessed = len(results) - len(unanswered)

    L = ["=" * 78, f" {title}", "=" * 78, ""]
    if grade == "INSUFFICIENT DATA":
        L.append(f"  SCORE: withheld          VERDICT: {grade}")
        L.append(f"  Only {assessed} of {len(results)} checks could run ({coverage:.0%} of the "
                 f"weighted total).")
        L.append(f"  That is too little to grade a strategy on. Supply --trades and a benchmark")
        L.append(f"  for a real verdict. Individual findings below still stand on their own.")
    else:
        L.append(f"  SCORE: {pts}/10        VERDICT: {grade}")
        if fails:
            L.append(f"  {len(fails)} critical issue(s) found. See below.")
        L.append(f"  Based on {assessed} of {len(results)} checks ({coverage:.0%} of weighted total).")
    L.append(f"  {computed} of these checks were computed from your data, not asked of you.")
    L += ["", "-" * 78, f" {'CHECK':<42}{'RESULT':<8}{'FINDING'}", "-" * 78]
    for r in results:
        mark = "" if r.inferred else " *"
        L.append(f" {r.name + mark:<42}{_ICON[r.verdict]:<8}{r.headline}")
    L += ["-" * 78, " * = answered by you, not computed from your data", ""]

    if fails:
        L += ["CRITICAL ISSUES", ""]
        for r in fails:
            L.append(f"  [{r.name}] {r.headline}")
            if r.detail:
                L.append(f"      {r.detail}")
            if r.remedy:
                L.append(f"      -> {r.remedy}")
            L.append("")

    if warns:
        L += ["WARNINGS", ""]
        for r in warns:
            L.append(f"  [{r.name}] {r.headline}")
            if r.detail:
                L.append(f"      {r.detail}")
            L.append("")

    if unanswered:
        L.append("NOT ASSESSED — supply more data or answer these to complete the review")
        for r in unanswered:
            L.append(f"  - {r.name}: {r.headline}")
        L.append("")

    L += ["=" * 78,
          " A backtest is a hypothesis, not evidence. These checks exist because each",
          " one has, in practice, silently inflated a result until it was tested for.",
          "=" * 78]
    return to_ascii("\n".join(L))


# ----------------------------------------------------------------------- loading
def _tri(v: str | None) -> bool | None:
    if v is None:
        return None
    return v.strip().lower() in ("yes", "y", "true", "1")


# ------------------------------------------------------- TradingView trade list
#
# The Strategy Tester's "List of Trades" export is the one file a non-coding retail
# trader can actually produce, so it is worth parsing carefully.
#
# Structure: TWO rows per trade, paired by trade number - one entry, one exit. The
# P&L columns are filled on the exit row only. Column names vary by TradingView
# version and carry the instrument's currency ("Net P&L USDT", "Profit USD"), so
# every column is matched by shape rather than by exact name.
#
# One export yields all three inputs at once: the trade log, the equity curve (from
# cumulative P&L), and market exposure (from entry/exit dates).

def _tv_num(v) -> float:
    """Parse a TradingView number: currency symbols, thousands separators, unicode
    minus signs and parenthesised negatives all appear depending on locale."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan")
    s = str(v).strip()
    if not s or s.upper() in ("N/A", "NA", "-", "—"):
        return float("nan")
    s = s.replace("−", "-").replace("–", "-").replace(",", "").replace(" ", "")
    negative = s.startswith("(") and s.endswith(")")
    kept = [ch for ch in s if ch.isdigit() or ch in ".-+eE"]
    s2 = "".join(kept).rstrip("eE+-")
    try:
        val = float(s2)
    except ValueError:
        return float("nan")
    return -val if negative else val


def _tv_find(cols: list[str], *, must: list[str], must_not: list[str] = ()) -> str | None:
    """First column whose lowercased name contains every `must` fragment and none of
    `must_not`. Order of `cols` is the file's own column order."""
    for c in cols:
        lc = c.lower()
        if all(m in lc for m in must) and not any(x in lc for x in must_not):
            return c
    return None


def load_tradingview(path: str, initial_capital: float) -> tuple[pd.Series, pd.DataFrame, str]:
    """Parse a TradingView 'List of Trades' CSV.

    Returns (equity_curve, trades, note). Raises SystemExit with a readable message
    if the file is not recognisable, because the person reading it is not a
    programmer and a traceback tells them nothing.
    """
    raw = pd.read_csv(path)
    cols = list(raw.columns)
    lower = [c.lower() for c in cols]

    c_type = _tv_find(cols, must=["type"])
    c_date = _tv_find(cols, must=["date"]) or _tv_find(cols, must=["time"])
    c_trade = _tv_find(cols, must=["trade"])
    c_price = _tv_find(cols, must=["price"])
    # "Net P&L USD" / "Profit USD" - absolute currency, not the percentage twin.
    c_pnl = (_tv_find(cols, must=["p&l"], must_not=["%", "cum", "run", "draw"])
             or _tv_find(cols, must=["profit"], must_not=["%", "cum", "run", "draw"]))
    c_cum = (_tv_find(cols, must=["cum", "p&l"], must_not=["%"])
             or _tv_find(cols, must=["cum", "profit"], must_not=["%"]))
    c_cum_pct = (_tv_find(cols, must=["cum", "p&l", "%"])
                 or _tv_find(cols, must=["cum", "profit", "%"]))

    if c_date is None or (c_pnl is None and c_cum is None and c_cum_pct is None):
        raise SystemExit(
            "This does not look like a TradingView 'List of Trades' export.\n"
            f"  Columns found: {cols}\n"
            "  Expected a date column and a profit column.\n"
            "  In TradingView: Strategy Tester -> List of Trades -> the export icon\n"
            "  (top right of that panel). Export the LIST OF TRADES, not Performance\n"
            "  Summary and not the chart image.")

    df = raw.copy()
    df["_dt"] = pd.to_datetime(df[c_date], errors="coerce", format="mixed")
    if df["_dt"].isna().all():
        raise SystemExit(f"Could not read any dates from the '{c_date}' column.")
    df = df[df["_dt"].notna()].copy()

    # Pair the two rows of each trade. Prefer the explicit trade number; fall back to
    # reading entry/exit off the Type column when the export has no number column.
    if c_trade is not None:
        df["_grp"] = df[c_trade].astype(str).str.strip()
    elif c_type is not None:
        is_entry = df[c_type].astype(str).str.lower().str.contains("entry")
        df["_grp"] = is_entry.cumsum().astype(str)
    else:
        raise SystemExit("Export has neither a trade-number nor a Type column; cannot pair "
                         "entries with exits.")

    typ = (df[c_type].astype(str).str.lower() if c_type is not None
           else pd.Series("", index=df.index))

    rows = []
    for _, g in df.groupby("_grp", sort=False):
        g = g.sort_values("_dt")
        if len(g) < 2:
            continue                              # position still open at end of test
        t = typ.reindex(g.index)
        entry_mask = t.str.contains("entry", na=False)
        exit_mask = t.str.contains("exit", na=False)
        entry = g[entry_mask].iloc[0] if entry_mask.any() else g.iloc[0]
        exit_ = g[exit_mask].iloc[-1] if exit_mask.any() else g.iloc[-1]

        pnl = float("nan")
        if c_pnl is not None:
            pnl = _tv_num(exit_[c_pnl])
            if math.isnan(pnl):
                pnl = _tv_num(entry[c_pnl])

        rows.append({
            "entry_date": entry["_dt"], "exit_date": exit_["_dt"], "pnl": pnl,
            "_cum": _tv_num(exit_[c_cum]) if c_cum is not None else float("nan"),
            "_cum_pct": _tv_num(exit_[c_cum_pct]) if c_cum_pct is not None else float("nan"),
            "_entry_price": _tv_num(entry[c_price]) if c_price is not None else float("nan"),
            "_exit_price": _tv_num(exit_[c_price]) if c_price is not None else float("nan"),
        })

    if not rows:
        raise SystemExit("No completed trades found. Every trade needs an entry row and an "
                         "exit row; a position still open at the end of the test is skipped.")

    trades = pd.DataFrame(rows).sort_values("exit_date").reset_index(drop=True)

    # Equity curve. Cumulative P&L is the trustworthy source when present because it
    # already reflects TradingView's own accounting; summing individual trade P&L is
    # the fallback and can drift on pyramided positions.
    if trades["_cum"].notna().any():
        equity = initial_capital + trades["_cum"].ffill().fillna(0.0)
        basis = "cumulative P&L"
    elif trades["_cum_pct"].notna().any():
        equity = initial_capital * (1 + trades["_cum_pct"].ffill().fillna(0.0) / 100.0)
        basis = "cumulative P&L %"
    elif trades["pnl"].notna().any():
        equity = initial_capital + trades["pnl"].fillna(0.0).cumsum()
        basis = "summed trade P&L"
    else:
        raise SystemExit("Found trades but no usable profit figures in the export.")

    equity.index = pd.DatetimeIndex(trades["exit_date"])
    equity = equity.astype(float)
    # Seed the curve with starting capital so the first trade's result is counted.
    start = pd.Series([float(initial_capital)],
                      index=pd.DatetimeIndex([trades["entry_date"].min()]))
    equity = pd.concat([start, equity])
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()

    note = (f"Read {len(trades)} completed trades from TradingView. Equity curve built from "
            f"{basis}, starting at {initial_capital:,.0f}.")
    return equity, trades, note


def _tv_buy_and_hold(trades: pd.DataFrame, equity_index: pd.DatetimeIndex) -> pd.Series | None:
    """Approximate buy-and-hold of the traded instrument from the prices in the
    export. Exact at the endpoints, which is all the total-return comparison needs;
    the path between them is interpolated and must not be used for anything that
    depends on the shape of the curve."""
    px = pd.concat([
        pd.Series(trades["_entry_price"].values, index=pd.DatetimeIndex(trades["entry_date"])),
        pd.Series(trades["_exit_price"].values, index=pd.DatetimeIndex(trades["exit_date"])),
    ]).dropna().sort_index()
    px = px[~px.index.duplicated(keep="last")]
    if len(px) < 2 or (px <= 0).any():
        return None
    return px.reindex(px.index.union(equity_index)).interpolate("time").reindex(equity_index).ffill().bfill()


def _read_csv_series(path: str, date_col: str, value_col: str, what: str) -> pd.Series:
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    d, v = cols.get(date_col), cols.get(value_col)
    if not d or not v:
        raise SystemExit(f"{what} CSV needs '{date_col}' and '{value_col}' columns; "
                         f"found {list(df.columns)}")
    s = pd.Series(df[v].values, index=pd.to_datetime(df[d]))
    return s.sort_index().astype(float)


def _fetch_benchmark(ticker: str, start, end) -> pd.Series | None:
    """Fetched on the user's own machine, under their own terms of use. Absent
    yfinance the benchmark checks are skipped rather than the run failing."""
    try:
        import yfinance as yf
    except ImportError:
        print("note: yfinance not installed - skipping benchmark checks. "
              "Supply --benchmark-csv to enable them.", file=sys.stderr)
        return None
    try:
        df = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"),
                                       end=end.strftime("%Y-%m-%d"),
                                       interval="1d", auto_adjust=True)
        if df.empty:
            print(f"note: no benchmark data returned for {ticker}.", file=sys.stderr)
            return None
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        return df["Close"].astype(float)
    except Exception as exc:
        print(f"note: benchmark fetch failed ({exc}) - skipping benchmark checks.", file=sys.stderr)
        return None


def _infer_exposure(trades: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    flags = pd.Series(False, index=index)
    for _, row in trades.iterrows():
        try:
            flags.loc[pd.to_datetime(row["entry_date"]):pd.to_datetime(row["exit_date"])] = True
        except Exception:
            continue
    return flags


def build_report(args) -> str:
    tv_trades = None
    approx_benchmark = False

    if args.tradingview:
        equity, tv_trades, note = load_tradingview(args.tradingview, args.initial_capital)
        print(f"note: {note}", file=sys.stderr)
    else:
        equity = _read_csv_series(args.equity, "date", "equity", "--equity")
    if len(equity) < 2:
        raise SystemExit("Need at least two equity points to say anything.")
    years = (equity.index[-1] - equity.index[0]).days / 365.25

    bench = None
    if args.benchmark_csv:
        bench = _read_csv_series(args.benchmark_csv, "date", "close", "--benchmark-csv")
    elif tv_trades is not None:
        # A TradingView strategy runs on one instrument, which may be crypto or FX.
        # Comparing it to SPY would be meaningless, so buy-and-hold of the traded
        # instrument is reconstructed from the prices in the export instead.
        bench = _tv_buy_and_hold(tv_trades, equity.index)
        approx_benchmark = bench is not None
        if bench is None:
            print("note: no price column in the export, so 'beats buy and hold' cannot be "
                  "checked. Supply --benchmark-csv to enable it.", file=sys.stderr)
    elif not args.no_benchmark:
        bench = _fetch_benchmark(args.benchmark, equity.index[0], equity.index[-1])
    if bench is not None:
        bench = bench.reindex(equity.index).ffill().bfill()
        if bench.isna().all():
            bench = None

    trades = None
    if tv_trades is not None:
        trades = tv_trades
    elif args.trades:
        trades = pd.read_csv(args.trades)
        trades.columns = [c.strip().lower() for c in trades.columns]

    exposure = None
    if args.positions:
        pf = pd.read_csv(args.positions)
        pf.columns = [c.strip().lower() for c in pf.columns]
        exposure = pd.Series(pf["in_market"].astype(float).astype(bool).values,
                             index=pd.to_datetime(pf["date"]))
        exposure = exposure.sort_index().reindex(equity.index).ffill().fillna(False).astype(bool)
    elif trades is not None and {"entry_date", "exit_date"} <= set(trades.columns):
        exposure = _infer_exposure(trades, equity.index)

    results: list[CheckResult] = []
    if bench is not None:
        results.append(check_benchmark(equity, bench))
        # An interpolated price path has the right endpoints but understates the
        # instrument's real day-to-day movement. Total return survives that; the
        # exposure-matched and drawdown checks depend on the shape of the curve and
        # would be quietly wrong, so they are withheld rather than approximated.
        if approx_benchmark:
            results.append(_unknown(
                "exposure_matched", "Selection skill (exposure-matched)",
                "Needs a real price history - supply --benchmark-csv"))
        elif exposure is not None:
            results.append(check_exposure_matched(equity, bench, exposure))
        else:
            results.append(_unknown("exposure_matched", "Selection skill (exposure-matched)",
                                    "Needs --positions or --trades with entry/exit dates"))
    else:
        results.append(_unknown("benchmark", "Beats buy-and-hold", "No benchmark available"))
        results.append(_unknown("exposure_matched", "Selection skill (exposure-matched)",
                                "No benchmark available"))

    results.append(check_deflated_sharpe(equity, args.trial_dispersion))
    results.append(check_performance_decay(equity, bench))
    if bench is not None and not approx_benchmark:
        results.append(check_drawdown_realism(equity, bench))
    elif approx_benchmark:
        results.append(_unknown("drawdown_realism", "Return per unit of drawdown",
                                "Needs a real price history - supply --benchmark-csv"))
    else:
        results.append(_unknown("drawdown_realism", "Return per unit of drawdown",
                                "No benchmark available"))

    if trades is not None:
        results.append(check_sample_size(trades, years))
        results.append(check_return_concentration(trades))
        results.append(check_breakeven_cost(trades, args.assumed_cost))
        results.append(check_survivorship_exposure(trades, years))
    else:
        for k, n in [("sample_size", "Statistical sample size"),
                     ("return_concentration", "Return concentration"),
                     ("breakeven_cost", "Breakeven trading cost"),
                     ("survivorship_exposure", "Survivorship exposure")]:
            results.append(_unknown(k, n, "Needs --trades"))

    results.append(check_look_ahead(_tri(args.look_ahead)))
    results.append(check_out_of_sample(args.oos))
    results.append(check_parameter_sensitivity(_tri(args.plateau)))

    if args.plain:
        title = args.title if args.title != "Backtest Validation Report" else "Backtest Health Check"
        return render_plain(results, title)
    return render(results, args.title)


def main():
    p = argparse.ArgumentParser(
        description="Audit a trading backtest for the flaws that silently inflate it.",
        epilog="Your data never leaves your machine.")
    p.add_argument("--tradingview", metavar="FILE",
                   help="TradingView 'List of Trades' CSV - gives everything in one file")
    p.add_argument("--initial-capital", type=float, default=100000.0,
                   help="Starting capital used in the TradingView test (default 100000)")
    p.add_argument("--equity", help="CSV: date,equity")
    p.add_argument("--trades", help="CSV: entry_date,exit_date,pnl[,ticker]")
    p.add_argument("--positions", help="CSV: date,in_market")
    p.add_argument("--benchmark", default="SPY", help="Benchmark ticker (default SPY)")
    p.add_argument("--benchmark-csv", help="CSV: date,close — use instead of fetching")
    p.add_argument("--no-benchmark", action="store_true", help="Skip benchmark checks entirely")
    p.add_argument("--assumed-cost", type=float, default=5.0,
                   help="Realistic round-trip cost per trade in account currency (default 5.0)")
    p.add_argument("--trial-dispersion", type=float, default=0.5,
                   help="Assumed spread of annualised Sharpe across strategy variants (default 0.5)")
    p.add_argument("--title", default="Backtest Validation Report")
    p.add_argument("--plain", action="store_true",
                   help="Plain-language report for a non-technical reader")
    p.add_argument("--out", help="Write the report to this file as well as stdout")
    # the two checks that still cannot be computed
    p.add_argument("--look-ahead", help="Was screening data as-of the decision date? yes/no")
    p.add_argument("--oos", help="Out-of-sample type: temporal | cross_sectional | none")
    p.add_argument("--plateau", help="Is performance a broad parameter plateau? yes/no")
    args = p.parse_args()
    if not args.tradingview and not args.equity:
        p.error("give either --tradingview export.csv or --equity curve.csv")
    if args.tradingview and args.equity:
        p.error("--tradingview already contains the equity curve; drop --equity")

    report = build_report(args)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")


if __name__ == "__main__":
    main()

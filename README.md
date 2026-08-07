# backtest_validator

Your backtest says +847%. This tells you whether to believe it.

One file. Nothing gets uploaded — it runs on your own computer and your strategy
never leaves it.

```bash
python backtest_validator.py --tradingview trades.csv --plain
```

## Try it before you trust it

Two things you can run right now, using the files in this repo:

```bash
python backtest_validator.py --tradingview examples/tradingview_sample.csv --plain
python examples/noise_test.py
```

The second one builds 200 strategies out of pure random numbers, keeps the luckiest,
and shows you what it looks like. Spoiler: +209% return, Sharpe 1.47, 13% drawdown.
All of it noise.

---

## Getting your file out of TradingView

1. Open the **Strategy Tester** panel at the bottom of the chart
2. Click the **List of Trades** tab
3. Click the export icon at the top right of that panel
4. Save the CSV

Export the **list of trades** — not the Performance Summary, and not a screenshot.

That one file has everything: your trades, your account balance over time, and when
you were in the market. If your test started with something other than 100,000, add
`--initial-capital 25000` (or whatever you used).

## What it tells you

You get a plain-English report. No jargon, no ratios you have to look up:

```
  RESULT:  HIGH RISK  -  4 serious problems found

   1. Nearly all the profit came from a handful of trades
      5 trades out of 111 produced 97% of the profit. Remove those few
      and almost nothing is left. A result that depends on a few lucky
      outcomes is not a system.

   2. The typical trade lost money
      Half of all trades lost more than 304. The total only looks
      positive because a small number of large winners covered the
      rest. Trading fees and the spread come off every single trade,
      winners and losers alike.
```

## The eleven things it checks

- Did it actually beat just buying and holding?
- Did it pick good trades, or only get lucky about *when* it was in the market?
- **Could this result just be luck?** — see below, this is the big one
- Did it work early on and quietly stop working later?
- Would normal trading costs wipe out the profit?
- Did a handful of trades produce nearly all the gains?
- Did the typical trade actually make money?
- Are there enough trades to mean anything?
- Is the calmer ride real, or was it just sitting in cash?
- Did the test only include companies that still exist today?
- Was it ever tested on data it hadn't already seen?

### The one most people fail

If you changed the settings until the profit number looked good, you did not find a
strategy. You found the settings that best fit what already happened.

The tool works out how many versions you could have tried before a result this good
becomes what you would expect from luck alone:

```
  The result may be luck rather than skill
  If more than about 7 versions of this strategy were tried before this
  one was settled on - different settings, different markets, different
  rules - then a result this good is what you would expect from the best
  of those tries by chance alone.
```

It does not ask you how many you tried, because nobody remembers honestly. It gives
you the number and lets you compare it against what you actually did.

To show this is real: I generated **200 completely random fake price histories** —
no strategy, no skill, just a random number generator. The best of those 200 looked
like a solid strategy by every normal measure. The tool flagged it correctly.

## What it will not do

- It will not tell you what to buy or sell. It is not financial advice.
- It will not predict anything. It looks at how carefully a result was measured, not
  at what happens next.
- It will not tell you your strategy works. Passing every check means the obvious
  ways of fooling yourself have been ruled out. That is all it means.
- It will not soften a finding to be nice. That would remove the only reason to use it.

If it can't check enough of your data, it refuses to give a score rather than
grading you on two checks out of eleven.

## Honest limits

- **A TradingView strategy runs on one instrument.** "Beats buy and hold" compares
  against holding that same instrument, rebuilt from the prices in your export. Two
  checks that need full price history are skipped rather than guessed at — you'll
  see them listed as not checked.
- **It can't see inside your strategy.** It only sees the trades that came out.
- **"Worked early, stopped later" can have innocent explanations.** It flags the
  pattern; you decide what it means.
- **The survivorship check estimates** how many companies your test should have lost
  to bankruptcy or takeover. It can't prove your list was biased.

## If you don't use TradingView

Any tool that exports CSV works:

```bash
python backtest_validator.py --equity curve.csv --trades trades.csv
```

`--equity` needs `date,equity`. `--trades` needs `entry_date,exit_date,pnl` and
optionally `ticker`. Drop `--plain` for the technical report with the underlying
statistics. Requires `pandas` and `numpy`; `yfinance` is optional.

MetaTrader 4/5 reports are not supported yet.

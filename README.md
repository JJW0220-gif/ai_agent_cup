# AI CUP 2026 Agent starter

This is a rule-checked starter agent. By default it ranks eligible stocks by
their supplied model scores, drafts whole-lot orders, estimates fees/tax, and
rejects a projected portfolio that violates configured rules. Gemini scoring
and D-Plan drafting are optional features enabled explicitly on the command
line. With Gemini scoring enabled and no `--market` file, it fetches official
daily closing prices; it does not submit to the competition platform.
Existing holdings are retained through rank 30 while new names must enter the
top 22. Rebalancing existing positions is skipped when the target deviation is
less than 0.5% of NAV, and gross daily turnover is capped at 20% of NAV. Initial
portfolio establishment is exempt from the turnover cap. For existing
portfolios, if the cap would leave an invalid projected portfolio, the agent
defers the rebalance rather than exceeding the cap; an already invalid starting
portfolio causes decision generation to fail with its rule violations.

The official 150-stock universe supplied in `data/` was normalized from the
competition PDFs.

## Requirements

- Python 3.10+
- No third-party runtime dependencies

## Input files

`portfolio.json`:

```json
{
  "cash": "100000000",
  "positions": [
    {"symbol": "2330", "shares": 100000}
  ]
}
```

When using a manually prepared `market.csv`, it must contain one row per symbol
with `symbol,price`; an optional `score` column is used by the deterministic
ranker. Prices must be positive and should represent the price estimate used
for order sizing. For D-Plan output, use the official previous-trading-day
close.

With `--use-gemini`, `--market` can be omitted: the agent downloads the latest
official TPEx daily quote feed and requests TWSE's official daily quote table
for that same trading date. Gemini scores five dimensions: fundamentals (25%),
growth (35%), market momentum (15%), catalysts (15%), and risk/uncertainty
(10%). The weighted total is calculated by the agent and the component scores
are included in `decision.json`. The 15% market-momentum allocation is split
equally between Gemini's market assessment and an independently calculated
technical score. Technical scoring combines MACD(12,26,9), confirmed
60-session MACD divergences (40% of technical score), RSI(14) (30%), and
20/50-session moving-average trend (30%).
At least 26 sessions are required to score a symbol. For newer listings, an
unavailable 50-session moving average or 60-session divergence contributes no
trend or divergence adjustment; the unavailable value is recorded as `null`.
ATR(14) inversely adjusts equal-weight position sizes, bounded to +/-25% of
equal weight; it does not trigger trades. The decision output records these
indicator values and signals.

The automatic path downloads up to 60 sessions of daily OHLCV history from
TWSE and TPEx. With a manual `--market` CSV, also provide `--history` containing
`date,symbol,high,low,close,volume` rows for at least 26 sessions per symbol.
Unsupported fundamental, growth, and catalyst dimensions must receive a neutral
50 rather than an invented assessment; risk/uncertainty also considers source
reliability. Official daily quotes alone cannot substantiate fundamentals,
growth, or news catalysts. Supply cited company or news facts with `--evidence`
for those dimensions. The agent rejects missing symbols, stale snapshots, and
dates that do not match the DataPack's `as_of_date`.

The official eligible stock symbols and names are in
[`data/official_universe.json`](data/official_universe.json). This is the
default universe input; `--universe` can override it with another JSON array
containing symbols or `{ "symbol": "...", "name": "..." }` records.

## Run

From the project root:

```bash
python -m pip install -e .
ai-cup-agent \
  --portfolio portfolio.json \
  --market market.csv \
  --output decision.json

python -m unittest discover -s tests -v
```

### Optional Gemini integration

Redeem the competition Token Credits through the organizer's instructions,
then create a Gemini API key in Google AI Studio. Credits and API keys are
separate: the API key is a credential, so do not commit it or paste it into
source files. Set it in the local environment (`GEMINI_API_KEY` is preferred;
`GOOGLE_API_KEY` is also accepted). In WSL:

```bash
export GEMINI_API_KEY="your-local-api-key"
```

The API key is sent only in Google's `x-goog-api-key` request header. The agent
does not store it. `--use-gemini` asks Gemini to return a numeric screening score
for every eligible quote; the code rejects missing, duplicate, unknown, or
out-of-range scores before using them. It does not give Gemini authority to
create or submit orders. The existing deterministic strategy and rule checks
still determine the portfolio and reject non-compliant results. Gemini requests
wait up to 180 seconds. On timeout, the request may already have completed on
Google's side; check whether outputs were produced and avoid immediate retries
that could use additional API credits.

To create a D-Plan v4.3 draft from an initial cash-only DataPack, let the agent
fetch official prices and use Gemini for scoring and D-Plan reasoning:

```bash
ai-cup-agent \
  --datapack data/my-datapack.json \
  --use-gemini \
  --prelaunch-test \
  --gemini-dplan-reasoning \
  --dplan-output outputs/D-Plan_TEAM_123_2026-10-26.json \
  --output outputs/decision.json
```

`--prelaunch-test` is for rehearsals before the competition when the DataPack
points to a future trade date. It allows a recent snapshot to be more than four
calendar days before that future date; it does not permit stale/future quotes or
override a DataPack `as_of_date` mismatch. The resulting plan is a test draft,
not suitable for submission.

When `--market` is omitted and `--use-gemini` is enabled, official quote
snapshots also supply source/observation records for the D-Plan. The `--evidence`
file remains optional in this mode; add it when you have separately verified
news, macro, or company evidence. The agent does not invent those sources.

If you have separately verified news, macro, or company evidence, pass it with
`--evidence research-evidence.json`; official daily quotes alone are limited
market context and do not establish company fundamentals. Without
`--gemini-dplan-reasoning`, the draft uses deterministic review-required
reasoning. The evidence file has this shape (all source URLs, timestamps, facts,
and values must be checked against the underlying evidence):

```json
{
  "sources": [
    {
      "source_id": "S1",
      "authority": "twse",
      "name": "TWSE official daily closing prices",
      "url": "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date=20261021&type=ALLBUT0999&response=json",
      "content_as_of": "2026-10-25T18:30:00+08:00"
    }
  ],
  "observations": [
    {
      "obs_id": "O1",
      "source_ref": ["S1"],
      "topic": "tw_market",
      "statement": "用來源可核對的一句話描述事實。",
      "values": {"TAIEX_chg_pct": -0.5}
    }
  ]
}
```

When `--dplan-output` is used, the program derives decisions and orders from its
deterministic portfolio plan and DataPack, then validates the result against the
included D-Plan v4.3 JSON Schema and additional reference-chain, holding
coverage, order-derivation, weight, cash, net-exposure, and filename checks.
Projected cash posture includes estimated commissions and sell-side tax, using
the same transaction-cost assumptions as the portfolio planner.
The DataPack must contain settled account/holding data. The D-Plan remains a
draft, not an automatic submission or a guarantee of acceptance. Review the
evidence, generated reasoning, market assumptions, and official platform rules
before submitting. The project fetches official daily market prices only; it
does not fetch news or research evidence.

### ETF investment strategy document

The official DOCX template can be filled from approved JSON or drafted by
Gemini using a D-Plan's decisions as limited context. The Gemini path requires
a team-provided strategy direction and a local `GEMINI_API_KEY`:

```bash
ai-cup-strategy-document \
  --dplan outputs/D-Plan_TEAM_11825_2026-10-22.json \
  --guidance "以規則化方式主動選股，重視分散與風險控管" \
  --content-output outputs/ETF投資策略草稿_TEAM_11825.json \
  --output outputs/ETF投資策略說明_TEAM_11825.docx
```

Review the generated JSON before submitting the DOCX. The D-Plan provides only
one run's decisions and is not evidence of company fundamentals, a backtest,
or a long-term mandate. The generator preserves the supplied official DOCX
template and enforces the name (20 characters), theme (50 characters), and
investment-philosophy (100–300 characters) limits. To provide already approved
content instead, use `--input data/ETF策略說明輸入範例.json`.

The decision JSON includes the orders, projected portfolio, projected NAV, and
the generated rationale. A non-zero exit code indicates invalid input or
failed rule checks; do not submit such a decision.

The first version checks whole lots, 20–30 holdings, the eligible universe,
cash and per-stock NAV limits, and fee/tax-adjusted cash. It does not yet
account for corporate actions, the competition platform's exact price/rounding
conventions, the five-day price-drift remediation window, or submission
scheduling.

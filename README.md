# AI CUP 2026 Agent starter

This is a rule-checked starter agent. By default it ranks eligible stocks by
their supplied model scores, drafts whole-lot orders, estimates fees/tax, and
rejects a projected portfolio that violates configured rules. Gemini scoring
and D-Plan drafting are optional features enabled explicitly on the command
line. With Gemini scoring enabled and no `--market` file, it fetches official
daily closing prices; it does not submit to the competition platform.
Benchmark holdings do not affect stock ranking; benchmark data is retained for
portfolio rule checks such as Active Share.
Existing holdings are retained through rank 30 while new names must enter the
top 22. Rebalancing existing positions is skipped when the target deviation is
less than 0.5% of NAV, and gross daily turnover is capped at 20% of NAV. Initial
portfolio establishment and trades needed to restore competition-rule
compliance are exempt from the turnover cap.

The official 150-stock
universe and active-ETF directory supplied in `data/` were normalized from the
competition PDFs. The ETF directory contains ETF identities only, not their
constituent weights. The official top-15 benchmark holdings and weights must
still be supplied before competition use; the agent refuses to generate a
decision if benchmark data is missing or a benchmark comparison fails.

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
are included in `decision.json`. Unsupported dimensions must receive a neutral
50 rather than an invented assessment; risk/uncertainty also considers source
reliability. Official daily quotes alone cannot substantiate fundamentals,
growth, news catalysts, or multi-day/relative-market momentum. Supply cited
company or news facts with `--evidence` for those dimensions. The agent rejects
missing symbols, stale snapshots, and dates that do not match the DataPack's
`as_of_date`.

The official eligible stock symbols and names are in
[`data/official_universe.json`](data/official_universe.json). This is the
default universe input; `--universe` can override it with another JSON array
containing symbols or `{ "symbol": "...", "name": "..." }` records.

[`data/active_etfs.json`](data/active_etfs.json) lists the 30 active ETFs in
the supplied 2026/9/11 competition PDF. It is not the constituent holdings
file. Obtain the official top-15 holdings and weights for each ETF separately
and put those data in `benchmarks.json`.

The agent fetches the latest top-15 snapshots from ETF資訊網 for all 30 ETF
symbols in [`data/active_etfs.json`](data/active_etfs.json) each time it runs.
This is a third-party, unofficial data source; each resulting benchmark record
includes the source URL, snapshot date, and retrieval timestamp. Snapshots
older than five calendar days, missing ETF pages, incomplete tables, and
invalid weights stop decision generation. No official data is silently
substituted or inferred.

The fetched file is saved to `outputs/active_etf_benchmarks.json`. Its
structure is:

```json
{
  "source": "https://www.etfinfo.tw",
  "source_kind": "third_party_unofficial",
  "retrieved_at": "2026-10-21T00:00:00+00:00",
  "benchmarks": {
    "00980A": {
      "as_of": "2026-10-20",
      "source_url": "https://www.etfinfo.tw/etf/00980A/holdings",
      "weights": {"2330": "0.125", "2317": "0.0725"}
    }
  }
}
```

The web data is independently maintained and is not guaranteed to match issuer
disclosures or the competition organizer's calculation. Confirm the organizer
accepts this source before relying on it for eligibility. Offline runs can reuse
a recent saved snapshot with `--offline-benchmarks`.

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

Use `--offline-benchmarks` to avoid fetching the live ETF pages and read the
saved benchmark cache instead.

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
  --offline-benchmarks \
  --dplan-output outputs/D-Plan_TEAM_123_2026-10-26.json \
  --output outputs/decision.json
```

`--prelaunch-test` is for rehearsals before the competition when the DataPack
points to a future trade date. It allows a recent snapshot to be more than four
calendar days before that future date; it does not permit stale/future quotes or
override a DataPack `as_of_date` mismatch. The resulting plan is a test draft,
not suitable for submission. With `--offline-benchmarks`, it also permits old
but otherwise valid cached ETF benchmark snapshots and prints a warning; this
is only to support rehearsals and does not replace refreshing the benchmarks.
Omit this option during the competition, when quotes and benchmarks should be
current and the quotes should be from the previous trading day.

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

The decision JSON includes the orders, projected portfolio, projected NAV,
Active Share versus each supplied benchmark, and the generated rationale. A
non-zero exit code indicates invalid input or failed rule checks; do not submit
such a decision.

The first version checks whole lots, 20–30 holdings, the eligible universe,
cash and per-stock NAV limits, fee/tax-adjusted cash, and 15% Active Share. It
does not yet account for corporate actions, the competition platform's exact
price/rounding conventions, the five-day price-drift remediation window, or
submission scheduling.

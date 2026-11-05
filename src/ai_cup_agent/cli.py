from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ai_cup_agent.models import Portfolio, Quote
from ai_cup_agent.dplan import (
    build_dplan_draft,
    nav_values_match,
    portfolio_from_datapack,
)
from ai_cup_agent.gemini import DEFAULT_MODEL, score_quotes
from ai_cup_agent.strategy import build_decision
from ai_cup_agent.dplan_validation import validate_dplan
from ai_cup_agent.market_data import (
    MarketSnapshot,
    TPEX_HISTORY_URL,
    TWSE_DAILY_URL,
    fetch_official_market_snapshot,
    fetch_official_price_history,
)
from ai_cup_agent.technicals import (
    DailyBar,
    TechnicalIndicators,
    calculate_universe_indicators,
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_quotes(path: Path) -> dict[str, Quote]:
    quotes: dict[str, Quote] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            symbol = row["symbol"].strip()
            if symbol in quotes:
                raise ValueError(f"Duplicate quote for symbol {symbol}.")
            quotes[symbol] = Quote(
                symbol=symbol,
                price=Decimal(row["price"]),
                score=Decimal(row.get("score") or "0"),
            )
    return quotes


def _read_history(
    path: Path,
    symbols: set[str],
) -> dict[str, list[DailyBar]]:
    history: dict[str, list[DailyBar]] = {symbol: [] for symbol in symbols}
    seen: set[tuple[str, date]] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "date", "high", "low", "close", "volume"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(
                "Historical CSV is missing columns: " + ", ".join(missing)
            )
        for row in reader:
            symbol = row["symbol"].strip()
            if symbol not in symbols:
                continue
            try:
                bar_date = date.fromisoformat(row["date"].strip())
                high = Decimal(row["high"])
                low = Decimal(row["low"])
                close = Decimal(row["close"])
                volume = Decimal(row["volume"])
            except (ArithmeticError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Historical CSV has invalid data for {symbol}."
                ) from error
            if any(not value.is_finite() for value in (high, low, close, volume)):
                raise ValueError(
                    f"Historical CSV has non-finite data for {symbol}."
                )
            if high <= 0 or low <= 0 or close <= 0 or high < low or volume < 0:
                raise ValueError(
                    f"Historical CSV has invalid OHLCV values for {symbol}."
                )
            key = (symbol, bar_date)
            if key in seen:
                raise ValueError(
                    f"Historical CSV has duplicate row for {symbol} on {bar_date}."
                )
            seen.add(key)
            history[symbol].append(
                DailyBar(bar_date, high, low, close, volume)
            )
    return history


def _combine_evidence(
    supplied: Any | None,
    generated: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    if supplied is None:
        return generated
    if not isinstance(supplied, dict):
        raise ValueError("Evidence file must contain a JSON object.")
    sources = supplied.get("sources")
    observations = supplied.get("observations")
    if not isinstance(sources, list) or not isinstance(observations, list):
        raise ValueError("Evidence file must contain sources and observations arrays.")
    source_offset = len(sources)
    observation_offset = len(observations)
    combined_sources = [dict(row) for row in sources]
    source_id_map: dict[str, str] = {}
    for index, row in enumerate(generated["sources"], start=1):
        new_id = f"S{source_offset + index}"
        source_id_map[row["source_id"]] = new_id
        combined_sources.append({**row, "source_id": new_id})
    combined_observations = [dict(row) for row in observations]
    for index, row in enumerate(generated["observations"], start=1):
        combined_observations.append(
            {
                **row,
                "obs_id": f"O{observation_offset + index}",
                "source_ref": [source_id_map[ref] for ref in row["source_ref"]],
            }
        )
    return {"sources": combined_sources, "observations": combined_observations}


def _read_universe(path: Path) -> set[str]:
    rows = _read_json(path)
    symbols = {
        str(row["symbol"]) if isinstance(row, dict) else str(row)
        for row in rows
    }
    if not symbols or len(symbols) != len(rows):
        raise ValueError("Eligible-stock universe must be a non-empty list of unique symbols.")
    return symbols


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a rule-checked AI CUP daily portfolio decision."
    )
    parser.add_argument("--portfolio", type=Path, help="Current portfolio JSON.")
    parser.add_argument(
        "--datapack",
        type=Path,
        help="Competition DataPack JSON; also supplies settled positions for a D-Plan.",
    )
    parser.add_argument(
        "--market",
        type=Path,
        help="Optional CSV with symbol,price[,score]; without it, --use-gemini fetches official daily quotes.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        help=(
            "Optional official-history CSV with date,symbol,high,low,close,volume; "
            "required with --market for technical indicators and ATR sizing."
        ),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        help="JSON with sourced D-Plan `sources` and `observations`.",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Use Gemini API to produce screening scores from the supplied input data.",
    )
    parser.add_argument(
        "--prelaunch-test",
        action="store_true",
        help=(
            "Testing only: allow a recent official quote snapshot to predate the "
            "future DataPack trade_date by more than four days."
        ),
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--gemini-dplan-reasoning",
        action="store_true",
        help="Ask Gemini to draft D-Plan reasoning; otherwise use a deterministic review-required draft.",
    )
    default_universe = Path(__file__).resolve().parents[2] / "data" / "official_universe.json"
    parser.add_argument(
        "--universe",
        type=Path,
        default=default_universe,
        help=f"JSON array of eligible symbols (default: {default_universe}).",
    )
    parser.add_argument("--output", type=Path, help="Write decision JSON to this path.")
    parser.add_argument(
        "--dplan-output",
        type=Path,
        help="Write a D-Plan v4.3 draft; requires --datapack and evidence or auto-fetched official quotes.",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    try:
        datapack: dict[str, Any] | None = None
        if args.datapack:
            datapack_portfolio, raw_datapack = portfolio_from_datapack(
                _read_json(args.datapack)
            )
            datapack = raw_datapack
            if args.portfolio:
                portfolio = Portfolio.from_json(_read_json(args.portfolio))
                if (
                    portfolio.cash != datapack_portfolio.cash
                    or {
                        position.symbol: position.shares
                        for position in portfolio.positions
                    }
                    != {
                        position.symbol: position.shares
                        for position in datapack_portfolio.positions
                    }
                ):
                    raise ValueError("--portfolio does not match the supplied DataPack.")
            else:
                portfolio = datapack_portfolio
        elif args.portfolio:
            portfolio = Portfolio.from_json(_read_json(args.portfolio))
        else:
            raise ValueError("Provide either --portfolio or --datapack.")
        allowed_symbols = _read_universe(args.universe)
        market_snapshot: MarketSnapshot | None = None
        if args.prelaunch_test and (args.market or not args.use_gemini):
            raise ValueError(
                "--prelaunch-test requires automatic market data "
                "(omit --market and provide --use-gemini)."
            )
        if args.market:
            quotes = _read_quotes(args.market)
        else:
            if not args.use_gemini:
                raise ValueError(
                    "Provide --market, or use --use-gemini to fetch official market data "
                    "and generate screening scores."
                )
            requested_symbols = allowed_symbols | {
                position.symbol for position in portfolio.positions
            }
            snapshot_trade_date = (
                date.fromisoformat(datapack["trade_date"])
                if datapack is not None
                else date.today()
            )
            snapshot_as_of_date = (
                date.fromisoformat(datapack["as_of_date"])
                if datapack is not None and datapack.get("as_of_date") is not None
                else None
            )
            market_snapshot = fetch_official_market_snapshot(
                requested_symbols,
                trade_date=snapshot_trade_date,
                as_of_date=snapshot_as_of_date,
                allow_prelaunch_snapshot=args.prelaunch_test,
            )
            quotes = market_snapshot.quotes
        technical_symbols = set(quotes) & allowed_symbols
        if args.history:
            history = _read_history(args.history, technical_symbols)
        elif market_snapshot is not None:
            history = fetch_official_price_history(
                technical_symbols,
                through_date=market_snapshot.trade_date,
                tpex_symbols=market_snapshot.tpex_symbols,
            )
        else:
            raise ValueError(
                "--market requires --history to calculate MACD, RSI, moving "
                "averages, and ATR."
            )
        if market_snapshot is not None:
            stale_history = sorted(
                symbol
                for symbol, bars in history.items()
                if bars
                and (
                    max(bar.date for bar in bars) > market_snapshot.trade_date
                    or (
                        market_snapshot.trade_date
                        - max(bar.date for bar in bars)
                    ).days > 4
                )
            )
            if stale_history:
                raise ValueError(
                    "Historical prices are future-dated or stale for: "
                    + ", ".join(stale_history)
                )
        technical_indicators: dict[str, TechnicalIndicators] = (
            calculate_universe_indicators(history, technical_symbols)
        )
        if market_snapshot is not None and not args.history:
            fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
            technical_values: dict[str, float] = {}
            for symbol, indicator in sorted(technical_indicators.items()):
                for metric, value in (
                    ("macd", indicator.macd),
                    ("signal", indicator.signal),
                    ("histogram", indicator.histogram),
                    ("rsi_14", indicator.rsi),
                    ("atr_14", indicator.atr),
                    ("atr_pct", indicator.atr_pct),
                    ("ma_20", indicator.ma20),
                    ("ma_50", indicator.ma50),
                    ("bullish_divergence", Decimal(int(indicator.bullish_divergence))),
                    ("bearish_divergence", Decimal(int(indicator.bearish_divergence))),
                ):
                    if value is not None:
                        technical_values[f"{symbol}_{metric}"] = float(value)
            market_snapshot.evidence["sources"].extend(
                [
                    {
                        "source_id": "S3",
                        "authority": "twse",
                        "name": "TWSE official historical daily stock quotes",
                        "url": (
                            f"{TWSE_DAILY_URL}?date={market_snapshot.trade_date:%Y%m%d}"
                            "&type=ALLBUT0999&response=json"
                        ),
                        "content_as_of": (
                            f"{market_snapshot.trade_date.isoformat()}T18:00:00+08:00"
                        ),
                        "fetched_at": fetched_at,
                    },
                    {
                        "source_id": "S4",
                        "authority": "tpex",
                        "name": "TPEx official historical daily stock quotes",
                        "url": TPEX_HISTORY_URL,
                        "content_as_of": (
                            f"{market_snapshot.trade_date.isoformat()}T18:00:00+08:00"
                        ),
                        "fetched_at": fetched_at,
                    },
                ]
            )
            market_snapshot.evidence["observations"].append(
                {
                    "obs_id": "O2",
                    "source_ref": ["S3", "S4"],
                    "topic": "technical_indicators",
                    "statement": (
                        f"MACD、RSI、ATR 與均線依截至 "
                        f"{market_snapshot.trade_date.isoformat()} 的官方日線資料計算；"
                        "每檔至少使用 26 個交易日；未達所需視窗的 50 日均線與 "
                        "60 日背離訊號不加分也不扣分。"
                    ),
                    "values": technical_values,
                }
            )
        if args.dplan_output and datapack is not None:
            for holding in datapack.get("holdings", []):
                if not isinstance(holding, dict) or not holding.get("shares", 0):
                    continue
                symbol = holding["code"]
                close_price = holding.get("close_price")
                quote = quotes.get(symbol)
                if close_price is None:
                    raise ValueError(
                        f"DataPack has no close_price for held symbol {symbol}."
                    )
                if quote is None or quote.price != Decimal(str(close_price)):
                    raise ValueError(
                        f"Market price for held symbol {symbol} does not match "
                        "the DataPack close_price."
                    )
            account = datapack.get("account")
            if not isinstance(account, dict):
                raise ValueError("DataPack account is required for D-Plan output.")
            calculated_nav = portfolio.cash + sum(
                (
                    Decimal(position.shares) * quotes[position.symbol].price
                    for position in portfolio.positions
                ),
                Decimal("0"),
            )
            if not nav_values_match(
                calculated_nav, Decimal(str(account["nav_tradable"]))
            ):
                raise ValueError(
                    "DataPack nav_tradable does not match cash plus held shares "
                    "at the supplied closing prices."
                )
        allowed_symbols = _read_universe(args.universe)
        supplied_evidence = _read_json(args.evidence) if args.evidence else None
        evidence = (
            _combine_evidence(
                supplied_evidence,
                market_snapshot.evidence,
            )
            if market_snapshot is not None
            else supplied_evidence
        )
        if args.dplan_output and datapack is None:
            raise ValueError("--dplan-output requires --datapack.")
        if args.dplan_output and evidence is None:
            raise ValueError(
                "--dplan-output requires --evidence when using a market CSV; "
                "automatic official market data supplies its own evidence."
            )
        if args.gemini_dplan_reasoning and not args.dplan_output:
            raise ValueError("--gemini-dplan-reasoning requires --dplan-output.")
        use_gemini = args.use_gemini
        screening_scores = None
        if use_gemini:
            screening_scores = score_quotes(
                quotes,
                allowed_symbols,
                evidence=evidence,
                technical_indicators=technical_indicators,
                model=args.gemini_model,
            )
            quotes = {
                symbol: Quote(
                    symbol=quote.symbol,
                    price=quote.price,
                    score=(
                        screening_scores[symbol].total
                        if symbol in screening_scores
                        else quote.score
                    ),
                    change_pct=quote.change_pct,
                    volume=quote.volume,
                    turnover=quote.turnover,
                )
                for symbol, quote in quotes.items()
            }
        decision = build_decision(
            portfolio,
            quotes,
            allowed_symbols,
            technical_indicators=technical_indicators,
        )
        decision_json = decision.to_json()
        if screening_scores is not None:
            decision_json["screening_scores"] = {
                symbol: {
                    **{
                        name: str(score)
                        for name, score in card.components.items()
                    },
                    "technical_score": str(card.technical_score),
                    "total": str(card.total),
                }
                for symbol, card in sorted(screening_scores.items())
            }
        decision_json["technical_indicators"] = {
            symbol: {
                "macd": str(indicator.macd),
                "signal": str(indicator.signal),
                "histogram": str(indicator.histogram),
                "macd_cross": indicator.macd_cross,
                "bullish_divergence": indicator.bullish_divergence,
                "bearish_divergence": indicator.bearish_divergence,
                "rsi_14": str(indicator.rsi),
                "atr_14": str(indicator.atr),
                "atr_pct": str(indicator.atr_pct),
                "ma_20": str(indicator.ma20),
                "ma_50": (
                    str(indicator.ma50) if indicator.ma50 is not None else None
                ),
                "ma_20_rising": indicator.ma20_rising,
                "ma_50_rising": indicator.ma50_rising,
                "technical_score": str(indicator.technical_score),
            }
            for symbol, indicator in sorted(technical_indicators.items())
        }
        output = json.dumps(decision_json, ensure_ascii=False, indent=2)
        dplan_json: dict[str, Any] | None = None
        if args.dplan_output:
            if datapack is None or evidence is None:
                raise ValueError("D-Plan inputs are unavailable.")
            dplan_json = build_dplan_draft(
                evidence=evidence,
                datapack=datapack,
                portfolio=portfolio,
                quotes=quotes,
                decision=decision,
                model=args.gemini_model,
                use_gemini_reasoning=args.gemini_dplan_reasoning,
            )
            account = datapack.get("account")
            if not isinstance(account, dict):
                raise ValueError("DataPack account is unavailable for D-Plan validation.")
            schema_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "D-Plan.schema_v4.3.json"
            )
            dplan_errors = validate_dplan(
                dplan_json,
                schema_path=schema_path,
                portfolio=portfolio,
                quotes=quotes,
                nav=Decimal(str(account["nav_tradable"])),
                allowed_symbols=allowed_symbols,
            )
            if (
                dplan_json.get("team_id") != datapack.get("team_id")
                or dplan_json.get("trade_date") != datapack.get("trade_date")
            ):
                dplan_errors.append("D-Plan team_id/trade_date must match the DataPack.")
            expected_filename = (
                f"D-Plan_{datapack['team_id']}_{datapack['trade_date']}.json"
            )
            if args.dplan_output.name != expected_filename:
                dplan_errors.append(
                    f"D-Plan output filename must be {expected_filename}."
                )
            if dplan_errors:
                raise ValueError(
                    "D-Plan validation failed:\n- " + "\n- ".join(dplan_errors)
                )
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        if args.dplan_output and dplan_json is not None:
            args.dplan_output.write_text(
                json.dumps(dplan_json, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ai-cup-agent: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

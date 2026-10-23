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
from ai_cup_agent.dplan import build_dplan_draft, portfolio_from_datapack
from ai_cup_agent.fetch_benchmarks import (
    MAX_SNAPSHOT_AGE_DAYS,
    SOURCE_KIND,
    SOURCE_ROOT,
    fetch_benchmarks,
)
from ai_cup_agent.gemini import DEFAULT_MODEL, score_quotes
from ai_cup_agent.rules import validate_benchmarks
from ai_cup_agent.strategy import build_decision
from ai_cup_agent.dplan_validation import validate_dplan
from ai_cup_agent.market_data import MarketSnapshot, fetch_official_market_snapshot


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


def _read_benchmarks(
    path: Path,
    expected_benchmarks: set[str],
    *,
    allow_stale: bool = False,
) -> tuple[dict[str, dict[str, Decimal]], dict[str, dict[str, str]]]:
    rows = _read_json(path)
    if not isinstance(rows, dict) or rows.get("source") != SOURCE_ROOT:
        raise ValueError("Benchmark file must include the expected ETFInfo source metadata.")
    if rows.get("source_kind") != SOURCE_KIND:
        raise ValueError(f"Benchmark source must be marked {SOURCE_KIND}.")
    retrieved_at = rows.get("retrieved_at")
    if not isinstance(retrieved_at, str):
        raise ValueError("Benchmark file is missing its retrieval timestamp.")
    try:
        datetime.fromisoformat(retrieved_at)
    except ValueError as error:
        raise ValueError("Benchmark retrieval timestamp is not valid ISO 8601.") from error
    raw_benchmarks = rows.get("benchmarks")
    if not isinstance(raw_benchmarks, dict):
        raise ValueError("Benchmark file must contain a benchmarks object.")
    benchmarks: dict[str, dict[str, Decimal]] = {}
    metadata: dict[str, dict[str, str]] = {}
    for name, record in raw_benchmarks.items():
        if not isinstance(record, dict):
            raise ValueError(f"Benchmark {name} must include snapshot metadata.")
        if not isinstance(record.get("weights"), dict):
            raise ValueError(f"Benchmark {name} must include its stock weights.")
        source_url = record.get("source_url")
        expected_url = f"{SOURCE_ROOT}/etf/{name}/holdings"
        if source_url != expected_url:
            raise ValueError(f"Benchmark {name} has an unexpected source URL.")
        as_of = record.get("as_of")
        if not isinstance(as_of, str):
            raise ValueError(f"Benchmark {name} is missing its snapshot date.")
        snapshot_date = date.fromisoformat(as_of)
        age_days = (date.today() - snapshot_date).days
        if age_days < 0 or (age_days > MAX_SNAPSHOT_AGE_DAYS and not allow_stale):
            raise ValueError(
                f"Benchmark {name} snapshot date {as_of} is future-dated or older "
                f"than {MAX_SNAPSHOT_AGE_DAYS} days."
            )
        benchmarks[name] = {
            symbol: Decimal(str(weight))
            for symbol, weight in record["weights"].items()
        }
        metadata[name] = {"as_of": as_of, "source_url": source_url}

    errors = validate_benchmarks(benchmarks, expected_benchmarks)
    if errors:
        raise ValueError("Invalid Active Share benchmark data:\n- " + "\n- ".join(errors))
    return benchmarks, metadata


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
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=Path("outputs") / "active_etf_benchmarks.json",
        help="Benchmark cache/output path.",
    )
    parser.add_argument(
        "--offline-benchmarks",
        action="store_true",
        help="Use the existing benchmark cache instead of fetching fresh snapshots.",
    )
    default_etfs = Path(__file__).resolve().parents[2] / "data" / "active_etfs.json"
    parser.add_argument(
        "--active-etfs",
        type=Path,
        default=default_etfs,
        help=f"Official active-ETF list (default: {default_etfs}).",
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
            allowed_symbols = _read_universe(args.universe)
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
            if calculated_nav != Decimal(str(account["nav_tradable"])):
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
        expected_benchmarks = _read_universe(args.active_etfs)
        if args.offline_benchmarks:
            benchmarks, benchmark_metadata = _read_benchmarks(
                args.benchmarks,
                expected_benchmarks,
                allow_stale=args.prelaunch_test,
            )
            if args.prelaunch_test:
                today = date.today()
                for name, metadata in benchmark_metadata.items():
                    age_days = (
                        today - date.fromisoformat(metadata["as_of"])
                    ).days
                    if age_days > MAX_SNAPSHOT_AGE_DAYS:
                        print(
                            "WARNING: Prelaunch test is using stale ETF benchmark "
                            f"{name} from {metadata['as_of']} ({age_days} days old); "
                            "do not submit this draft.",
                            file=sys.stderr,
                        )
        else:
            result = fetch_benchmarks(sorted(expected_benchmarks))
            args.benchmarks.parent.mkdir(parents=True, exist_ok=True)
            args.benchmarks.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            benchmarks, benchmark_metadata = _read_benchmarks(
                args.benchmarks, expected_benchmarks
            )
        decision = build_decision(
            portfolio,
            quotes,
            allowed_symbols,
            benchmarks,
            expected_benchmarks=expected_benchmarks,
        )
        decision_json = decision.to_json()
        decision_json["benchmark_metadata"] = benchmark_metadata
        if screening_scores is not None:
            decision_json["screening_scores"] = {
                symbol: {
                    **{
                        name: str(score)
                        for name, score in card.components.items()
                    },
                    "total": str(card.total),
                }
                for symbol, card in sorted(screening_scores.items())
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

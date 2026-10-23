from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

SOURCE_ROOT = "https://www.etfinfo.tw"
SOURCE_KIND = "third_party_unofficial"
USER_AGENT = "Mozilla/5.0 (compatible; AICUPPortfolioAgent/0.1)"
MAX_SNAPSHOT_AGE_DAYS = 5


class _HoldingsPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.tables: list[list[list[str]]] = []
        self._in_title = False
        self._table_index: int | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag == "table":
            self.tables.append([])
            self._table_index = len(self.tables) - 1
        elif tag == "tr" and self._table_index is not None:
            self._row = []
            self.tables[self._table_index].append(self._row)
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in {"td", "th"} and self._cell_parts is not None:
            if self._row is not None:
                self._row.append(" ".join("".join(self._cell_parts).split()))
            self._cell_parts = None
        elif tag == "table":
            self._table_index = None
            self._row = None


def _extract_holdings_page(
    html: str,
    symbol: str,
    *,
    today: date,
    max_age_days: int,
) -> dict[str, Any]:
    parser = _HoldingsPageParser()
    parser.feed(html)
    title = " ".join(" ".join(parser.title_parts).split())
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", title)
    if not date_match:
        raise ValueError(f"{symbol}: holdings snapshot date was not found in the page title.")
    as_of = date.fromisoformat(date_match.group(1))
    age_days = (today - as_of).days
    if age_days < 0:
        raise ValueError(f"{symbol}: snapshot date {as_of} is in the future.")
    if age_days > max_age_days:
        raise ValueError(
            f"{symbol}: latest snapshot {as_of} is {age_days} days old "
            f"(maximum {max_age_days})."
        )

    holdings: dict[str, str] = {}
    for table in parser.tables:
        if not table:
            continue
        header = table[0]
        code_column = next(
            (index for index, value in enumerate(header) if "代號" in value),
            None,
        )
        weight_column = next(
            (index for index, value in enumerate(header) if "權重" in value),
            None,
        )
        if code_column is None or weight_column is None:
            continue
        for row in table[1:]:
            if max(code_column, weight_column) >= len(row):
                continue
            code_match = re.match(r"\s*([A-Z0-9]{1,12})", row[code_column].upper())
            weight_match = re.search(
                r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*%",
                row[weight_column],
            )
            if not code_match or not weight_match:
                continue
            stock_symbol = code_match.group(1)
            weight = Decimal(weight_match.group(1).replace(",", "")) / Decimal("100")
            if weight <= 0 or stock_symbol in holdings:
                raise ValueError(
                    f"{symbol}: invalid or duplicate holding {stock_symbol}."
                )
            holdings[stock_symbol] = str(weight)
        if holdings:
            break

    if len(holdings) != 15:
        raise ValueError(
            f"{symbol}: expected exactly 15 top holdings with weights, "
            f"parsed {len(holdings)}."
        )
    if sum((Decimal(weight) for weight in holdings.values()), Decimal("0")) > Decimal("1"):
        raise ValueError(f"{symbol}: top-holdings weights exceed 100%.")

    return {
        "as_of": as_of.isoformat(),
        "source_url": f"{SOURCE_ROOT}/etf/{symbol}/holdings",
        "weights": holdings,
    }


def _fetch_one(
    symbol: str,
    *,
    today: date,
    max_age_days: int,
    retries: int,
) -> dict[str, Any]:
    url = f"{SOURCE_ROOT}/etf/{symbol}/holdings"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            )
            with urlopen(request, timeout=20) as response:
                if response.status != 200:
                    raise ValueError(f"{symbol}: source returned HTTP {response.status}.")
                html = response.read().decode("utf-8")
            return _extract_holdings_page(
                html,
                symbol,
                today=today,
                max_age_days=max_age_days,
            )
        except (OSError, UnicodeDecodeError, URLError, ValueError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    raise ValueError(f"{symbol}: failed to fetch holdings: {last_error}")


def fetch_benchmarks(
    symbols: list[str],
    *,
    max_age_days: int = MAX_SNAPSHOT_AGE_DAYS,
    retries: int = 2,
    today: date | None = None,
) -> dict[str, Any]:
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("Active ETF symbols must be a non-empty list of unique values.")
    snapshot_date = today or date.today()
    benchmarks: dict[str, Any] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                _fetch_one,
                symbol,
                today=snapshot_date,
                max_age_days=max_age_days,
                retries=retries,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                benchmarks[symbol] = future.result()
            except ValueError as error:
                failures.append(str(error))
    if failures:
        raise ValueError(
            "Benchmark refresh incomplete; no decision should be generated:\n- "
            + "\n- ".join(sorted(failures))
        )
    return {
        "source": SOURCE_ROOT,
        "source_kind": SOURCE_KIND,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {symbol: benchmarks[symbol] for symbol in symbols},
    }


def _read_etf_symbols(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("Active ETF list must be a JSON array.")
    symbols = [
        str(row["symbol"]) if isinstance(row, dict) else str(row)
        for row in rows
    ]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("Active ETF list must contain unique symbols.")
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the latest top-15 holdings snapshots for active ETFs."
    )
    default_etfs = Path(__file__).resolve().parents[2] / "data" / "active_etfs.json"
    parser.add_argument("--active-etfs", type=Path, default=default_etfs)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "active_etf_benchmarks.json",
    )
    parser.add_argument("--max-age-days", type=int, default=MAX_SNAPSHOT_AGE_DAYS)
    args = parser.parse_args()

    try:
        if args.max_age_days < 0:
            raise ValueError("--max-age-days cannot be negative.")
        result = fetch_benchmarks(
            _read_etf_symbols(args.active_etfs),
            max_age_days=args.max_age_days,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Fetched {len(result['benchmarks'])} ETF snapshots to {args.output}.")
        return 0
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        print(f"ai-cup-benchmarks: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

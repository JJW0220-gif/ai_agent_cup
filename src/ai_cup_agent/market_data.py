from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from http.client import IncompleteRead
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_cup_agent.models import Quote

TWSE_DAILY_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
TAIPEI_TIMEZONE = timezone(timedelta(hours=8))
USER_AGENT = "Mozilla/5.0 (compatible; AICUPPortfolioAgent/0.1)"
MAX_SNAPSHOT_AGE_DAYS = 4
MAX_REQUEST_ATTEMPTS = 3


@dataclass(frozen=True)
class MarketSnapshot:
    trade_date: date
    quotes: dict[str, Quote]
    evidence: dict[str, list[dict[str, Any]]]


def _request_json(url: str, *, venue: str) -> Any:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError(
                        f"{venue} market API returned HTTP {response.status}."
                    )
                return json.loads(response.read().decode("utf-8"))
        except IncompleteRead as error:
            if attempt == MAX_REQUEST_ATTEMPTS:
                received = len(error.partial)
                expected = received + error.expected
                raise ValueError(
                    f"Could not retrieve {venue} official market data: "
                    f"response ended early ({received} of {expected} bytes received) "
                    f"after {MAX_REQUEST_ATTEMPTS} attempts."
                ) from error
            sleep(0.5 * attempt)
        except HTTPError as error:
            raise ValueError(f"{venue} market API returned HTTP {error.code}.") from error
        except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Could not retrieve {venue} official market data: {error}"
            ) from error
    raise AssertionError("Market request retry loop exited unexpectedly.")


def _request_rows(url: str, *, venue: str) -> list[dict[str, Any]]:
    payload = _request_json(url, venue=venue)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{venue} market API returned no daily quote rows.")
    if any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"{venue} market API returned an invalid row.")
    return payload


def _parse_roc_date(value: Any, *, venue: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{venue} market row has no ROC date.")
    match = re.fullmatch(r"\s*(\d{2,3})(\d{2})(\d{2})\s*", value)
    if not match:
        raise ValueError(f"{venue} market API returned an unsupported date: {value!r}.")
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year + 1911, month, day)
    except ValueError as error:
        raise ValueError(f"{venue} market API returned an invalid date: {value!r}.") from error


def _decimal(value: Any, *, field: str, symbol: str, venue: str) -> Decimal:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError(f"{venue} quote {symbol} has no numeric {field}.")
    text = str(value).replace(",", "").strip()
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(
            f"{venue} quote {symbol} has invalid {field}: {value!r}."
        ) from error
    if not number.is_finite():
        raise ValueError(f"{venue} quote {symbol} has non-finite {field}.")
    return number


def _parse_venue(
    rows: list[dict[str, Any]],
    *,
    venue: str,
    requested_symbols: set[str],
) -> tuple[date, dict[str, Quote]]:
    if venue == "TWSE":
        fields = {
            "date": "Date",
            "symbol": "Code",
            "price": "ClosingPrice",
            "change": "Change",
            "volume": "TradeVolume",
            "turnover": "TradeValue",
        }
    else:
        fields = {
            "date": "Date",
            "symbol": "SecuritiesCompanyCode",
            "price": "Close",
            "change": "Change",
            "volume": "TradingShares",
            "turnover": "TransactionAmount",
        }
    parsed_dates = [
        _parse_roc_date(row.get(fields["date"]), venue=venue) for row in rows
    ]
    snapshot_date = max(parsed_dates)
    quotes: dict[str, Quote] = {}
    for row, row_date in zip(rows, parsed_dates):
        if row_date != snapshot_date:
            continue
        symbol_value = row.get(fields["symbol"])
        if not isinstance(symbol_value, str):
            continue
        symbol = symbol_value.strip()
        if symbol not in requested_symbols:
            continue
        if symbol in quotes:
            raise ValueError(f"{venue} market API returned duplicate quote for {symbol}.")
        price = _decimal(
            row.get(fields["price"]), field="close", symbol=symbol, venue=venue
        )
        if price <= 0:
            raise ValueError(f"{venue} quote {symbol} has a non-positive close.")
        change = _decimal(
            row.get(fields["change"]), field="change", symbol=symbol, venue=venue
        )
        previous_close = price - change
        change_pct = (
            change / previous_close * Decimal("100")
            if previous_close > 0
            else Decimal("0")
        )
        volume = _decimal(
            row.get(fields["volume"]), field="volume", symbol=symbol, venue=venue
        )
        turnover = _decimal(
            row.get(fields["turnover"]), field="turnover", symbol=symbol, venue=venue
        )
        quotes[symbol] = Quote(
            symbol=symbol,
            price=price,
            score=Decimal("0"),
            change_pct=change_pct,
            volume=volume,
            turnover=turnover,
        )
    return snapshot_date, quotes


def _parse_twse_daily(
    payload: Any,
    *,
    expected_date: date,
    requested_symbols: set[str],
) -> dict[str, Quote]:
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        status = payload.get("stat") if isinstance(payload, dict) else "invalid response"
        raise ValueError(f"TWSE daily quote API did not return data: {status}.")
    response_date = payload.get("date")
    if response_date != expected_date.strftime("%Y%m%d"):
        raise ValueError(
            f"TWSE daily quote API returned date {response_date!r}; "
            f"expected {expected_date:%Y%m%d}."
        )
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError("TWSE daily quote API returned no tables.")
    table = next(
        (
            item
            for item in tables
            if isinstance(item, dict)
            and "每日收盤行情(全部" in str(item.get("title", ""))
        ),
        None,
    )
    if not isinstance(table, dict):
        raise ValueError("TWSE daily quote table was not found.")
    fields = table.get("fields")
    rows = table.get("data")
    if not isinstance(fields, list) or not isinstance(rows, list):
        raise ValueError("TWSE daily quote table is malformed.")
    field_indexes = {name: index for index, name in enumerate(fields)}
    needed = {
        "code": "證券代號",
        "volume": "成交股數",
        "turnover": "成交金額",
        "close": "收盤價",
        "direction": "漲跌(+/-)",
        "change": "漲跌價差",
    }
    missing_fields = [name for name in needed.values() if name not in field_indexes]
    if missing_fields:
        raise ValueError(
            "TWSE daily quote table is missing columns: " + ", ".join(missing_fields)
        )
    quotes: dict[str, Quote] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) < len(fields):
            continue
        symbol = str(row[field_indexes[needed["code"]]]).strip()
        if symbol not in requested_symbols:
            continue
        if symbol in quotes:
            raise ValueError(f"TWSE daily quote API returned duplicate quote for {symbol}.")
        price = _decimal(
            row[field_indexes[needed["close"]]],
            field="close",
            symbol=symbol,
            venue="TWSE",
        )
        if price <= 0:
            raise ValueError(f"TWSE quote {symbol} has a non-positive close.")
        change = _decimal(
            row[field_indexes[needed["change"]]],
            field="change",
            symbol=symbol,
            venue="TWSE",
        )
        direction = str(row[field_indexes[needed["direction"]]])
        sign_match = re.search(r"[+-]", direction)
        if sign_match and sign_match.group(0) == "-":
            change = -change
        elif not sign_match:
            change = Decimal("0")
        previous_close = price - change
        change_pct = (
            change / previous_close * Decimal("100")
            if previous_close > 0
            else Decimal("0")
        )
        volume = _decimal(
            row[field_indexes[needed["volume"]]],
            field="volume",
            symbol=symbol,
            venue="TWSE",
        )
        turnover = _decimal(
            row[field_indexes[needed["turnover"]]],
            field="turnover",
            symbol=symbol,
            venue="TWSE",
        )
        quotes[symbol] = Quote(
            symbol=symbol,
            price=price,
            score=Decimal("0"),
            change_pct=change_pct,
            volume=volume,
            turnover=turnover,
        )
    return quotes


def fetch_official_market_snapshot(
    symbols: set[str],
    *,
    trade_date: date,
    as_of_date: date | None = None,
    today: date | None = None,
    allow_prelaunch_snapshot: bool = False,
) -> MarketSnapshot:
    if not symbols:
        raise ValueError("Cannot fetch market prices for an empty symbol list.")
    tpex_rows = _request_rows(TPEX_URL, venue="TPEx")
    tpex_date, tpex_quotes = _parse_venue(
        tpex_rows, venue="TPEx", requested_symbols=symbols
    )
    twse_url = (
        f"{TWSE_DAILY_URL}?date={tpex_date:%Y%m%d}"
        "&type=ALLBUT0999&response=json"
    )
    twse_payload = _request_json(twse_url, venue="TWSE")
    twse_quotes = _parse_twse_daily(
        twse_payload,
        expected_date=tpex_date,
        requested_symbols=symbols,
    )
    if not twse_quotes:
        raise ValueError(
            f"TWSE daily quote API returned no requested symbols for {tpex_date}."
        )
    snapshot_date = tpex_date
    if as_of_date is not None and snapshot_date != as_of_date:
        raise ValueError(
            f"Official market snapshot is {snapshot_date}, but DataPack as_of_date "
            f"is {as_of_date}; download matching previous-close prices."
        )
    if snapshot_date >= trade_date:
        raise ValueError(
            f"Market snapshot {snapshot_date} is not before D-Plan trade_date {trade_date}."
        )
    current_date = today or date.today()
    snapshot_age = (current_date - snapshot_date).days
    if snapshot_age < 0:
        raise ValueError(
            f"Official market snapshot {snapshot_date} is later than system date "
            f"{current_date}."
        )
    if snapshot_age > MAX_SNAPSHOT_AGE_DAYS:
        raise ValueError(
            f"Official market snapshot {snapshot_date} is stale; maximum age is "
            f"{MAX_SNAPSHOT_AGE_DAYS} calendar days."
        )
    if (
        not allow_prelaunch_snapshot
        and (trade_date - snapshot_date).days > MAX_SNAPSHOT_AGE_DAYS
    ):
        raise ValueError(
            f"Market snapshot {snapshot_date} is too far before trade_date "
            f"{trade_date}; refresh prices on the previous trading day."
        )

    quotes = twse_quotes | tpex_quotes
    missing = sorted(symbols - quotes.keys())
    if missing:
        raise ValueError(
            "Official market feeds have no closing quotes for: " + ", ".join(missing)
        )

    captured_at = datetime.now(TAIPEI_TIMEZONE).isoformat(timespec="seconds")
    content_as_of = datetime.combine(
        snapshot_date, datetime.min.time(), tzinfo=TAIPEI_TIMEZONE
    ).replace(hour=18, minute=0).isoformat(timespec="seconds")
    source_rows = [
        {
            "source_id": "S1",
            "authority": "twse",
            "name": "TWSE official daily listed-stock closing quotes",
            "url": twse_url,
            "content_as_of": content_as_of,
            "fetched_at": captured_at,
        },
        {
            "source_id": "S2",
            "authority": "tpex",
            "name": "TPEx official daily OTC-stock closing quotes",
            "url": TPEX_URL,
            "content_as_of": content_as_of,
            "fetched_at": captured_at,
        },
    ]
    values: dict[str, float] = {}
    for symbol, quote in sorted(quotes.items()):
        if quote.change_pct is None or quote.volume is None or quote.turnover is None:
            raise ValueError(f"Official quote for {symbol} is missing market features.")
        values[f"{symbol}_close"] = float(quote.price)
        values[f"{symbol}_change_pct"] = float(quote.change_pct)
        values[f"{symbol}_volume"] = float(quote.volume)
        values[f"{symbol}_turnover"] = float(quote.turnover)
    observations = [
        {
            "obs_id": "O1",
            "source_ref": ["S1", "S2"],
            "topic": "daily_market",
            "statement": (
                f"TWSE 與 TPEx 官方日行情快照日期為 {snapshot_date.isoformat()}；"
                f"涵蓋策略標的池中的 {len(quotes)} 檔證券，包含收盤價、漲跌幅、"
                "成交股數與成交金額。"
            ),
            "values": values,
        }
    ]
    return MarketSnapshot(
        trade_date=snapshot_date,
        quotes=quotes,
        evidence={"sources": source_rows, "observations": observations},
    )

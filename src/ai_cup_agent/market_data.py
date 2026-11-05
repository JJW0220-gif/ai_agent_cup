from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from threading import Lock
from time import monotonic, sleep
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_cup_agent.models import Quote
from ai_cup_agent.technicals import (
    DailyBar,
    HISTORY_SESSIONS,
    MIN_TECHNICAL_SESSIONS,
)

TWSE_DAILY_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
TPEX_HISTORY_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
TAIPEI_TIMEZONE = timezone(timedelta(hours=8))
USER_AGENT = "Mozilla/5.0 (compatible; AICUPPortfolioAgent/0.1)"
MAX_SNAPSHOT_AGE_DAYS = 4
MAX_REQUEST_ATTEMPTS = 3
TPEX_MIN_REQUEST_INTERVAL_SECONDS = 0.25
_tpex_request_lock = Lock()
_last_tpex_request = 0.0


@dataclass(frozen=True)
class MarketSnapshot:
    trade_date: date
    quotes: dict[str, Quote]
    evidence: dict[str, list[dict[str, Any]]]
    tpex_symbols: set[str]


def _request_json(url: str, *, venue: str) -> Any:
    global _last_tpex_request
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        if venue == "TPEx":
            with _tpex_request_lock:
                delay = TPEX_MIN_REQUEST_INTERVAL_SECONDS - (
                    monotonic() - _last_tpex_request
                )
                if delay > 0:
                    sleep(delay)
                _last_tpex_request = monotonic()
        try:
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError(
                        f"{venue} market API returned HTTP {response.status}."
                    )
                body = response.read()
                try:
                    return json.loads(body.decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    if attempt == MAX_REQUEST_ATTEMPTS:
                        content_type = response.headers.get(
                            "Content-Type", "unknown"
                        )
                        preview = body[:120].decode("utf-8", errors="replace").strip()
                        if not preview:
                            preview = "<empty>"
                        raise ValueError(
                            f"Could not retrieve {venue} official market data: "
                            f"invalid JSON response (HTTP {response.status}, "
                            f"content type {content_type}, body starts with "
                            f"{preview!r}) after {MAX_REQUEST_ATTEMPTS} attempts."
                        ) from error
                    sleep(float(attempt))
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
            if error.code == 429 and attempt < MAX_REQUEST_ATTEMPTS:
                retry_after = error.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 0.0
                except ValueError:
                    delay = 0.0
                sleep(max(delay, float(attempt)))
                continue
            raise ValueError(f"{venue} market API returned HTTP {error.code}.") from error
        except (OSError, URLError) as error:
            if attempt == MAX_REQUEST_ATTEMPTS:
                raise ValueError(
                    f"Could not retrieve {venue} official market data after "
                    f"{MAX_REQUEST_ATTEMPTS} attempts: {error}"
                ) from error
            sleep(0.5 * attempt)
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
        and (trade_date - current_date).days > 1
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
        tpex_symbols=set(tpex_quotes),
    )


def _parse_twse_history(
    payload: Any,
    *,
    expected_date: date,
    requested_symbols: set[str],
) -> dict[str, DailyBar]:
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        status = payload.get("stat") if isinstance(payload, dict) else "invalid response"
        if isinstance(status, str) and "沒有符合條件的資料" in status:
            return {}
        raise ValueError(f"TWSE historical API returned no data: {status}.")
    if payload.get("date") != expected_date.strftime("%Y%m%d"):
        raise ValueError(
            f"TWSE historical API returned date {payload.get('date')!r}; "
            f"expected {expected_date:%Y%m%d}."
        )
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError("TWSE historical API returned no tables.")
    table = next(
        (
            item
            for item in tables
            if isinstance(item, dict)
            and "每日收盤行情(全部" in str(item.get("title", ""))
        ),
        None,
    )
    if table is None:
        return {}
    fields = table.get("fields")
    rows = table.get("data")
    if not isinstance(fields, list) or not isinstance(rows, list):
        raise ValueError("TWSE historical daily quote table is malformed.")
    indexes = {name: index for index, name in enumerate(fields)}
    required = ("證券代號", "最高價", "最低價", "收盤價", "成交股數")
    if any(name not in indexes for name in required):
        raise ValueError("TWSE historical daily quote table is missing OHLCV fields.")
    bars: dict[str, DailyBar] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) < len(fields):
            continue
        symbol = str(row[indexes["證券代號"]]).strip()
        if symbol not in requested_symbols:
            continue
        raw_values = [row[indexes[name]] for name in required[1:]]
        if any(str(value).strip() in {"", "--", "-"} for value in raw_values):
            continue
        high = _decimal(row[indexes["最高價"]], field="high", symbol=symbol, venue="TWSE")
        low = _decimal(row[indexes["最低價"]], field="low", symbol=symbol, venue="TWSE")
        close = _decimal(row[indexes["收盤價"]], field="close", symbol=symbol, venue="TWSE")
        volume = _decimal(row[indexes["成交股數"]], field="volume", symbol=symbol, venue="TWSE")
        if high <= 0 or low <= 0 or close <= 0:
            continue
        if symbol in bars:
            raise ValueError(
                f"TWSE historical API returned duplicate quote for {symbol}."
            )
        bars[symbol] = DailyBar(expected_date, high, low, close, volume)
    return bars


def _parse_tpex_history(
    payload: Any,
    *,
    symbol: str,
    through_date: date,
) -> list[DailyBar]:
    if not isinstance(payload, dict):
        raise ValueError("TPEx historical API returned an invalid response.")
    if payload.get("stat") == "查無資料":
        return []
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError(
            f"TPEx historical API returned no table for {symbol}: "
            f"{payload.get('stat', 'unknown response')}."
        )
    table = next(
        (
            item
            for item in tables
            if isinstance(item, dict)
            and isinstance(item.get("data"), list)
        ),
        None,
    )
    if table is None:
        raise ValueError(f"TPEx historical API returned no price table for {symbol}.")
    fields = table.get("fields")
    rows = table.get("data")
    if not isinstance(fields, list) or not isinstance(rows, list):
        raise ValueError(f"TPEx historical price table is malformed for {symbol}.")
    indexes = {str(name).strip(): index for index, name in enumerate(fields)}
    required = ("日 期", "最高", "最低", "收盤", "成交張數")
    if any(name not in indexes for name in required):
        raise ValueError(
            f"TPEx historical price table is missing OHLCV fields for {symbol}."
        )
    bars: list[DailyBar] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < len(fields):
            continue
        date_text = str(row[indexes["日 期"]]).strip()
        match = re.fullmatch(r"(\d{2,3})/(\d{1,2})/(\d{1,2})\*?", date_text)
        if not match:
            raise ValueError(
                f"TPEx historical API returned invalid date {date_text!r} for {symbol}."
            )
        roc_year, month, day = (int(part) for part in match.groups())
        try:
            bar_date = date(roc_year + 1911, month, day)
        except ValueError as error:
            raise ValueError(
                f"TPEx historical API returned invalid date {date_text!r} for {symbol}."
            ) from error
        if bar_date > through_date:
            continue
        raw_values = [
            row[indexes[name]]
            for name in ("最高", "最低", "收盤", "成交張數")
        ]
        if any(str(value).strip() in {"", "--", "-"} for value in raw_values):
            continue
        high = _decimal(raw_values[0], field="high", symbol=symbol, venue="TPEx")
        low = _decimal(raw_values[1], field="low", symbol=symbol, venue="TPEx")
        close = _decimal(raw_values[2], field="close", symbol=symbol, venue="TPEx")
        volume = _decimal(raw_values[3], field="volume", symbol=symbol, venue="TPEx")
        if high <= 0 or low <= 0 or close <= 0:
            continue
        bars.append(DailyBar(bar_date, high, low, close, volume))
    return bars


def _month_offsets(through_date: date, max_calendar_days: int) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for offset in range(max_calendar_days):
        current = through_date - timedelta(days=offset)
        key = (current.year - 1911, current.month)
        if key not in seen:
            months.append(key)
            seen.add(key)
    return months


def _fetch_tpex_month(
    symbol: str,
    year: int,
    month: int,
    through_date: date,
) -> list[DailyBar]:
    query = urlencode(
        {
            "code": symbol,
            "date": f"{year + 1911}/{month:02d}/01",
        }
    )
    payload = _request_json(
        f"{TPEX_HISTORY_URL}?{query}",
        venue="TPEx",
    )
    return _parse_tpex_history(payload, symbol=symbol, through_date=through_date)


def _fetch_tpex_history(
    symbol: str,
    *,
    through_date: date,
    sessions: int,
    max_calendar_days: int,
) -> list[DailyBar]:
    bars_by_date: dict[date, DailyBar] = {}
    for year, month in _month_offsets(through_date, max_calendar_days):
        for bar in _fetch_tpex_month(symbol, year, month, through_date):
            bars_by_date[bar.date] = bar
        if len(bars_by_date) >= sessions:
            break
    return sorted(bars_by_date.values(), key=lambda bar: bar.date)[-sessions:]


def _fetch_twse_history(
    symbols: set[str],
    *,
    through_date: date,
    sessions: int,
    max_calendar_days: int,
) -> dict[str, list[DailyBar]]:
    history: dict[str, list[DailyBar]] = {symbol: [] for symbol in symbols}
    for offset in range(max_calendar_days):
        day = through_date - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        twse_url = (
            f"{TWSE_DAILY_URL}?date={day:%Y%m%d}"
            "&type=ALLBUT0999&response=json"
        )
        payload = _request_json(twse_url, venue="TWSE")
        bars = _parse_twse_history(
            payload,
            expected_date=day,
            requested_symbols=symbols,
        )
        for symbol, bar in bars.items():
            history[symbol].append(bar)
        if all(len(bars) >= sessions for bars in history.values()):
            break
        sleep(0.1)
    return history


def fetch_official_price_history(
    symbols: set[str],
    *,
    through_date: date,
    tpex_symbols: set[str] | None = None,
    sessions: int = HISTORY_SESSIONS,
    max_calendar_days: int = 120,
) -> dict[str, list[DailyBar]]:
    if not symbols:
        raise ValueError("Cannot fetch historical prices for an empty symbol list.")
    if sessions < MIN_TECHNICAL_SESSIONS:
        raise ValueError(
            "Historical price window must contain at least "
            f"{MIN_TECHNICAL_SESSIONS} sessions."
        )
    tpex = set(tpex_symbols or ()) & symbols
    twse = symbols - tpex
    history: dict[str, list[DailyBar]] = {symbol: [] for symbol in symbols}
    if twse:
        history.update(
            _fetch_twse_history(
                twse,
                through_date=through_date,
                sessions=sessions,
                max_calendar_days=max_calendar_days,
            )
        )
    if tpex:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    _fetch_tpex_history,
                    symbol,
                    through_date=through_date,
                    sessions=sessions,
                    max_calendar_days=max_calendar_days,
                ): symbol
                for symbol in sorted(tpex)
            }
            for future in as_completed(futures):
                symbol = futures[future]
                history[symbol] = future.result()

    incomplete = sorted(
        symbol
        for symbol, bars in history.items()
        if len(bars) < MIN_TECHNICAL_SESSIONS
    )
    if incomplete:
        counts = ", ".join(
            f"{symbol} ({len(history[symbol])})" for symbol in incomplete
        )
        raise ValueError(
            f"Could not retrieve {MIN_TECHNICAL_SESSIONS} official daily bars "
            f"for: {counts}."
        )
    return {
        symbol: sorted(
            {bar.date: bar for bar in bars}.values(),
            key=lambda bar: bar.date,
        )[-sessions:]
        for symbol, bars in history.items()
    }

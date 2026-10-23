from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from ai_cup_agent.models import Portfolio, Quote
from ai_cup_agent.rules import LOT_SIZE, cash_after_order


def _resolve_ref(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported external JSON Schema reference: {ref}")
    value: Any = root
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or token not in value:
            raise ValueError(f"Unresolvable JSON Schema reference: {ref}")
        value = value[token]
    if not isinstance(value, dict):
        raise ValueError(f"JSON Schema reference is not an object: {ref}")
    return value


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return (
            isinstance(value, (int, float, Decimal))
            and not isinstance(value, bool)
            and Decimal(str(value)).is_finite()
        )
    raise ValueError(f"Unsupported JSON Schema type: {type_name}")


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, (int, float, Decimal)) and not isinstance(left, bool):
        if isinstance(right, (int, float, Decimal)) and not isinstance(right, bool):
            return Decimal(str(left)) == Decimal(str(right))
    return left == right


def _validate_value(
    value: Any,
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    reference = schema.get("$ref")
    if reference is not None:
        _validate_value(value, _resolve_ref(root, reference), root, path, errors)
        return

    expected_type = schema.get("type")
    types = expected_type if isinstance(expected_type, list) else [expected_type]
    if expected_type is not None and not any(_matches_type(value, item) for item in types):
        errors.append(f"{path}: expected type {' or '.join(types)}.")
        return
    if "const" in schema and not _json_equal(value, schema["const"]):
        errors.append(f"{path}: must equal {schema['const']!r}.")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        errors.append(f"{path}: value is not one of the allowed options.")

    if isinstance(value, str):
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: does not match required pattern {schema['pattern']}.")
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: is shorter than minLength.")
        if len(value) > schema.get("maxLength", float("inf")):
            errors.append(f"{path}: is longer than maxLength.")
        if schema.get("format") == "uri":
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                errors.append(f"{path}: must be an absolute HTTP(S) URI.")

    if isinstance(value, dict):
        if len(value) < schema.get("minProperties", 0):
            errors.append(f"{path}: has fewer than minProperties.")
        if len(value) > schema.get("maxProperties", float("inf")):
            errors.append(f"{path}: has more than maxProperties.")
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}: missing required property {required!r}.")
        properties = schema.get("properties", {})
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                _validate_value(item, properties[key], root, child_path, errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{child_path}: additional property is not allowed.")
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate_value(
                    item, schema["additionalProperties"], root, child_path, errors
                )

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: has fewer than minItems.")
        if len(value) > schema.get("maxItems", float("inf")):
            errors.append(f"{path}: has more than maxItems.")
        if schema.get("uniqueItems"):
            for index, item in enumerate(value):
                if any(_json_equal(item, previous) for previous in value[:index]):
                    errors.append(f"{path}[{index}]: duplicate array item.")
                    break
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_value(item, item_schema, root, f"{path}[{index}]", errors)

    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            errors.append(f"{path}: invalid number.")
            return
        if not number.is_finite():
            errors.append(f"{path}: number must be finite.")
            return
        if "minimum" in schema and number < Decimal(str(schema["minimum"])):
            errors.append(f"{path}: is below the minimum.")
        if "maximum" in schema and number > Decimal(str(schema["maximum"])):
            errors.append(f"{path}: is above the maximum.")
        if "multipleOf" in schema:
            multiple = Decimal(str(schema["multipleOf"]))
            if number % multiple:
                errors.append(f"{path}: is not a multiple of {multiple}.")


def validate_dplan_schema(
    dplan: Any,
    schema_path: Path,
) -> list[str]:
    with schema_path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise ValueError("D-Plan schema file must contain a JSON object.")
    errors: list[str] = []
    _validate_value(dplan, schema, schema, "$", errors)
    return errors


def validate_dplan_semantics(
    dplan: Mapping[str, Any],
    *,
    portfolio: Portfolio,
    quotes: Mapping[str, Quote],
    nav: Decimal,
    allowed_symbols: set[str],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(dplan, Mapping):
        return ["D-Plan must be a JSON object."]
    sources = dplan.get("sources", [])
    observations = dplan.get("observations", [])
    inferences = dplan.get("inferences", [])
    decisions = dplan.get("decisions", [])
    no_trade = dplan.get("no_trade_decisions", [])
    orders = dplan.get("orders", [])
    market_view = dplan.get("market_view", {})
    if not all(
        isinstance(items, list)
        for items in (sources, observations, inferences, decisions, no_trade, orders)
    ):
        return ["D-Plan reference layers must be arrays."]

    for prefix, items, field in (
        ("S", sources, "source_id"),
        ("O", observations, "obs_id"),
        ("I", inferences, "inf_id"),
        ("D", decisions, "decision_id"),
    ):
        identifiers = [
            row.get(field)
            for row in items
            if isinstance(row, dict) and isinstance(row.get(field), str)
        ]
        expected_ids = {f"{prefix}{number}" for number in range(1, len(identifiers) + 1)}
        if len(identifiers) != len(set(identifiers)) or set(identifiers) != expected_ids:
            errors.append(f"{prefix} identifiers must start at 1 and be consecutive.")

    source_ids = {row.get("source_id") for row in sources if isinstance(row, dict)}
    observation_ids = {
        row.get("obs_id") for row in observations if isinstance(row, dict)
    }
    inference_ids = {row.get("inf_id") for row in inferences if isinstance(row, dict)}
    decision_ids = {row.get("decision_id") for row in decisions if isinstance(row, dict)}
    for row in observations:
        if isinstance(row, dict):
            for ref in row.get("source_ref", []):
                if ref not in source_ids:
                    errors.append(f"Observation references missing source {ref}.")
    for ref in market_view.get("basis_refs", []) if isinstance(market_view, dict) else []:
        if ref not in observation_ids:
            errors.append(f"Market view references missing observation {ref}.")
    for row in inferences:
        if isinstance(row, dict):
            for ref in row.get("premise_refs", []):
                if ref not in observation_ids:
                    errors.append(f"Inference references missing observation {ref}.")
    for row in decisions:
        if isinstance(row, dict):
            for ref in row.get("inference_refs", []):
                if ref not in inference_ids:
                    errors.append(f"Decision references missing inference {ref}.")
    for row in no_trade:
        if isinstance(row, dict):
            for ref in row.get("reason_refs", []):
                if ref not in inference_ids:
                    errors.append(f"No-trade decision references missing inference {ref}.")

    initial = {position.symbol: position.shares for position in portfolio.positions}
    decision_by_ticker: dict[str, dict[str, Any]] = {}
    for row in decisions:
        if not isinstance(row, dict):
            continue
        ticker = row.get("ticker")
        if ticker in decision_by_ticker:
            errors.append(f"More than one decision for {ticker}.")
        decision_by_ticker[ticker] = row
    no_trade_by_ticker: dict[str, dict[str, Any]] = {}
    for row in no_trade:
        if not isinstance(row, dict):
            continue
        ticker = row.get("ticker")
        if ticker in no_trade_by_ticker:
            errors.append(f"More than one no-trade decision for {ticker}.")
        no_trade_by_ticker[ticker] = row
    overlap = decision_by_ticker.keys() & no_trade_by_ticker.keys()
    if overlap:
        errors.append("Tickers cannot appear in both decisions and no-trade decisions: "
                      + ", ".join(sorted(overlap)))
    covered = decision_by_ticker.keys() | no_trade_by_ticker.keys()
    missing_coverage = initial.keys() - covered
    if missing_coverage:
        errors.append("Starting holdings have no decision coverage: "
                      + ", ".join(sorted(missing_coverage)))
    cited_inference_ids = {
        ref
        for row in decisions + no_trade
        if isinstance(row, dict)
        for ref in row.get("inference_refs", row.get("reason_refs", []))
    }
    if cited_inference_ids != inference_ids:
        errors.append("Every inference must be referenced by a decision or no-trade decision.")

    computed_orders: dict[str, tuple[str, int]] = {}
    for ticker, row in decision_by_ticker.items():
        if ticker not in quotes:
            errors.append(f"Missing closing price for decision ticker {ticker}.")
            continue
        if ticker not in allowed_symbols:
            errors.append(f"Decision ticker {ticker} is outside the eligible universe.")
        old_shares = initial.get(ticker, 0)
        action = row.get("action")
        weight = Decimal(str(row.get("target_weight", 0)))
        if ticker != "2330" and weight > Decimal("0.10"):
            errors.append(f"{ticker} target weight exceeds 10%.")
        if ticker == "2330" and weight > Decimal("0.25"):
            errors.append("2330 target weight exceeds 25%.")
        target_shares = int(
            (
                weight * nav / quotes[ticker].price / LOT_SIZE
            ).to_integral_value(rounding=ROUND_FLOOR)
        ) * LOT_SIZE
        if action == "BUY" and old_shares != 0:
            errors.append(f"BUY decision for {ticker} must be a new holding.")
        if action == "ADD" and (old_shares == 0 or target_shares <= old_shares):
            errors.append(f"ADD decision for {ticker} must increase an existing position.")
        if action == "TRIM" and not 0 < target_shares < old_shares:
            errors.append(f"TRIM decision for {ticker} must reduce but retain shares.")
        if action == "SELL_ALL" and (target_shares != 0 or old_shares == 0):
            errors.append(f"SELL_ALL decision for {ticker} must exit an existing position.")
        delta = target_shares - old_shares
        if delta:
            computed_orders[ticker] = ("BUY" if delta > 0 else "SELL", abs(delta))

    for ticker in no_trade_by_ticker:
        if initial.get(ticker, 0) <= 0:
            errors.append(f"No-trade ticker {ticker} is not a starting holding.")
        if ticker in decision_by_ticker:
            continue

    actual_orders: dict[str, tuple[str, int]] = {}
    for order in orders:
        if not isinstance(order, dict):
            continue
        ticker = order.get("ticker")
        if ticker in actual_orders:
            errors.append(f"More than one order for {ticker}.")
        actual_orders[ticker] = (order.get("side"), order.get("shares"))
        if order.get("decision_ref") not in decision_ids:
            errors.append(f"Order for {ticker} has no matching decision.")
    if actual_orders != computed_orders:
        errors.append("Orders do not match the official target-weight derivation.")

    decision_by_id = {
        row.get("decision_id"): row for row in decisions if isinstance(row, dict)
    }
    sell_rows = {
        ticker: row
        for ticker, row in decision_by_ticker.items()
        if row.get("action") in {"TRIM", "SELL_ALL"}
    }
    funding_value = Decimal("0")
    funding_marked_tickers: set[str] = set()
    for ticker, row in sell_rows.items():
        funding_refs = row.get("funding_for")
        if funding_refs is None:
            continue
        funding_marked_tickers.add(ticker)
        for ref in funding_refs:
            funded = decision_by_id.get(ref)
            if not isinstance(funded, dict) or funded.get("action") not in {"BUY", "ADD"}:
                errors.append(f"Funding reference {ref} must point to a BUY or ADD decision.")
    for ticker in funding_marked_tickers:
        row = sell_rows[ticker]
        if ticker in quotes:
            old_shares = initial.get(ticker, 0)
            new_shares = int(
                (
                    Decimal(str(row["target_weight"]))
                    * nav
                    / quotes[ticker].price
                    / LOT_SIZE
                ).to_integral_value(rounding=ROUND_FLOOR)
            ) * LOT_SIZE
            funding_value += (
                Decimal(max(old_shares - new_shares, 0)) * quotes[ticker].price
            )

    if any(
        not isinstance(ticker, str) or not re.fullmatch(r"[0-9]{4}", ticker)
        for ticker in decision_by_ticker
    ):
        errors.append("D-Plan decisions must use four-digit stock tickers.")
    target_positions: dict[str, int] = dict(initial)
    for ticker in initial.keys() | decision_by_ticker.keys():
        if ticker in decision_by_ticker and ticker in quotes:
            row = decision_by_ticker[ticker]
            weight = Decimal(str(row["target_weight"]))
            target_positions[ticker] = int(
                (
                    weight * nav / quotes[ticker].price / LOT_SIZE
                ).to_integral_value(rounding=ROUND_FLOOR)
            ) * LOT_SIZE
    target_positions = {
        ticker: shares for ticker, shares in target_positions.items() if shares > 0
    }
    if not 20 <= len(target_positions) <= 30:
        errors.append("Projected holding count must be between 20 and 30.")
    for ticker in target_positions:
        if ticker not in allowed_symbols:
            errors.append(f"Projected holding {ticker} is outside the eligible universe.")
        if target_positions[ticker] % LOT_SIZE:
            errors.append(f"Projected holding {ticker} is not a whole board lot.")

    buy_value = Decimal("0")
    sell_value = Decimal("0")
    estimated_cash = portfolio.cash
    for order in orders:
        if not isinstance(order, dict) or order.get("ticker") not in quotes:
            continue
        value = Decimal(order["shares"]) * quotes[order["ticker"]].price
        if order["side"] == "BUY":
            buy_value += value
        else:
            sell_value += value
        estimated_cash = cash_after_order(estimated_cash, order["side"], value)
    if estimated_cash < 0:
        errors.append("Projected cash after orders is negative.")
    if estimated_cash / nav > Decimal("0.25"):
        errors.append("Projected cash exceeds 25% of NAV.")
    posture = market_view.get("posture") if isinstance(market_view, dict) else None
    if funding_value and isinstance(posture, dict):
        cash_range = posture.get("target_cash_pct_range", [])
        if len(cash_range) == 2 and nav > 0:
            cash_without_funding_sells = estimated_cash - funding_value
            if cash_without_funding_sells / nav >= Decimal(str(cash_range[0])):
                errors.append(
                    "Funding-for declaration has no cash shortfall without its marked sells."
                )
    total_nav = estimated_cash + sum(
        (
            Decimal(shares) * quotes[ticker].price
            for ticker, shares in target_positions.items()
            if ticker in quotes
        ),
        Decimal("0"),
    )
    if total_nav <= 0:
        errors.append("Projected NAV must be positive.")
    else:
        for ticker, shares in target_positions.items():
            if ticker in quotes:
                limit = Decimal("0.25") if ticker == "2330" else Decimal("0.10")
                if Decimal(shares) * quotes[ticker].price / total_nav > limit:
                    errors.append(f"Projected weight for {ticker} exceeds {limit:.0%}.")

    if isinstance(market_view, dict) and isinstance(market_view.get("posture"), dict):
        posture = market_view["posture"]
        cash_range = posture.get("target_cash_pct_range", [])
        if len(cash_range) == 2 and cash_range[0] > cash_range[1]:
            errors.append("Market-view target cash range lower bound exceeds upper bound.")
        if len(cash_range) == 2 and nav > 0:
            cash_ratio = estimated_cash / nav
            if not Decimal(str(cash_range[0])) <= cash_ratio <= Decimal(str(cash_range[1])):
                errors.append("Projected cash ratio is outside the market-view range.")
        intent = posture.get("net_exposure_intent")
        net_flow = buy_value - sell_value
        if intent == "increase" and not net_flow > 0:
            errors.append("Increase posture requires net buying.")
        elif intent == "reduce" and not net_flow < 0:
            errors.append("Reduce posture requires net selling.")
        elif intent == "hold" and abs(net_flow) > nav * Decimal("0.02"):
            errors.append("Hold posture exceeds the 2% NAV net-flow tolerance.")

    return errors


def validate_dplan(
    dplan: Any,
    *,
    schema_path: Path,
    portfolio: Portfolio,
    quotes: Mapping[str, Quote],
    nav: Decimal,
    allowed_symbols: set[str],
) -> list[str]:
    schema_errors = validate_dplan_schema(dplan, schema_path)
    if schema_errors:
        return schema_errors
    if not isinstance(dplan, dict):
        return ["D-Plan must be a JSON object."]
    return validate_dplan_semantics(
        dplan,
        portfolio=portfolio,
        quotes=quotes,
        nav=nav,
        allowed_symbols=allowed_symbols,
    )

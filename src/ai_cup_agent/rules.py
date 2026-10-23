from __future__ import annotations

from decimal import Decimal
from typing import Mapping

from ai_cup_agent.models import Portfolio

ZERO = Decimal("0")
ONE = Decimal("1")
LOT_SIZE = 1000
COMMISSION_RATE = Decimal("0.001425")
SELL_TAX_RATE = Decimal("0.003")
MAX_CASH_WEIGHT = Decimal("0.25")
MAX_TSMC_WEIGHT = Decimal("0.25")
MAX_OTHER_WEIGHT = Decimal("0.10")
MIN_HOLDINGS = 20
MAX_HOLDINGS = 30
MIN_ACTIVE_SHARE = Decimal("0.15")


def portfolio_nav(portfolio: Portfolio, prices: Mapping[str, Decimal]) -> Decimal:
    missing = sorted(
        position.symbol for position in portfolio.positions if position.symbol not in prices
    )
    if missing:
        raise ValueError(f"Missing prices for held symbols: {', '.join(missing)}")
    return portfolio.cash + sum(
        (Decimal(position.shares) * prices[position.symbol] for position in portfolio.positions),
        ZERO,
    )


def cash_after_order(cash: Decimal, side: str, notional: Decimal) -> Decimal:
    commission = notional * COMMISSION_RATE
    if side == "BUY":
        return cash - notional - commission
    if side == "SELL":
        return cash + notional - commission - notional * SELL_TAX_RATE
    raise ValueError(f"Unsupported order side: {side}.")


def active_share(
    portfolio_weights: Mapping[str, Decimal],
    benchmark_weights: Mapping[str, Decimal],
) -> Decimal:
    symbols = portfolio_weights.keys() | benchmark_weights.keys()
    return sum(
        (abs(portfolio_weights.get(symbol, ZERO) - benchmark_weights.get(symbol, ZERO))
         for symbol in symbols),
        ZERO,
    ) / Decimal("2")


def _top_weights(
    portfolio: Portfolio, prices: Mapping[str, Decimal], nav: Decimal
) -> dict[str, Decimal]:
    holdings = sorted(
        portfolio.positions,
        key=lambda position: position.shares * prices[position.symbol],
        reverse=True,
    )[:15]
    return {
        position.symbol: Decimal(position.shares) * prices[position.symbol] / nav
        for position in holdings
    }


def validate_benchmarks(
    benchmarks: Mapping[str, Mapping[str, Decimal]],
    expected_benchmarks: set[str],
) -> list[str]:
    errors: list[str] = []
    missing = sorted(expected_benchmarks - benchmarks.keys())
    unexpected = sorted(benchmarks.keys() - expected_benchmarks)
    if missing:
        errors.append(f"Missing Active Share benchmark data for: {', '.join(missing)}.")
    if unexpected:
        errors.append(f"Unexpected Active Share benchmarks: {', '.join(unexpected)}.")

    for benchmark_name in expected_benchmarks & benchmarks.keys():
        weights = benchmarks[benchmark_name]
        if not 1 <= len(weights) <= 15:
            errors.append(
                f"Benchmark {benchmark_name} must contain between 1 and 15 top holdings."
            )
        if any(weight <= ZERO for weight in weights.values()):
            errors.append(f"Benchmark {benchmark_name} has a non-positive weight.")
        if sum(weights.values(), ZERO) > Decimal("1"):
            errors.append(f"Benchmark {benchmark_name} weights exceed 100% of NAV.")
    return errors


def validate_portfolio(
    portfolio: Portfolio,
    prices: Mapping[str, Decimal],
    allowed_symbols: set[str],
    benchmarks: Mapping[str, Mapping[str, Decimal]],
    expected_benchmarks: set[str],
) -> list[str]:
    errors = validate_benchmarks(benchmarks, expected_benchmarks)
    symbols = [position.symbol for position in portfolio.positions]
    if len(symbols) != len(set(symbols)):
        errors.append("Portfolio contains duplicate symbols.")
    if any(position.shares <= 0 for position in portfolio.positions):
        errors.append("All held share counts must be positive.")
    if any(position.shares % LOT_SIZE != 0 for position in portfolio.positions):
        errors.append("Every holding must be a whole board lot (1,000 shares).")
    if not MIN_HOLDINGS <= len(set(symbols)) <= MAX_HOLDINGS:
        errors.append(f"Holding count must be between {MIN_HOLDINGS} and {MAX_HOLDINGS}.")
    if not symbols or any(symbol not in allowed_symbols for symbol in symbols):
        errors.append("All holdings must belong to the official eligible-stock list.")
    if portfolio.cash < ZERO:
        errors.append("Cash cannot be negative.")

    try:
        nav = portfolio_nav(portfolio, prices)
    except ValueError as error:
        errors.append(str(error))
        return errors

    if nav <= ZERO:
        errors.append("NAV must be positive.")
        return errors
    if portfolio.cash / nav > MAX_CASH_WEIGHT:
        errors.append("Cash exceeds 25% of NAV.")
    for position in portfolio.positions:
        weight = Decimal(position.shares) * prices[position.symbol] / nav
        limit = MAX_TSMC_WEIGHT if position.symbol == "2330" else MAX_OTHER_WEIGHT
        if weight > limit:
            errors.append(
                f"{position.symbol} weight {weight:.4%} exceeds limit {limit:.0%}."
            )

    portfolio_weights = _top_weights(portfolio, prices, nav)
    for benchmark_name in expected_benchmarks & benchmarks.keys():
        weights = benchmarks[benchmark_name]
        if (
            not 1 <= len(weights) <= 15
            or any(weight <= ZERO for weight in weights.values())
            or sum(weights.values(), ZERO) > Decimal("1")
        ):
            continue
        share = active_share(portfolio_weights, weights)
        if share < MIN_ACTIVE_SHARE:
            errors.append(
                f"Active Share versus {benchmark_name} is {share:.2%}; "
                f"minimum is {MIN_ACTIVE_SHARE:.0%}."
            )
    return errors

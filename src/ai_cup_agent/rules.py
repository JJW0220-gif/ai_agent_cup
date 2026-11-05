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


def validate_portfolio(
    portfolio: Portfolio,
    prices: Mapping[str, Decimal],
    allowed_symbols: set[str],
) -> list[str]:
    errors: list[str] = []
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

    return errors

from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
from typing import Mapping

from ai_cup_agent.models import AgentDecision, Order, Portfolio, Position, Quote
from ai_cup_agent.rules import (
    LOT_SIZE,
    active_share,
    cash_after_order,
    portfolio_nav,
    validate_benchmarks,
    validate_portfolio,
)

ZERO = Decimal("0")
RETENTION_RANK = 30
REBALANCE_THRESHOLD = Decimal("0.005")
DAILY_TURNOVER_LIMIT = Decimal("0.20")


def _round_lots(value: Decimal, price: Decimal) -> int:
    shares = (value / price / LOT_SIZE).to_integral_value(rounding=ROUND_FLOOR)
    return int(shares) * LOT_SIZE


def _plan_trades(
    starting_positions: Mapping[str, int],
    target_shares: Mapping[str, int],
    prices: Mapping[str, Decimal],
    nav: Decimal,
    *,
    turnover_limit: Decimal | None,
    deviation_threshold: Decimal,
) -> tuple[list[Order], dict[str, int]]:
    current_shares = dict(starting_positions)
    symbols = set(starting_positions) | set(target_shares)
    orders: list[Order] = []
    remaining_turnover = turnover_limit

    for side in ("SELL", "BUY"):
        for symbol in sorted(symbols):
            old_shares = starting_positions.get(symbol, 0)
            new_shares = target_shares.get(symbol, 0)
            delta = new_shares - old_shares
            if (side == "SELL" and delta >= 0) or (side == "BUY" and delta <= 0):
                continue
            if (
                symbol in target_shares
                and abs(Decimal(delta) * prices[symbol]) < nav * deviation_threshold
            ):
                continue

            requested_lots = abs(delta) // LOT_SIZE
            if remaining_turnover is None:
                order_lots = requested_lots
            else:
                affordable_lots = int(
                    (remaining_turnover / prices[symbol] / LOT_SIZE).to_integral_value(
                        rounding=ROUND_FLOOR
                    )
                )
                order_lots = min(requested_lots, affordable_lots)
            if order_lots <= 0:
                continue

            traded_shares = order_lots * LOT_SIZE
            notional = Decimal(traded_shares) * prices[symbol]
            orders.append(Order(symbol=symbol, side=side, lots=order_lots))
            current_shares[symbol] = (
                old_shares - traded_shares if side == "SELL"
                else old_shares + traded_shares
            )
            if remaining_turnover is not None:
                remaining_turnover -= notional

    return orders, current_shares


def build_decision(
    portfolio: Portfolio,
    quotes: Mapping[str, Quote],
    allowed_symbols: set[str],
    benchmarks: Mapping[str, Mapping[str, Decimal]],
    *,
    expected_benchmarks: set[str],
    target_holdings: int = 22,
    target_cash_weight: Decimal = Decimal("0.10"),
) -> AgentDecision:
    if not 20 <= target_holdings <= 30:
        raise ValueError("target_holdings must be between 20 and 30.")
    if not ZERO <= target_cash_weight <= Decimal("0.25"):
        raise ValueError("target_cash_weight must be between 0 and 25%.")
    if any(quote.price <= ZERO for quote in quotes.values()):
        raise ValueError("All quote prices must be positive.")
    benchmark_errors = validate_benchmarks(benchmarks, expected_benchmarks)
    if benchmark_errors:
        raise ValueError("Invalid Active Share benchmark data:\n- " + "\n- ".join(benchmark_errors))

    prices = {symbol: quote.price for symbol, quote in quotes.items()}
    nav = portfolio_nav(portfolio, prices)
    if nav <= ZERO:
        raise ValueError("Starting NAV must be positive.")

    candidates = [
        quote for symbol, quote in quotes.items()
        if symbol in allowed_symbols and symbol != "2330"
    ]
    tsmc = quotes.get("2330")
    if tsmc is not None and "2330" in allowed_symbols:
        candidates.append(tsmc)
    if len(candidates) < target_holdings:
        raise ValueError(
            f"Need at least {target_holdings} eligible quotes; received {len(candidates)}."
        )

    ranked = sorted(
        candidates,
        key=lambda quote: (quote.score, quote.symbol),
        reverse=True,
    )
    rank_by_symbol = {
        quote.symbol: rank
        for rank, quote in enumerate(ranked, start=1)
    }
    starting_positions = {position.symbol: position.shares for position in portfolio.positions}
    retained = [
        quote
        for quote in ranked
        if quote.symbol in starting_positions
        and rank_by_symbol[quote.symbol] <= RETENTION_RANK
    ][:target_holdings]
    selected_symbols = {quote.symbol for quote in retained}
    selected = retained + [
        quote
        for quote in ranked
        if quote.symbol not in selected_symbols
    ][:target_holdings - len(retained)]
    target_invested_value = nav * (Decimal("1") - target_cash_weight)
    per_stock_target = target_invested_value / Decimal(target_holdings)
    target_shares = {
        quote.symbol: _round_lots(per_stock_target, quote.price)
        for quote in selected
    }
    if any(shares == 0 for shares in target_shares.values()):
        raise ValueError("NAV is too small to allocate at least one lot to every target holding.")

    turnover_limit = (
        None if not portfolio.positions else nav * DAILY_TURNOVER_LIMIT
    )

    def build_projected(
        limit: Decimal | None,
        threshold: Decimal,
    ) -> tuple[list[Order], Portfolio, list[str]]:
        planned_orders, resulting_shares = _plan_trades(
            starting_positions,
            target_shares,
            prices,
            nav,
            turnover_limit=limit,
            deviation_threshold=threshold,
        )
        cash = portfolio.cash
        for order in planned_orders:
            notional = Decimal(order.lots * LOT_SIZE) * prices[order.symbol]
            cash = cash_after_order(cash, order.side, notional)
        projected_portfolio = Portfolio(
            cash=cash,
            positions=tuple(
                Position(symbol=symbol, shares=shares)
                for symbol, shares in sorted(resulting_shares.items())
                if shares > 0
            ),
        )
        errors = validate_portfolio(
            projected_portfolio, prices, allowed_symbols, benchmarks, expected_benchmarks
        )
        if cash < ZERO:
            errors.append(f"Planned orders would make cash negative ({cash}).")
        return planned_orders, projected_portfolio, errors

    orders, projected, errors = build_projected(
        turnover_limit,
        REBALANCE_THRESHOLD,
    )
    turnover_exception = not portfolio.positions
    if errors and turnover_limit is not None:
        orders, projected, errors = build_projected(None, ZERO)
        turnover_exception = True
    if errors:
        raise ValueError("Projected portfolio failed rule checks:\n- " + "\n- ".join(errors))

    nav_after_costs = portfolio_nav(projected, prices)
    projected_weights = {
        position.symbol: Decimal(position.shares) * prices[position.symbol] / nav_after_costs
        for position in sorted(
            projected.positions,
            key=lambda item: item.shares * prices[item.symbol],
            reverse=True,
        )[:15]
    }
    shares_by_benchmark = {
        name: active_share(projected_weights, weights)
        for name, weights in benchmarks.items()
    }
    return AgentDecision(
        orders=tuple(orders),
        projected_portfolio=projected,
        nav=nav_after_costs,
        active_share_by_benchmark=shares_by_benchmark,
        rationale=(
            f"Ranked eligible stocks by supplied model score, retained existing "
            f"holdings through rank {RETENTION_RANK}, and selected {target_holdings} "
            f"holdings. Rebalanced toward {target_cash_weight:.0%} cash only when "
            f"an existing holding deviated by at least "
            f"{REBALANCE_THRESHOLD:.1%} of NAV. "
            + (
                "The daily turnover cap was waived for initial establishment or "
                "to restore competition-rule compliance. "
                if turnover_exception
                else f"Gross daily turnover was capped at "
                f"{DAILY_TURNOVER_LIMIT:.0%} of NAV. "
            )
            + "Validated the projected portfolio against configured competition "
            "constraints."
        ),
    )

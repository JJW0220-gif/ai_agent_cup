import unittest
from datetime import date, timedelta
from decimal import Decimal

from ai_cup_agent.models import Portfolio, Position, Quote
from ai_cup_agent.rules import validate_portfolio
from ai_cup_agent.strategy import build_decision
from ai_cup_agent.technicals import DailyBar, calculate_technical_indicators


class PortfolioValidationTests(unittest.TestCase):
    def test_accepts_valid_twenty_stock_portfolio(self) -> None:
        positions = tuple(
            Position(symbol=f"S{index:02}", shares=1000) for index in range(20)
        )
        prices = {position.symbol: Decimal("100") for position in positions}
        portfolio = Portfolio(cash=Decimal("500000"), positions=positions)
        errors = validate_portfolio(
            portfolio,
            prices,
            {position.symbol for position in positions},
        )

        self.assertEqual(errors, [])

    def test_rejects_cash_above_rule_limit(self) -> None:
        positions = tuple(
            Position(symbol=f"S{index:02}", shares=1000) for index in range(20)
        )
        prices = {position.symbol: Decimal("100") for position in positions}
        portfolio = Portfolio(cash=Decimal("900000"), positions=positions)
        errors = validate_portfolio(
            portfolio,
            prices,
            {position.symbol for position in positions},
        )
        self.assertTrue(any("Cash exceeds 25%" in error for error in errors))

class StrategyTests(unittest.TestCase):
    def test_uses_atr_to_adjust_target_position_sizes(self) -> None:
        symbols = {f"S{index:02}" for index in range(22)}
        quotes = {
            symbol: Quote(symbol, Decimal("100"), Decimal(str(index)))
            for index, symbol in enumerate(sorted(symbols))
        }
        indicators = {}
        for symbol in symbols:
            daily_range = Decimal("5") if symbol == "S00" else Decimal("0.2")
            bars = [
                DailyBar(
                    date(2026, 1, 1) + timedelta(days=index),
                    Decimal("100") + daily_range,
                    Decimal("100") - daily_range,
                    Decimal("100"),
                    Decimal("10000"),
                )
                for index in range(60)
            ]
            indicators[symbol] = calculate_technical_indicators(bars)
        starting = Portfolio(cash=Decimal("100000000"), positions=())

        decision = build_decision(
            starting,
            quotes,
            symbols,
            technical_indicators=indicators,
        )

        shares = {
            position.symbol: position.shares
            for position in decision.projected_portfolio.positions
        }
        self.assertGreater(shares["S01"], shares["S00"])

    def test_keeps_existing_holding_through_rank_thirty(self) -> None:
        symbols = {f"S{index:02}" for index in range(33)}
        quotes = {
            symbol: Quote(symbol, Decimal("100"), Decimal(symbol[1:]))
            for symbol in symbols
        }
        held_symbols = {"S09", "S01"} | {
            f"S{index:02}" for index in range(11, 31)
        }
        starting = Portfolio(
            cash=Decimal("500000"),
            positions=tuple(
                Position(symbol, 1000) for symbol in sorted(held_symbols)
            ),
        )

        decision = build_decision(
            starting,
            quotes,
            symbols,
        )

        selected = {
            position.symbol for position in decision.projected_portfolio.positions
        }
        self.assertEqual(len(selected), 22)
        self.assertIn("S09", selected)
        self.assertNotIn("S10", selected)
        self.assertNotIn("S01", selected)
        self.assertIn("S32", selected)

    def test_caps_replacement_turnover_at_twenty_percent_of_nav(self) -> None:
        symbols = {f"S{index:02}" for index in range(33)}
        quotes = {
            symbol: Quote(symbol, Decimal("100"), Decimal(symbol[1:]))
            for symbol in symbols
        }
        held_symbols = {"S00", "S01", "S02"} | {
            f"S{index:02}" for index in range(14, 33)
        }
        starting = Portfolio(
            cash=Decimal("12000000"),
            positions=tuple(
                Position(symbol, 40000) for symbol in sorted(held_symbols)
            ),
        )

        decision = build_decision(
            starting,
            quotes,
            symbols,
        )

        turnover = sum(
            (
                Decimal(order.lots * 1000) * quotes[order.symbol].price
                for order in decision.orders
            ),
            Decimal("0"),
        )
        self.assertLessEqual(turnover, Decimal("20000000"))
        self.assertLess(turnover, Decimal("24000000"))
        self.assertIn("Gross daily turnover was capped at 20% of NAV", decision.rationale)

    def test_defers_rebalance_instead_of_exceeding_turnover_cap(self) -> None:
        new_symbols = {f"N{index:02}" for index in range(10)}
        retained_symbols = {f"H{index:02}" for index in range(10, 22)}
        displaced_symbols = {f"H{index:02}" for index in range(10)}
        other_symbols = {f"M{index:02}" for index in range(20)}
        symbols = new_symbols | retained_symbols | displaced_symbols | other_symbols
        scores = {
            **{symbol: Decimal(100 + index) for index, symbol in enumerate(new_symbols)},
            **{symbol: Decimal(80 + index) for index, symbol in enumerate(retained_symbols)},
            **{symbol: Decimal(30 + index) for index, symbol in enumerate(other_symbols)},
            **{symbol: Decimal(index) for index, symbol in enumerate(displaced_symbols)},
        }
        quotes = {
            symbol: Quote(symbol, Decimal("100"), score)
            for symbol, score in scores.items()
        }
        starting = Portfolio(
            cash=Decimal("12000000"),
            positions=tuple(
                Position(symbol, 40000)
                for symbol in sorted(retained_symbols | displaced_symbols)
            ),
        )

        decision = build_decision(starting, quotes, symbols)

        turnover = sum(
            (
                Decimal(order.lots * 1000) * quotes[order.symbol].price
                for order in decision.orders
            ),
            Decimal("0"),
        )
        self.assertLessEqual(turnover, Decimal("20000000"))
        self.assertEqual(decision.orders, ())
        self.assertEqual(decision.projected_portfolio, starting)
        self.assertIn("all trades were deferred", decision.rationale)

    def test_ranks_only_by_score_without_benchmark_overlap_penalty(self) -> None:
        symbols = {f"S{index:02}" for index in range(23)}
        scores = {
            symbol: Decimal(str(index))
            for index, symbol in enumerate(sorted(symbols))
        }
        scores["S00"] = Decimal("1")
        scores["S01"] = Decimal("0")
        quotes = {
            symbol: Quote(symbol, Decimal("100"), score)
            for symbol, score in scores.items()
        }
        starting = Portfolio(
            cash=Decimal("100000000"),
            positions=(),
        )

        decision = build_decision(
            starting,
            quotes,
            symbols,
        )

        selected = {
            position.symbol for position in decision.projected_portfolio.positions
        }
        self.assertIn("S00", selected)
        self.assertNotIn("S01", selected)

    def test_skips_small_rebalances_below_half_percent_of_nav(self) -> None:
        symbols = {f"S{index:02}" for index in range(22)}
        quotes = {
            symbol: Quote(symbol, Decimal("100"), Decimal(str(index)))
            for index, symbol in enumerate(sorted(symbols))
        }
        existing_symbols = {f"S{index:02}" for index in range(20)}
        starting = Portfolio(
            cash=Decimal("12000000"),
            positions=tuple(
                Position(symbol, 44000) for symbol in sorted(existing_symbols)
            ),
        )

        decision = build_decision(
            starting,
            quotes,
            symbols,
        )

        self.assertFalse(any(order.side == "SELL" for order in decision.orders))
        projected = {
            position.symbol: position.shares
            for position in decision.projected_portfolio.positions
        }
        self.assertTrue(
            all(projected[symbol] == 44000 for symbol in existing_symbols)
        )

    def test_generates_compliant_whole_lot_orders(self) -> None:
        symbols = {f"S{index:02}" for index in range(22)}
        quotes = {
            symbol: Quote(symbol, Decimal("100"), Decimal(str(index)))
            for index, symbol in enumerate(sorted(symbols))
        }
        starting = Portfolio(
            cash=Decimal("500000"),
            positions=tuple(
                Position(symbol, 1000) for symbol in sorted(symbols)[:20]
            ),
        )

        decision = build_decision(
            starting,
            quotes,
            symbols,
        )

        self.assertEqual(len(decision.projected_portfolio.positions), 22)
        self.assertTrue(
            all(position.shares % 1000 == 0 for position in decision.projected_portfolio.positions)
        )
        self.assertGreaterEqual(decision.projected_portfolio.cash, Decimal("0"))


if __name__ == "__main__":
    unittest.main()

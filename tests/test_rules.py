import unittest
from datetime import date
from decimal import Decimal

from ai_cup_agent.fetch_benchmarks import _extract_holdings_page
from ai_cup_agent.models import Portfolio, Position, Quote
from ai_cup_agent.rules import active_share, validate_portfolio
from ai_cup_agent.strategy import build_decision


class ActiveShareTests(unittest.TestCase):
    def test_computes_half_sum_of_absolute_weight_differences(self) -> None:
        result = active_share(
            {"2330": Decimal("0.2"), "2317": Decimal("0.1")},
            {"2330": Decimal("0.1"), "2454": Decimal("0.2")},
        )
        self.assertEqual(result, Decimal("0.2"))

    def test_parses_snapshot_date_and_holdings_from_page(self) -> None:
        holdings = [("2330", "12.5"), ("2317", "7.25")] + [
            (f"{index:04}", "1.0") for index in range(1000, 1013)
        ]
        rows = "".join(
            f"<tr><td>{code}台灣股票</td><td>{weight}%</td></tr>"
            for code, weight in holdings
        )
        html = (
            "<html><head><title>00980A holdings 2026-10-20</title></head>"
            "<body><table><tr><th>代號</th><th>權重</th></tr>"
            f"{rows}</table></body></html>"
        )

        result = _extract_holdings_page(
            html,
            "00980A",
            today=date(2026, 10, 21),
            max_age_days=5,
        )

        self.assertEqual(result["as_of"], "2026-10-20")
        self.assertEqual(result["weights"]["2330"], "0.125")
        self.assertEqual(result["weights"]["2317"], "0.0725")
        self.assertEqual(result["source_url"], "https://www.etfinfo.tw/etf/00980A/holdings")

    def test_rejects_stale_snapshot(self) -> None:
        html = (
            "<html><head><title>00980A holdings 2026-10-01</title></head>"
            "<table><tr><th>代號</th><th>權重</th></tr>"
            "<tr><td>2330 台積電</td><td>12%</td></tr></table></html>"
        )

        with self.assertRaisesRegex(ValueError, "days old"):
            _extract_holdings_page(
                html,
                "00980A",
                today=date(2026, 10, 21),
                max_age_days=5,
            )

    def test_rejects_incomplete_top_fifteen(self) -> None:
        html = (
            "<html><head><title>00980A holdings 2026-10-20</title></head>"
            "<table><tr><th>代號</th><th>權重</th></tr>"
            "<tr><td>2330台積電</td><td>12%</td></tr></table></html>"
        )

        with self.assertRaisesRegex(ValueError, "exactly 15"):
            _extract_holdings_page(
                html,
                "00980A",
                today=date(2026, 10, 21),
                max_age_days=5,
            )


class PortfolioValidationTests(unittest.TestCase):
    def test_accepts_valid_twenty_stock_portfolio(self) -> None:
        positions = tuple(
            Position(symbol=f"S{index:02}", shares=1000) for index in range(20)
        )
        prices = {position.symbol: Decimal("100") for position in positions}
        portfolio = Portfolio(cash=Decimal("500000"), positions=positions)
        benchmarks = {"ETF-A": {"OUTSIDE": Decimal("0.5")}}

        errors = validate_portfolio(
            portfolio,
            prices,
            {position.symbol for position in positions},
            benchmarks,
            {"ETF-A"},
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
            {"ETF-A": {"OUTSIDE": Decimal("0.5")}},
            {"ETF-A"},
        )
        self.assertTrue(any("Cash exceeds 25%" in error for error in errors))

    def test_rejects_missing_active_etf_benchmark(self) -> None:
        positions = tuple(
            Position(symbol=f"S{index:02}", shares=1000) for index in range(20)
        )
        prices = {position.symbol: Decimal("100") for position in positions}
        portfolio = Portfolio(cash=Decimal("500000"), positions=positions)

        errors = validate_portfolio(
            portfolio,
            prices,
            {position.symbol for position in positions},
            {"ETF-A": {"OUTSIDE": Decimal("0.5")}},
            {"ETF-A", "ETF-B"},
        )

        self.assertTrue(any("Missing Active Share benchmark data for: ETF-B" in error for error in errors))

    def test_rejects_benchmark_with_more_than_fifteen_holdings(self) -> None:
        weights = {f"S{index:02}": Decimal("0.01") for index in range(16)}
        positions = tuple(
            Position(symbol=f"P{index:02}", shares=1000) for index in range(20)
        )
        prices = {position.symbol: Decimal("100") for position in positions}
        prices.update({symbol: Decimal("100") for symbol in weights})
        portfolio = Portfolio(cash=Decimal("500000"), positions=positions)

        errors = validate_portfolio(
            portfolio,
            prices,
            {position.symbol for position in positions},
            {"ETF-A": weights},
            {"ETF-A"},
        )

        self.assertTrue(any("between 1 and 15 top holdings" in error for error in errors))


class StrategyTests(unittest.TestCase):
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
            cash=Decimal("100000000"),
            positions=tuple(
                Position(symbol, 1000) for symbol in sorted(held_symbols)
            ),
        )

        decision = build_decision(
            starting,
            quotes,
            symbols,
            {"ETF-A": {"OUTSIDE": Decimal("0.5")}},
            expected_benchmarks={"ETF-A"},
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
            {"ETF-A": {"OUTSIDE": Decimal("0.5")}},
            expected_benchmarks={"ETF-A"},
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
            {"ETF-A": {"S00": Decimal("1")}},
            expected_benchmarks={"ETF-A"},
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
            {"ETF-A": {"OUTSIDE": Decimal("0.5")}},
            expected_benchmarks={"ETF-A"},
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
            cash=Decimal("100000000"),
            positions=tuple(
                Position(symbol, 1000) for symbol in sorted(symbols)[:20]
            ),
        )

        decision = build_decision(
            starting,
            quotes,
            symbols,
            {"ETF-A": {"OUTSIDE": Decimal("0.5")}},
            expected_benchmarks={"ETF-A"},
        )

        self.assertEqual(len(decision.projected_portfolio.positions), 22)
        self.assertTrue(
            all(position.shares % 1000 == 0 for position in decision.projected_portfolio.positions)
        )
        self.assertGreaterEqual(
            decision.active_share_by_benchmark["ETF-A"], Decimal("0.15")
        )
        self.assertGreaterEqual(decision.projected_portfolio.cash, Decimal("0"))


if __name__ == "__main__":
    unittest.main()

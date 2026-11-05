from __future__ import annotations

import unittest
from datetime import date, timedelta
from decimal import Decimal

from ai_cup_agent.technicals import (
    DailyBar,
    atr_adjusted_weights,
    calculate_technical_indicators,
)


class TechnicalIndicatorTests(unittest.TestCase):
    @staticmethod
    def bars(*, daily_range: Decimal) -> list[DailyBar]:
        start = date(2026, 1, 1)
        result = []
        for index in range(60):
            close = Decimal("100") + Decimal(index) / Decimal("10")
            result.append(
                DailyBar(
                    start + timedelta(days=index),
                    close + daily_range,
                    close - daily_range,
                    close,
                    Decimal("10000"),
                )
            )
        return result

    def test_calculates_macd_rsi_atr_and_moving_averages(self) -> None:
        indicator = calculate_technical_indicators(
            self.bars(daily_range=Decimal("1"))
        )

        self.assertEqual(indicator.rsi, Decimal("100"))
        self.assertGreater(indicator.macd, Decimal("0"))
        self.assertGreater(indicator.atr_pct, Decimal("0"))
        self.assertEqual(
            indicator.ma20,
            sum(
                (
                    Decimal("100") + Decimal(index) / Decimal("10")
                    for index in range(40, 60)
                ),
                Decimal("0"),
            )
            / Decimal("20"),
        )
        self.assertGreaterEqual(indicator.technical_score, Decimal("0"))
        self.assertLessEqual(indicator.technical_score, Decimal("100"))

    def test_atr_sizing_reduces_high_volatility_allocation_with_bounded_weights(self) -> None:
        indicators = {
            "LOW": calculate_technical_indicators(
                self.bars(daily_range=Decimal("0.2"))
            ),
            "HIGH": calculate_technical_indicators(
                self.bars(daily_range=Decimal("5"))
            ),
        }

        weights = atr_adjusted_weights(
            ["LOW", "HIGH"],
            indicators,
            total_weight=Decimal("0.90"),
        )

        self.assertGreater(weights["LOW"], weights["HIGH"])
        self.assertEqual(sum(weights.values()), Decimal("0.90"))
        self.assertGreaterEqual(weights["HIGH"], Decimal("0.90") / 2 * Decimal("0.75"))
        self.assertLessEqual(weights["LOW"], Decimal("0.90") / 2 * Decimal("1.25"))

    def test_scores_new_listings_without_unavailable_long_history_indicators(self) -> None:
        indicator = calculate_technical_indicators(
            self.bars(daily_range=Decimal("1"))[:33]
        )

        self.assertIsNone(indicator.ma50)
        self.assertIsNone(indicator.ma50_rising)
        self.assertFalse(indicator.bullish_divergence)
        self.assertFalse(indicator.bearish_divergence)
        self.assertGreaterEqual(indicator.technical_score, Decimal("0"))
        self.assertLessEqual(indicator.technical_score, Decimal("100"))

    def test_rejects_history_shorter_than_minimum_sessions(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 26 daily bars"):
            calculate_technical_indicators(
                self.bars(daily_range=Decimal("1"))[:25]
            )


if __name__ == "__main__":
    unittest.main()

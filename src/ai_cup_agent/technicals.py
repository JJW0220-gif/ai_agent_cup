from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Mapping, Sequence


ZERO = Decimal("0")
ONE = Decimal("1")
MIN_TECHNICAL_SESSIONS = 26
HISTORY_SESSIONS = 60


@dataclass(frozen=True)
class DailyBar:
    date: date
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class TechnicalIndicators:
    macd: Decimal
    signal: Decimal
    histogram: Decimal
    rsi: Decimal
    atr: Decimal
    atr_pct: Decimal
    ma20: Decimal
    ma50: Decimal | None
    ma20_rising: bool
    ma50_rising: bool | None
    bullish_divergence: bool
    bearish_divergence: bool
    macd_cross: str
    technical_score: Decimal


def _ema(values: Sequence[Decimal], period: int) -> list[Decimal]:
    alpha = Decimal("2") / Decimal(period + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append(value * alpha + result[-1] * (ONE - alpha))
    return result


def _rsi(closes: Sequence[Decimal], period: int = 14) -> Decimal:
    changes = [right - left for left, right in zip(closes, closes[1:])]
    gains = [max(change, ZERO) for change in changes]
    losses = [max(-change, ZERO) for change in changes]
    average_gain = sum(gains[:period], ZERO) / Decimal(period)
    average_loss = sum(losses[:period], ZERO) / Decimal(period)
    for gain, loss in zip(gains[period:], losses[period:]):
        average_gain = (
            average_gain * Decimal(period - 1) + gain
        ) / Decimal(period)
        average_loss = (
            average_loss * Decimal(period - 1) + loss
        ) / Decimal(period)
    if average_loss == ZERO:
        return Decimal("100") if average_gain else Decimal("50")
    relative_strength = average_gain / average_loss
    return Decimal("100") - Decimal("100") / (ONE + relative_strength)


def _atr(bars: Sequence[DailyBar], period: int = 14) -> Decimal:
    true_ranges: list[Decimal] = []
    previous_close: Decimal | None = None
    for bar in bars:
        high_low = bar.high - bar.low
        if previous_close is None:
            true_range = high_low
        else:
            true_range = max(
                high_low,
                abs(bar.high - previous_close),
                abs(bar.low - previous_close),
            )
        true_ranges.append(true_range)
        previous_close = bar.close
    average = sum(true_ranges[:period], ZERO) / Decimal(period)
    for true_range in true_ranges[period:]:
        average = (
            average * Decimal(period - 1) + true_range
        ) / Decimal(period)
    return average


def _divergences(
    bars: Sequence[DailyBar],
    macd_values: Sequence[Decimal],
) -> tuple[bool, bool]:
    lows: list[int] = []
    highs: list[int] = []
    left_right_bars = 3
    for index in range(left_right_bars, len(bars) - left_right_bars):
        neighborhood = bars[index - left_right_bars:index + left_right_bars + 1]
        if bars[index].low == min(bar.low for bar in neighborhood):
            lows.append(index)
        if bars[index].high == max(bar.high for bar in neighborhood):
            highs.append(index)

    bullish = bearish = False
    if len(lows) >= 2:
        previous, latest = lows[-2:]
        bullish = (
            latest - previous >= 5
            and bars[latest].low < bars[previous].low
            and macd_values[latest] > macd_values[previous]
        )
    if len(highs) >= 2:
        previous, latest = highs[-2:]
        bearish = (
            latest - previous >= 5
            and bars[latest].high > bars[previous].high
            and macd_values[latest] < macd_values[previous]
        )
    return bullish, bearish


def calculate_technical_indicators(
    bars: Sequence[DailyBar],
) -> TechnicalIndicators:
    ordered = sorted(bars, key=lambda bar: bar.date)
    if len(ordered) < MIN_TECHNICAL_SESSIONS:
        raise ValueError(
            f"Technical indicators require at least {MIN_TECHNICAL_SESSIONS} "
            f"daily bars; received {len(ordered)}."
        )
    if any(
        bar.close <= ZERO or bar.high < bar.low or bar.high <= ZERO
        for bar in ordered
    ):
        raise ValueError("Historical bars contain invalid high, low, or close prices.")
    closes = [bar.close for bar in ordered]
    fast_ema = _ema(closes, 12)
    slow_ema = _ema(closes, 26)
    macd_values = [fast - slow for fast, slow in zip(fast_ema, slow_ema)]
    signal_values = _ema(macd_values, 9)
    macd = macd_values[-1]
    signal = signal_values[-1]
    histogram = macd - signal
    rsi = _rsi(closes)
    atr = _atr(ordered)
    close = closes[-1]
    atr_pct = atr / close * Decimal("100")
    ma20 = sum(closes[-20:], ZERO) / Decimal("20")
    previous_ma20 = sum(closes[-21:-1], ZERO) / Decimal("20")
    ma20_rising = ma20 > previous_ma20
    ma50 = (
        sum(closes[-50:], ZERO) / Decimal("50")
        if len(closes) >= 50
        else None
    )
    ma50_rising = (
        ma50 > sum(closes[-51:-1], ZERO) / Decimal("50")
        if len(closes) >= 51 and ma50 is not None
        else None
    )
    if len(ordered) >= HISTORY_SESSIONS:
        bullish_divergence, bearish_divergence = _divergences(
            ordered, macd_values
        )
    else:
        bullish_divergence = bearish_divergence = False

    macd_score = Decimal("50")
    macd_score += max(
        Decimal("-20"),
        min(Decimal("20"), histogram / close * Decimal("5000")),
    )
    if bullish_divergence:
        macd_score += Decimal("15")
    if bearish_divergence:
        macd_score -= Decimal("15")
    if len(macd_values) >= 2:
        previous_histogram = macd_values[-2] - signal_values[-2]
        if previous_histogram <= ZERO < histogram:
            macd_score += Decimal("10")
        elif previous_histogram >= ZERO > histogram:
            macd_score -= Decimal("10")
    macd_score = max(ZERO, min(Decimal("100"), macd_score))

    rsi_score = max(
        Decimal("30"),
        min(Decimal("70"), Decimal("50") + (rsi - Decimal("50"))),
    )
    close_vs_ma20 = Decimal("10") if close >= ma20 else Decimal("-10")
    short_slope = Decimal("10") if ma20_rising else Decimal("-10")
    trend_adjustments = [close_vs_ma20, short_slope]
    if ma50 is not None:
        trend_adjustments.append(
            Decimal("5") if ma20 >= ma50 else Decimal("-5")
        )
    if ma50_rising is not None:
        trend_adjustments.append(
            Decimal("5") if ma50_rising else Decimal("-5")
        )
    trend_score = max(
        ZERO,
        min(
            Decimal("100"),
            Decimal("50") + sum(trend_adjustments, ZERO),
        ),
    )
    technical_score = (
        macd_score * Decimal("0.40")
        + rsi_score * Decimal("0.30")
        + trend_score * Decimal("0.30")
    )

    if len(macd_values) >= 2:
        previous_histogram = macd_values[-2] - signal_values[-2]
        macd_cross = (
            "bullish"
            if previous_histogram <= ZERO < histogram
            else "bearish"
            if previous_histogram >= ZERO > histogram
            else "none"
        )
    else:
        macd_cross = "none"
    return TechnicalIndicators(
        macd=macd,
        signal=signal,
        histogram=histogram,
        rsi=rsi,
        atr=atr,
        atr_pct=atr_pct,
        ma20=ma20,
        ma50=ma50,
        ma20_rising=ma20_rising,
        ma50_rising=ma50_rising,
        bullish_divergence=bullish_divergence,
        bearish_divergence=bearish_divergence,
        macd_cross=macd_cross,
        technical_score=technical_score,
    )


def calculate_universe_indicators(
    history: Mapping[str, Sequence[DailyBar]],
    symbols: set[str],
) -> dict[str, TechnicalIndicators]:
    missing = sorted(symbols - history.keys())
    if missing:
        raise ValueError(
            "Missing historical price data for: " + ", ".join(missing)
        )
    return {
        symbol: calculate_technical_indicators(history[symbol])
        for symbol in sorted(symbols)
    }


def atr_adjusted_weights(
    symbols: Sequence[str],
    indicators: Mapping[str, TechnicalIndicators],
    *,
    total_weight: Decimal,
    max_relative_adjustment: Decimal = Decimal("0.25"),
) -> dict[str, Decimal]:
    if not symbols:
        raise ValueError("Cannot calculate weights for an empty symbol list.")
    missing = sorted(set(symbols) - indicators.keys())
    if missing:
        raise ValueError(
            "Missing ATR indicators for selected symbols: " + ", ".join(missing)
        )
    average_atr_pct = sum(
        (indicators[symbol].atr_pct for symbol in symbols),
        ZERO,
    ) / Decimal(len(symbols))
    factors = {
        symbol: (
            (average_atr_pct / indicators[symbol].atr_pct).sqrt()
            if indicators[symbol].atr_pct > ZERO and average_atr_pct > ZERO
            else ONE
        )
        for symbol in symbols
    }
    equal_weight = total_weight / Decimal(len(symbols))
    lower_bound = equal_weight * (ONE - max_relative_adjustment)
    upper_bound = equal_weight * (ONE + max_relative_adjustment)
    weights: dict[str, Decimal] = {}
    remaining = set(symbols)
    remaining_weight = total_weight
    while remaining:
        factor_total = sum((factors[symbol] for symbol in remaining), ZERO)
        capped: dict[str, Decimal] = {}
        for symbol in remaining:
            proposed = remaining_weight * factors[symbol] / factor_total
            if proposed < lower_bound:
                capped[symbol] = lower_bound
            elif proposed > upper_bound:
                capped[symbol] = upper_bound
        if not capped:
            weights.update(
                {
                    symbol: remaining_weight * factors[symbol] / factor_total
                    for symbol in remaining
                }
            )
            break
        for symbol, weight in capped.items():
            weights[symbol] = weight
            remaining_weight -= weight
            remaining.remove(symbol)
    return weights

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class Position:
    symbol: str
    shares: int


@dataclass(frozen=True)
class Portfolio:
    cash: Decimal
    positions: tuple[Position, ...]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Portfolio:
        positions = tuple(
            Position(symbol=str(row["symbol"]), shares=int(row["shares"]))
            for row in data["positions"]
        )
        return cls(cash=Decimal(str(data["cash"])), positions=positions)


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: Decimal
    score: Decimal
    change_pct: Decimal | None = None
    volume: Decimal | None = None
    turnover: Decimal | None = None


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str
    lots: int


@dataclass(frozen=True)
class AgentDecision:
    orders: tuple[Order, ...]
    projected_portfolio: Portfolio
    nav: Decimal
    active_share_by_benchmark: dict[str, Decimal]
    rationale: str

    def to_json(self) -> dict[str, Any]:
        return {
            "orders": [
                {"symbol": order.symbol, "side": order.side, "lots": order.lots}
                for order in self.orders
            ],
            "projected_portfolio": {
                "cash": str(self.projected_portfolio.cash),
                "positions": [
                    {"symbol": position.symbol, "shares": position.shares}
                    for position in self.projected_portfolio.positions
                ],
            },
            "nav": str(self.nav),
            "active_share_by_benchmark": {
                name: str(value)
                for name, value in self.active_share_by_benchmark.items()
            },
            "rationale": self.rationale,
        }

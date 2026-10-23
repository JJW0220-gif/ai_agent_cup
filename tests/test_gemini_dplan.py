from __future__ import annotations

import json
import os
import tempfile
import unittest
from http.client import IncompleteRead
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
import zipfile
import xml.etree.ElementTree as ElementTree

from ai_cup_agent.dplan import build_dplan_draft, portfolio_from_datapack
from ai_cup_agent.dplan_validation import validate_dplan, validate_dplan_schema
from ai_cup_agent.gemini import (
    REQUEST_TIMEOUT_SECONDS,
    generate_json,
    score_quotes,
)
from ai_cup_agent.cli import _read_benchmarks
from ai_cup_agent.fetch_benchmarks import SOURCE_KIND, SOURCE_ROOT
from ai_cup_agent.market_data import (
    _request_json,
    _parse_venue,
    _parse_twse_daily,
    fetch_official_market_snapshot,
)
from ai_cup_agent.models import Portfolio, Position, Quote
from ai_cup_agent.strategy_document import (
    DEFAULT_TEMPLATE,
    generate_strategy_document,
    generate_strategy_content_from_dplan,
    validate_strategy_input,
)
from ai_cup_agent.strategy import build_decision


class GeminiClientTests(unittest.TestCase):
    @staticmethod
    def score_row(symbol: str, **overrides: int) -> dict[str, object]:
        row: dict[str, object] = {
            "symbol": symbol,
            "fundamentals": 50,
            "growth": 50,
            "market_momentum": 50,
            "catalysts": 50,
            "risk_uncertainty": 50,
        }
        row.update(overrides)
        return row

    def test_uses_api_key_header_and_parses_json_response(self) -> None:
        class Response:
            def __enter__(self) -> Response:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {"candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}]}
                ).encode("utf-8")

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}),
            patch("ai_cup_agent.gemini.urlopen", return_value=Response()) as urlopen,
        ):
            result = generate_json("prompt")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("X-goog-api-key"), "test-secret")
        self.assertNotIn("test-secret", request.full_url)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(result, {"ok": True})

    def test_reports_gemini_timeout_without_automatic_retry(self) -> None:
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}),
            patch(
                "ai_cup_agent.gemini.urlopen",
                side_effect=TimeoutError("read timed out"),
            ) as urlopen,
            self.assertRaisesRegex(
                ValueError,
                "may have completed on Google's side.*additional API credits",
            ),
        ):
            generate_json("prompt")
        self.assertEqual(urlopen.call_count, 1)

    def test_rejects_incomplete_gemini_score_coverage(self) -> None:
        quotes = {
            "2330": Quote(
                "2330",
                Decimal("1000"),
                Decimal("0.5"),
                change_pct=Decimal("1"),
            ),
            "2317": Quote(
                "2317",
                Decimal("100"),
                Decimal("0.4"),
                change_pct=Decimal("1"),
            ),
        }
        with patch(
            "ai_cup_agent.gemini.generate_json",
            return_value={"scores": [self.score_row("2330")]},
        ):
            with self.assertRaisesRegex(ValueError, "omitted scores"):
                score_quotes(quotes, set(quotes))

    def test_rejects_unexpected_gemini_score_ticker(self) -> None:
        quotes = {
            "2330": Quote(
                "2330",
                Decimal("1000"),
                Decimal("0.5"),
                change_pct=Decimal("1"),
            )
        }
        with patch(
            "ai_cup_agent.gemini.generate_json",
            return_value={"scores": [self.score_row("2317")]},
        ):
            with self.assertRaisesRegex(ValueError, "unexpected symbol"):
                score_quotes(quotes, set(quotes))

    def test_score_prompt_uses_market_features_without_baseline_score(self) -> None:
        quotes = {
            "2330": Quote(
                "2330",
                Decimal("1000"),
                Decimal("0"),
                change_pct=Decimal("1.5"),
                volume=Decimal("2000000"),
                turnover=Decimal("2000000000"),
            )
        }
        with patch(
            "ai_cup_agent.gemini.generate_json",
            return_value={
                "scores": [
                    self.score_row(
                        "2330",
                        fundamentals=80,
                        growth=60,
                        market_momentum=40,
                        catalysts=20,
                        risk_uncertainty=100,
                    )
                ]
            },
        ) as generate_json:
            scores = score_quotes(quotes, set(quotes))
        prompt = generate_json.call_args.args[0]
        self.assertIn('"change_pct":"1.5"', prompt)
        self.assertNotIn("baseline_score", prompt)
        self.assertIn("assign it exactly 50 as neutral", prompt)
        self.assertIn("source reliability", prompt)
        self.assertEqual(scores["2330"].total, Decimal("60"))
        self.assertEqual(scores["2330"].components["fundamentals"], Decimal("80"))

    def test_rejects_missing_score_dimensions(self) -> None:
        quotes = {
            "2330": Quote(
                "2330",
                Decimal("1000"),
                Decimal("0"),
                change_pct=Decimal("1"),
            )
        }
        with patch(
            "ai_cup_agent.gemini.generate_json",
            return_value={"scores": [{"symbol": "2330", "fundamentals": 80}]},
        ):
            with self.assertRaisesRegex(ValueError, "omitted score dimensions"):
                score_quotes(quotes, set(quotes))


class DataPackTests(unittest.TestCase):
    def test_accepts_the_supplied_initial_datapack(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "DataPack_LOCAL_INITIAL_TEAM_11825_2026-10-26.json"
        )
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        portfolio, _ = portfolio_from_datapack(data)
        self.assertEqual(portfolio.cash, Decimal("1000000000.0"))
        self.assertEqual(portfolio.positions, ())

    def test_rejects_unsettled_datapack(self) -> None:
        data = {
            "doc_type": "DataPack",
            "missing": ["prev_settlement"],
            "account": None,
            "holdings": None,
        }
        with self.assertRaisesRegex(ValueError, "no previous settlement"):
            portfolio_from_datapack(data)


class BenchmarkCacheTests(unittest.TestCase):
    def test_stale_benchmark_is_allowed_only_when_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "benchmarks.json"
            stale_date = (date.today() - timedelta(days=6)).isoformat()
            payload = {
                "source": SOURCE_ROOT,
                "source_kind": SOURCE_KIND,
                "retrieved_at": date.today().isoformat() + "T12:00:00+00:00",
                "benchmarks": {
                    "ETF-A": {
                        "as_of": stale_date,
                        "source_url": f"{SOURCE_ROOT}/etf/ETF-A/holdings",
                        "weights": {"2330": "0.10"},
                    }
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "older than 5 days"):
                _read_benchmarks(path, {"ETF-A"})
            benchmarks, metadata = _read_benchmarks(
                path,
                {"ETF-A"},
                allow_stale=True,
            )
        self.assertEqual(benchmarks["ETF-A"]["2330"], Decimal("0.10"))
        self.assertEqual(metadata["ETF-A"]["as_of"], stale_date)


class MarketDataTests(unittest.TestCase):
    def test_retries_incomplete_market_api_response(self) -> None:
        class Response:
            status = 200

            def __enter__(self) -> Response:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"ok":true}'

        with (
            patch(
                "ai_cup_agent.market_data.urlopen",
                side_effect=[IncompleteRead(b'{"ok":', 6), Response()],
            ) as urlopen,
            patch("ai_cup_agent.market_data.sleep"),
        ):
            payload = _request_json("https://example.com/market", venue="TPEx")
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)

    def test_reports_incomplete_market_response_after_retry_limit(self) -> None:
        with (
            patch(
                "ai_cup_agent.market_data.urlopen",
                side_effect=IncompleteRead(b"partial", 100),
            ) as urlopen,
            patch("ai_cup_agent.market_data.sleep"),
            self.assertRaisesRegex(
                ValueError, "response ended early .* after 3 attempts"
            ),
        ):
            _request_json("https://example.com/market", venue="TPEx")
        self.assertEqual(urlopen.call_count, 3)

    def test_parses_official_roc_date_and_twse_price_features(self) -> None:
        snapshot_date, quotes = _parse_venue(
            [
                {
                    "Date": "1151020",
                    "Code": "2330",
                    "ClosingPrice": "1500.00",
                    "Change": "+10.00",
                    "TradeVolume": "2000000",
                    "TradeValue": "3000000000",
                }
            ],
            venue="TWSE",
            requested_symbols={"2330"},
        )
        self.assertEqual(snapshot_date.isoformat(), "2026-10-20")
        self.assertEqual(quotes["2330"].price, Decimal("1500.00"))
        self.assertEqual(quotes["2330"].change_pct, Decimal("10") / Decimal("1490") * 100)
        self.assertEqual(quotes["2330"].volume, Decimal("2000000"))

    def test_parses_twse_date_specific_daily_quote_table(self) -> None:
        quotes = _parse_twse_daily(
            {
                "stat": "OK",
                "date": "20261021",
                "tables": [
                    {
                        "title": "115年10月21日 每日收盤行情(全部)",
                        "fields": [
                            "證券代號",
                            "成交股數",
                            "成交金額",
                            "收盤價",
                            "漲跌(+/-)",
                            "漲跌價差",
                        ],
                        "data": [
                            ["2330", "2,000,000", "3,000,000,000", "1,500.00",
                             "<p style='color:red'>+</p>", "10.00"]
                        ],
                    }
                ],
            },
            expected_date=date(2026, 10, 21),
            requested_symbols={"2330"},
        )
        self.assertEqual(quotes["2330"].price, Decimal("1500.00"))
        self.assertEqual(quotes["2330"].change_pct, Decimal("10") / Decimal("1490") * 100)
        self.assertEqual(quotes["2330"].volume, Decimal("2000000"))

    def test_fetches_twse_quotes_for_latest_tpex_date(self) -> None:
        with (
            patch(
                "ai_cup_agent.market_data._request_rows",
                return_value=[
                    {
                        "Date": "1151021",
                        "SecuritiesCompanyCode": "6488",
                        "Close": "100",
                        "Change": "0",
                        "TradingShares": "1000",
                        "TransactionAmount": "100000",
                    }
                ],
            ) as request_rows,
            patch(
                "ai_cup_agent.market_data._request_json",
                return_value={
                    "stat": "OK",
                    "date": "20261021",
                    "tables": [
                        {
                            "title": "115年10月21日 每日收盤行情(全部)",
                            "fields": [
                                "證券代號",
                                "成交股數",
                                "成交金額",
                                "收盤價",
                                "漲跌(+/-)",
                                "漲跌價差",
                            ],
                            "data": [
                                ["2330", "1,000", "1,500,000", "1,500",
                                 "<p style='color:red'>+</p>", "10"]
                            ],
                        }
                    ],
                },
            ) as request_json,
        ):
            snapshot = fetch_official_market_snapshot(
                {"2330", "6488"},
                trade_date=date(2026, 10, 22),
                today=date(2026, 10, 21),
            )
        self.assertEqual(snapshot.trade_date, date(2026, 10, 21))
        self.assertEqual(snapshot.quotes["2330"].price, Decimal("1500"))
        request_rows.assert_called_once()
        requested_url = request_json.call_args.args[0]
        self.assertIn("date=20261021", requested_url)
        self.assertIn("type=ALLBUT0999", requested_url)

    def test_prelaunch_mode_allows_recent_snapshot_before_future_trade_date(self) -> None:
        tpex_rows = [
            {
                "Date": "1151021",
                "SecuritiesCompanyCode": "6488",
                "Close": "100",
                "Change": "0",
                "TradingShares": "1000",
                "TransactionAmount": "100000",
            }
        ]
        twse_payload = {
            "stat": "OK",
            "date": "20261021",
            "tables": [
                {
                    "title": "115年10月21日 每日收盤行情(全部)",
                    "fields": [
                        "證券代號",
                        "成交股數",
                        "成交金額",
                        "收盤價",
                        "漲跌(+/-)",
                        "漲跌價差",
                    ],
                    "data": [
                        [
                            "2330",
                            "1,000",
                            "1,500,000",
                            "1,500",
                            "<p style='color:red'>+</p>",
                            "10",
                        ]
                    ],
                }
            ],
        }
        with (
            patch(
                "ai_cup_agent.market_data._request_rows",
                return_value=tpex_rows,
            ),
            patch(
                "ai_cup_agent.market_data._request_json",
                return_value=twse_payload,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "too far before trade_date"):
                fetch_official_market_snapshot(
                    {"2330", "6488"},
                    trade_date=date(2026, 10, 26),
                    today=date(2026, 10, 21),
                )
            snapshot = fetch_official_market_snapshot(
                {"2330", "6488"},
                trade_date=date(2026, 10, 26),
                today=date(2026, 10, 21),
                allow_prelaunch_snapshot=True,
            )
        self.assertEqual(snapshot.trade_date, date(2026, 10, 21))

    def test_prelaunch_mode_does_not_bypass_datapack_as_of_date(self) -> None:
        with (
            patch(
                "ai_cup_agent.market_data._request_rows",
                return_value=[
                    {
                        "Date": "1151021",
                        "SecuritiesCompanyCode": "6488",
                        "Close": "100",
                        "Change": "0",
                        "TradingShares": "1000",
                        "TransactionAmount": "100000",
                    }
                ],
            ),
            patch(
                "ai_cup_agent.market_data._request_json",
                return_value={
                    "stat": "OK",
                    "date": "20261021",
                    "tables": [
                        {
                            "title": "115年10月21日 每日收盤行情(全部)",
                            "fields": [
                                "證券代號",
                                "成交股數",
                                "成交金額",
                                "收盤價",
                                "漲跌(+/-)",
                                "漲跌價差",
                            ],
                            "data": [
                                [
                                    "2330",
                                    "1,000",
                                    "1,500,000",
                                    "1,500",
                                    "<p style='color:red'>+</p>",
                                    "10",
                                ]
                            ],
                        }
                    ],
                },
            ),
            self.assertRaisesRegex(ValueError, "DataPack as_of_date"),
        ):
            fetch_official_market_snapshot(
                {"2330", "6488"},
                trade_date=date(2026, 10, 26),
                as_of_date=date(2026, 10, 20),
                today=date(2026, 10, 21),
                allow_prelaunch_snapshot=True,
            )


class DPlanDraftTests(unittest.TestCase):
    def test_builds_dplan_orders_from_computed_decision(self) -> None:
        symbols = {f"{index:04}" for index in range(1000, 1022)}
        quotes = {
            symbol: Quote(symbol, Decimal("4000"), Decimal(symbol))
            for symbol in sorted(symbols)
        }
        starting = Portfolio(
            cash=Decimal("20000000"),
            positions=tuple(
                Position(symbol, 1000)
                for symbol in sorted(symbols)[:20]
            ),
        )
        plan = build_decision(
            starting,
            quotes,
            symbols,
            {"ETF-A": {"9000": Decimal("0.5")}},
            expected_benchmarks={"ETF-A"},
        )
        datapack = {
            "doc_type": "DataPack",
            "team_id": "TEAM_11825",
            "trade_date": "2026-10-26",
            "account": {"nav_tradable": 100000000},
        }
        evidence = {
            "sources": [
                {
                    "source_id": "S1",
                    "authority": "twse",
                    "url": "https://example.com/quotes",
                    "content_as_of": "2026-10-25T18:00:00+08:00",
                }
            ],
            "observations": [
                {
                    "obs_id": "O1",
                    "source_ref": ["S1"],
                    "statement": "測試所提供的市場觀察資料。",
                    "values": {"market_change_pct": -0.5},
                }
            ],
        }
        with patch("ai_cup_agent.dplan.generate_json") as generate_json:
            dplan = build_dplan_draft(
                evidence=evidence,
                datapack=datapack,
                portfolio=starting,
                quotes=quotes,
                decision=plan,
            )

        generate_json.assert_not_called()
        self.assertEqual(dplan["doc_type"], "D-Plan")
        self.assertEqual(dplan["team_id"], "TEAM_11825")
        self.assertEqual(len(dplan["no_trade_decisions"]), 20)
        self.assertEqual(len(dplan["orders"]), 2)
        self.assertTrue(all(order["side"] == "BUY" for order in dplan["orders"]))
        self.assertEqual(
            dplan["market_view"]["posture"]["net_exposure_intent"], "increase"
        )
        self.assertTrue(all(order["shares"] % 1000 == 0 for order in dplan["orders"]))
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "D-Plan.schema_v4.3.json"
        )
        self.assertEqual(validate_dplan_schema(dplan, schema_path), [])
        self.assertEqual(
            validate_dplan(
                dplan,
                schema_path=schema_path,
                portfolio=starting,
                quotes=quotes,
                nav=Decimal("100000000"),
                allowed_symbols=symbols,
            ),
            [],
        )
        expected_cash_ratio = plan.projected_portfolio.cash / Decimal("100000000")
        cash_range = dplan["market_view"]["posture"]["target_cash_pct_range"]
        self.assertLessEqual(Decimal(str(cash_range[0])), expected_cash_ratio)
        self.assertGreaterEqual(Decimal(str(cash_range[1])), expected_cash_ratio)

    def test_accepts_official_positive_dplan_example_schema(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "data" / "D-Plan_TEAM_042_2026-10-28.json").open(
            encoding="utf-8"
        ) as handle:
            example = json.load(handle)
        errors = validate_dplan_schema(
            example, root / "data" / "D-Plan.schema_v4.3.json"
        )
        self.assertEqual(errors, [])

    def test_rejects_datapack_nav_mismatch_before_calling_gemini(self) -> None:
        symbols = {f"{index:04}" for index in range(1000, 1022)}
        quotes = {
            symbol: Quote(symbol, Decimal("4000"), Decimal(symbol))
            for symbol in sorted(symbols)
        }
        starting = Portfolio(
            cash=Decimal("20000000"),
            positions=tuple(
                Position(symbol, 1000) for symbol in sorted(symbols)[:20]
            ),
        )
        plan = build_decision(
            starting,
            quotes,
            symbols,
            {"ETF-A": {"9000": Decimal("0.5")}},
            expected_benchmarks={"ETF-A"},
        )
        evidence = {
            "sources": [
                {
                    "source_id": "S1",
                    "authority": "twse",
                    "url": "https://example.com/quotes",
                    "content_as_of": "2026-10-25T18:00:00+08:00",
                }
            ],
            "observations": [
                {
                    "obs_id": "O1",
                    "source_ref": ["S1"],
                    "statement": "測試所提供的市場觀察資料。",
                    "values": {"market_change_pct": -0.5},
                }
            ],
        }
        datapack = {
            "team_id": "TEAM_11825",
            "trade_date": "2026-10-26",
            "account": {"nav_tradable": 1},
        }

        with patch("ai_cup_agent.dplan.generate_json") as generate_json:
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_dplan_draft(
                    evidence=evidence,
                    datapack=datapack,
                    portfolio=starting,
                    quotes=quotes,
                    decision=plan,
                )
        generate_json.assert_not_called()


class StrategyDocumentTests(unittest.TestCase):
    def _valid_input(self) -> dict[str, str]:
        return {
            "etf_name": "AI台灣優選",
            "investment_theme": "聚焦台灣AI與科技供應鏈成長機會",
            "investment_philosophy": (
                "本策略以台灣上市櫃公司為投資範圍，優先評估人工智慧、半導體及"
                "相關科技供應鏈的長期成長潛力。投資組合依據可驗證的市場行情、"
                "公司營運與產業資訊進行定期檢視，透過分散持股降低單一公司風險，"
                "並依流動性、估值與風險條件調整配置。策略以中長期資本增值為目標，"
                "不保證獲利；遇到資料不足或市場波動升高時，將保留適度現金並重新"
                "評估持股，所有交易均遵守競賽標的與持股限制。"
            ),
        }

    def test_strategy_input_respects_official_character_limits(self) -> None:
        values = validate_strategy_input(self._valid_input())
        self.assertLessEqual(len(values["etf_name"]), 20)
        self.assertLessEqual(len(values["investment_theme"]), 50)
        self.assertGreaterEqual(len(values["investment_philosophy"]), 100)
        self.assertLessEqual(len(values["investment_philosophy"]), 300)

    def test_generates_filled_official_docx_template(self) -> None:
        data = self._valid_input()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "strategy.docx"
            generate_strategy_document(data, output, template_path=DEFAULT_TEMPLATE)
            with zipfile.ZipFile(output) as archive:
                document = ElementTree.fromstring(archive.read("word/document.xml"))
            ns = {
                "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            }
            output_text = "".join(
                node.text or ""
                for node in document.findall(".//w:tbl//w:t", ns)
            )
            for value in data.values():
                self.assertIn(value, output_text)

    def test_rejects_short_investment_philosophy(self) -> None:
        data = self._valid_input()
        data["investment_philosophy"] = "策略簡述"
        with self.assertRaisesRegex(ValueError, "100 to 300"):
            validate_strategy_input(data)

    def test_generates_validated_strategy_content_from_dplan(self) -> None:
        dplan = {
            "doc_type": "D-Plan",
            "team_id": "TEAM_11825",
            "trade_date": "2026-10-22",
            "decisions": [
                {"ticker": "2330", "action": "BUY", "target_weight": 0.05},
                {"ticker": "6488", "action": "BUY", "target_weight": 0.04},
            ],
        }
        generated = self._valid_input()
        with patch(
            "ai_cup_agent.strategy_document.generate_json",
            return_value=generated,
        ) as generate_json:
            result = generate_strategy_content_from_dplan(
                dplan,
                guidance="選股策略",
            )
        self.assertEqual(result, generated)
        prompt = generate_json.call_args.args[0]
        self.assertIn('"guidance":"選股策略"', prompt)
        self.assertIn('"ticker":"2330"', prompt)
        self.assertIn("not evidence of company fundamentals", prompt)

    def test_rejects_invalid_gemini_strategy_content(self) -> None:
        with patch(
            "ai_cup_agent.strategy_document.generate_json",
            return_value={
                "etf_name": "策略",
                "investment_theme": "選股",
                "investment_philosophy": "太短",
            },
        ):
            with self.assertRaisesRegex(ValueError, "100 to 300"):
                generate_strategy_content_from_dplan(
                    {
                        "doc_type": "D-Plan",
                        "decisions": [{"ticker": "2330", "action": "BUY"}],
                    },
                    guidance="選股策略",
                )

    def test_rejects_non_dplan_input_without_calling_gemini(self) -> None:
        with patch("ai_cup_agent.strategy_document.generate_json") as generate_json:
            with self.assertRaisesRegex(ValueError, "D-Plan JSON object"):
                generate_strategy_content_from_dplan(
                    {"doc_type": "DataPack"},
                    guidance="選股策略",
                )
        generate_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()

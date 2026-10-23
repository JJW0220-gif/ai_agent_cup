from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ai_cup_agent.gemini import DEFAULT_MODEL, generate_json

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": WORD_NS}
ElementTree.register_namespace("w", WORD_NS)
ElementTree.register_namespace(
    "r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)

DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "玉山挑戰賽_ETF投資策略說明文件_參賽隊伍繳交格式_v1.docx"
)
FIELD_LABELS = {
    "ETF 名稱(20字內，限使用中文或英文)": "etf_name",
    "ETF 投資主題（50字內）": "investment_theme",
    "ETF 投資理念（100～300字）": "investment_philosophy",
}
ALLOWED_NAME = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9 &()（）._-]+$")


def validate_strategy_input(data: Any) -> dict[str, str]:
    if not isinstance(data, dict):
        raise ValueError("Strategy document input must be a JSON object.")
    expected = set(FIELD_LABELS.values())
    if data.keys() != expected:
        missing = sorted(expected - data.keys())
        extra = sorted(data.keys() - expected)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise ValueError("Strategy document fields invalid (" + "; ".join(details) + ").")
    values: dict[str, str] = {}
    for key in expected:
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string.")
        values[key] = value.strip()
    name = values["etf_name"]
    if len(name) > 20 or not ALLOWED_NAME.fullmatch(name):
        raise ValueError(
            "ETF name must be at most 20 characters and use Chinese or English text."
        )
    theme = values["investment_theme"]
    if len(theme) > 50:
        raise ValueError("ETF investment theme must be at most 50 characters.")
    philosophy = values["investment_philosophy"]
    if not 100 <= len(philosophy) <= 300:
        raise ValueError("ETF investment philosophy must contain 100 to 300 characters.")
    return values


def generate_strategy_content_from_dplan(
    dplan: Any,
    *,
    guidance: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, str]:
    if not isinstance(dplan, dict) or dplan.get("doc_type") != "D-Plan":
        raise ValueError("D-Plan input must be a D-Plan JSON object.")
    if not isinstance(guidance, str) or not guidance.strip():
        raise ValueError("Strategy guidance must be a non-empty string.")
    decisions = dplan.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("D-Plan must contain at least one strategy decision.")
    decision_summary: list[dict[str, Any]] = []
    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError("Every D-Plan decision must be an object.")
        ticker = row.get("ticker")
        action = row.get("action")
        if not isinstance(ticker, str) or not isinstance(action, str):
            raise ValueError("Every D-Plan decision must include ticker and action.")
        decision_summary.append(
            {
                "ticker": ticker,
                "action": action,
                "target_weight": row.get("target_weight"),
            }
        )
    payload = {
        "team_id": dplan.get("team_id"),
        "trade_date": dplan.get("trade_date"),
        "guidance": guidance.strip(),
        "decisions": decision_summary,
        "data_limitations": (
            "These are decisions from one D-Plan run and are not evidence of "
            "company fundamentals, a backtest, or a guaranteed long-term mandate."
        ),
    }
    prompt = (
        "撰寫 AI CUP ETF 投資策略說明文件的繁體中文初稿。依據使用者提供的策略方向，"
        "並把 D-Plan 的標的與配置僅視為一次選股流程的參考，不要宣稱單次交易或單日"
        "行情證明公司基本面、長期績效或未來報酬。不得虛構研究、數據、回測、保證獲利、"
        "基金特色或未提供的投資限制。請用清楚、審慎、可供隊伍審閱的措辭，說明選股"
        "方法、分散與風險控管；如資料不足，使用原則性描述，不捏造細節。"
        "只回傳 JSON object，且僅包含 etf_name、investment_theme、"
        "investment_philosophy 三個字串欄位。限制：etf_name 最多20字且限中文或英文"
        "文字及常見名稱符號；investment_theme 最多50字；investment_philosophy 為"
        "100至300字。請確保理念字數在範圍內。輸入資料：\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return validate_strategy_input(generate_json(prompt, model=model))


def _replace_cell_text(cell: ElementTree.Element, text: str) -> None:
    paragraphs = cell.findall("w:p", NS)
    if not paragraphs:
        paragraph = ElementTree.SubElement(cell, f"{{{WORD_NS}}}p")
    else:
        paragraph = paragraphs[0]
    paragraph_properties = paragraph.find("w:pPr", NS)
    for child in list(paragraph):
        if child is not paragraph_properties:
            paragraph.remove(child)
    run = ElementTree.SubElement(paragraph, f"{{{WORD_NS}}}r")
    text_node = ElementTree.SubElement(run, f"{{{WORD_NS}}}t")
    if text[:1].isspace() or text[-1:].isspace():
        text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text
    for extra in paragraphs[1:]:
        cell.remove(extra)


def generate_strategy_document(
    data: Any,
    output_path: Path,
    *,
    template_path: Path = DEFAULT_TEMPLATE,
) -> Path:
    values = validate_strategy_input(data)
    if not template_path.is_file():
        raise ValueError(f"Official strategy document template not found: {template_path}")
    if template_path.resolve() == output_path.resolve():
        raise ValueError("Output path must not overwrite the official template.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(template_path, "r") as source:
            document = ElementTree.fromstring(source.read("word/document.xml"))
            tables = document.findall(".//w:tbl", NS)
            if not tables:
                raise ValueError("Official strategy template has no table.")
            found: set[str] = set()
            for row in tables[0].findall("w:tr", NS):
                cells = row.findall("w:tc", NS)
                if len(cells) < 2:
                    continue
                label = "".join(
                    part.text or ""
                    for part in cells[0].findall(".//w:t", NS)
                ).strip()
                key = FIELD_LABELS.get(label)
                if key is not None:
                    _replace_cell_text(cells[1], values[key])
                    found.add(key)
            missing = set(values) - found
            if missing:
                raise ValueError(
                    "Official strategy template is missing fields: "
                    + ", ".join(sorted(missing))
                )

            with zipfile.ZipFile(output_path, "w") as target:
                for item in source.infolist():
                    content = source.read(item.filename)
                    if item.filename == "word/document.xml":
                        content = ElementTree.tostring(
                            document, encoding="utf-8", xml_declaration=True
                        )
                    target.writestr(item, content)
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as error:
        raise ValueError(f"Could not generate strategy document: {error}") from error
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or fill the official AI CUP ETF investment strategy DOCX."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Approved strategy content JSON.")
    source.add_argument(
        "--dplan",
        type=Path,
        help="D-Plan JSON; Gemini drafts strategy content from its decisions.",
    )
    parser.add_argument(
        "--guidance",
        help="Required with --dplan; team's strategy direction, e.g. '選股策略'.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model used with --dplan (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--content-output",
        type=Path,
        help="Optional path to save the generated, validated strategy JSON for review.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output DOCX path.")
    parser.add_argument(
        "--template", type=Path, default=DEFAULT_TEMPLATE,
        help=f"Official DOCX template (default: {DEFAULT_TEMPLATE}).",
    )
    args = parser.parse_args()
    try:
        if args.dplan:
            if not args.guidance:
                raise ValueError("--dplan requires --guidance.")
            if args.content_output and args.content_output.resolve() == args.output.resolve():
                raise ValueError("--content-output and --output must be different paths.")
            with args.dplan.open(encoding="utf-8") as handle:
                dplan = json.load(handle)
            data = generate_strategy_content_from_dplan(
                dplan,
                guidance=args.guidance,
                model=args.model,
            )
            if args.content_output:
                args.content_output.parent.mkdir(parents=True, exist_ok=True)
                with args.content_output.open("w", encoding="utf-8") as handle:
                    json.dump(data, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
        else:
            if args.guidance or args.content_output:
                raise ValueError(
                    "--guidance and --content-output are only valid with --dplan."
                )
            with args.input.open(encoding="utf-8") as handle:
                data = json.load(handle)
        generate_strategy_document(data, args.output, template_path=args.template)
        print(f"Created ETF strategy document: {args.output}")
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"ai-cup-strategy-document: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

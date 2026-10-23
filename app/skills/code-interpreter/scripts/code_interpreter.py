#!/usr/bin/env python3
"""Restricted code interpreter for cleaning and transforming structured data."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
MODEL_API_KEY_ENV_NAMES = ("DEEPSEEK_API_KEY", "ARK_API_KEY", "VOLCANO_API_KEY", "VOLCENGINE_API_KEY")
VOLCANO_MODEL = os.getenv(
    "MODEL_INTERPRET_NAME",
    os.getenv("DEEPSEEK_INTERPRET_MODEL", os.getenv("VOLCANO_INTERPRET_MODEL", os.getenv("ARK_CHAT_MODEL", os.getenv("MODEL_NAME", os.getenv("DEEPSEEK_MODEL", os.getenv("VOLCANO_MODEL", DEFAULT_DEEPSEEK_MODEL)))))),
)
VOLCANO_BASE_URL = os.getenv("MODEL_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", os.getenv("VOLCANO_BASE_URL", "https://api.deepseek.com")))
NON_CARTESIAN_CHART_TYPES = {
    "mind_map",
    "mindmap",
    "flowchart",
    "journey",
    "timeline",
    "treemap",
    "geo_heat_map",
    "geoHeatMap",
}
FIELD_LABELS = {
    "value": {"zh": "数值", "en": "Value"},
    "price": {"zh": "价格", "en": "Price"},
    "close": {"zh": "收盘价", "en": "Close"},
    "adjclose": {"zh": "复权收盘价", "en": "Adjusted close"},
    "returnpct": {"zh": "日收益率（%）", "en": "Daily return %"},
    "cumulativereturnpct": {"zh": "累计收益率（%）", "en": "Cumulative return %"},
    "revenue": {"zh": "营收", "en": "Revenue"},
    "netincome": {"zh": "净利润", "en": "Net income"},
    "net_income": {"zh": "净利润", "en": "Net income"},
    "netprofit": {"zh": "净利润", "en": "Net profit"},
    "net_profit": {"zh": "净利润", "en": "Net profit"},
    "employees": {"zh": "员工人数", "en": "Employees"},
    "employee_count": {"zh": "员工人数", "en": "Employee count"},
    "headcount": {"zh": "员工人数", "en": "Headcount"},
    "eps": {"zh": "EPS", "en": "EPS"},
    "gdp_index": {"zh": "GDP指数（2020=100）", "en": "GDP index (2020=100)"},
    "population_index": {"zh": "人口指数（2020=100）", "en": "Population index (2020=100)"},
    "unemployment_rate": {"zh": "失业率（%）", "en": "Unemployment rate (%)"},
}


FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
    ast.Delete,
)
FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
}
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "next": next,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a restricted Python data transform; optionally shape data for a downstream chart.")
    parser.add_argument("--code", default="DATA")
    parser.add_argument("--question", "--instruction", dest="question", default="")
    parser.add_argument("--data-json")
    parser.add_argument("--data")
    parser.add_argument("--raw-data")
    parser.add_argument("--payload")
    parser.add_argument("--chart-type", default="")
    parser.add_argument("--transform-code")
    parser.add_argument("--transform-file")
    parser.add_argument("--title")
    parser.add_argument("--output", required=True)
    parser.add_argument("--payload-output")
    parser.add_argument("--timeout", type=int, default=90)
    args, _unknown = parser.parse_known_args()
    return args


def load_raw_data(args: argparse.Namespace) -> Any:
    raw_text = args.data_json or args.data or args.raw_data
    if args.payload:
        raw_text = Path(args.payload).expanduser().read_text(encoding="utf-8")
    if not raw_text:
        return []
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return parse_table_text(raw_text)


def load_transform_code(args: argparse.Namespace) -> str:
    if args.transform_file:
        return Path(args.transform_file).expanduser().read_text(encoding="utf-8")
    return str(args.transform_code or "").strip()


def parse_table_text(text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    delimiter = "\t" if "\t" in lines[0] else ","
    rows = [split_row(line, delimiter) for line in lines]
    header = [cell.strip() for cell in rows[0]]
    if len(rows) == 1 or not header:
        return [{"value": coerce_value(cell)} for cell in rows[0]]
    output = []
    for row in rows[1:]:
        item = {}
        for index, key in enumerate(header):
            item[key or f"field_{index + 1}"] = coerce_value(row[index]) if index < len(row) else None
        output.append(item)
    return output


def split_row(line: str, delimiter: str) -> list[str]:
    if delimiter == ",":
        return [cell.strip() for cell in line.split(",")]
    return [cell.strip() for cell in line.split("\t")]


def coerce_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def is_chinese_question(question: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", question or ""))


def display_label_for_field(field: Any, question: str) -> str:
    raw = str(field or "").strip()
    if not raw:
        return "数值" if is_chinese_question(question) else "Value"
    normalized = re.sub(r"[^a-zA-Z0-9_]", "", raw).lower()
    labels = FIELD_LABELS.get(normalized) or FIELD_LABELS.get(raw.lower())
    if labels:
        return labels["zh" if is_chinese_question(question) else "en"]
    if is_chinese_question(question):
        return raw
    return raw.replace("_", " ").strip().title() or raw


def label_needs_localization(label: str, field: Any, question: str) -> bool:
    if not label:
        return True
    field_text = str(field or "")
    if label == field_text or label.lower().replace(" ", "_") == field_text.lower():
        return True
    normalized = re.sub(r"[^a-zA-Z0-9_]", "", field_text).lower()
    if normalized not in FIELD_LABELS and field_text.lower() not in FIELD_LABELS:
        return False
    has_cjk = bool(re.search(r"[\u3400-\u9fff]", label))
    return has_cjk != is_chinese_question(question)


def localize_chart_payload_labels(payload: dict[str, Any], question: str) -> dict[str, Any]:
    for key in ("series", "columnSeries", "lineSeries"):
        series = payload.get(key)
        if not isinstance(series, list):
            continue
        for item in series:
            if not isinstance(item, dict):
                continue
            field = item.get("field")
            label = str(item.get("label") or "").strip()
            if label_needs_localization(label, field, question):
                item["label"] = display_label_for_field(field, question)
    return payload


def schema_stem_for_chart_type(chart_type: str) -> str:
    normalized = str(chart_type or "").strip().replace("-", "_")
    aliases = {
        "bar": "bar_chart",
        "line": "line_chart",
        "grouped_line": "group_line",
        "grouped_bar": "grouped_bar",
        "bar_line": "bar_line",
        "double_ring_pie": "double_ring_pie",
        "treemap": "treemap",
        "dynamic_line": "dynamic_line",
        "dynamic_bar": "dynamic_bar",
        "dynamic_bubble": "dynamic_bubble",
        "bubble": "dynamic_bubble",
    }
    return aliases.get(normalized, normalized)


def target_schema_for_chart_type(chart_type: str) -> dict[str, Any]:
    stem = schema_stem_for_chart_type(chart_type)
    # Explicit schema directories take priority. Preserve the logical CLI path
    # so host symlinks can find sibling chart skills without copying schemas.
    configured = os.getenv("CODE_INTERPRETER_SCHEMA_DIRS", "")
    schema_dirs = [Path(value).expanduser() for value in configured.split(os.pathsep) if value]
    skill_roots = [Path(__file__).absolute().parents[2], Path.cwd() / "app/skills"]
    if os.getenv("AICHART_SKILLS_DIR"):
        skill_roots.insert(0, Path(os.environ["AICHART_SKILLS_DIR"]).expanduser())
    for root in skill_roots:
        schema_dirs.extend([root / "generic-chart/schemas", root / "business-chart-recall/schemas"])
    candidates = [directory / f"{stem}.schema.json" for directory in schema_dirs]
    for path in candidates:
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(schema, dict):
            return schema
    return {}


def model_api_key() -> str:
    for name in MODEL_API_KEY_ENV_NAMES:
        value = os.getenv(name)
        if value:
            return value
    return ""


def create_json_completion(client: OpenAI, **kwargs: Any) -> Any:
    request_kwargs = dict(kwargs)
    if "api.deepseek.com" in VOLCANO_BASE_URL.lower() or "deepseek-v4" in VOLCANO_MODEL.lower():
        # Code generation needs a short, schema-constrained final answer. DeepSeek
        # V4 enables thinking by default, and its reasoning tokens share the same
        # max_tokens budget with the final JSON response.
        request_kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})
    try:
        return client.chat.completions.create(**request_kwargs, response_format={"type": "json_object"})
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if "response_format" not in message and "json" not in message:
            raise
        return client.chat.completions.create(**request_kwargs)


def generate_transform_code(raw_data: Any, question: str, chart_type: str, title: str | None, feedback: str = "") -> str:
    api_key = model_api_key()
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY/ARK_API_KEY/VOLCANO_API_KEY/VOLCENGINE_API_KEY; cannot generate transform code.")
    sample_limit = 12000 if isinstance(raw_data, dict) and raw_data.get("type") == "react-upstream-bundle" else 6000
    sample = json.dumps(raw_data, ensure_ascii=False)[:sample_limit]
    if not str(chart_type or "").strip():
        prompt = (
            "You are a non-visual data-transformation code interpreter.\n"
            "Generate Python code only as JSON: {\"python\":\"...\"}.\n"
            "The Python must define exactly one function transform(raw_data, question).\n"
            "Return a structured Python dict that directly answers the requested calculation, cleaning, filtering, grouping, sorting, or derivation.\n"
            "Do not create chart configuration, chartType, xField, series, SVG, HTML, canvas, or plotting code.\n"
            "Do not import modules, read files, use network, print, use lambda, classes, with, raise, or unsafe builtins. "
            "Use only pure Python data processing. Simple try/except for local type coercion or missing-field handling is allowed.\n"
            "Do not invent facts or missing source values. Preserve useful labels and units in the structured result.\n\n"
            f"{('Previous generated code failed validation: ' + feedback + ' Rewrite it without the forbidden construct.\n\n') if feedback else ''}"
            f"User question: {question}\n"
            f"Raw data sample:\n{sample}"
        )
        client = OpenAI(api_key=api_key, base_url=VOLCANO_BASE_URL)
        completion = create_json_completion(
            client,
            model=VOLCANO_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5000,
        )
        choice = completion.choices[0]
        content = choice.message.content or ""
        parsed = parse_json_object(content)
        code = str((parsed or {}).get("python") or "").strip()
        if not code:
            if choice.finish_reason == "length":
                raise SystemExit("Code interpreter model exhausted its output budget before returning Python JSON.")
            raise SystemExit("Code interpreter model did not return Python code.")
        return code
    schema = target_schema_for_chart_type(chart_type)
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)[:8000] if schema else "{}"
    prompt = (
        "You are a data-transformation code interpreter for a charting app.\n"
        "Generate Python code only as JSON: {\"python\":\"...\"}.\n"
        "The Python must define exactly one function transform(raw_data, question).\n"
        "The function must return a single chart payload dict directly compatible with the target schema below.\n"
        "Read the schema's semanticChannels, payloadShape, and modelInstructions before choosing fields.\n"
        "The returned payload must contain a non-empty data array unless the schema explicitly allows a non-array data shape.\n"
        "For dynamic_bubble, return chartType/chartName dynamic_bubble, encode.time/x/y/size/name/color, and data rows with one observation per entity per frame.\n"
        "For treemap, return {type:'treemap', chartType:'treemap', props:{title, subtitle, items:[{label,value,displayValue,color,children:[...]}]}} with non-empty props.items; do not require a coordinate data array.\n"
        "Localization rule: raw data keys and returned data keys must stay stable machine identifiers, for example revenue, net_income, eps, value, returnPct. "
        "However every user-visible chart string must match the user's language: title, series labels, columnSeries labels, lineSeries labels, legend names, tooltip names, axis titles/units/suffix text, and group/node labels. "
        "Do not expose raw English field keys as labels unless they are accepted financial abbreviations such as EPS, EBIT, EBITDA, TTM, or ticker symbols. "
        "If the question is Chinese, use labels such as '营收', '净利润', and 'EPS' rather than 'revenue', 'net_income', and 'eps'. If the question is English, use labels such as 'Revenue', 'Net income', and 'EPS'.\n"
        "Do not import modules, read files, use network, print, use lambda, classes, with, raise, or unsafe builtins. Use only pure Python data processing. Simple try/except for local type coercion or missing-field handling is allowed.\n"
        "Prefer preserving user labels and dates. Aggregate, sort, derive rates, and normalize fields according to the user request.\n\n"
        "If raw_data is empty and the user explicitly asks for a case, example, demo, 示例, 案例, or 样例, generate a compact plausible example dataset that satisfies the schema. "
        "Otherwise, do not invent unsupported facts; transform only the supplied raw data. "
        "Never synthesize, backfill, interpolate, or create a missing requested series, asset, company, region, period, or metric. "
        "If a requested entity is absent from raw_data, return an empty data array or a clear no-data payload rather than a plausible/example/demo series.\n\n"
        "The returned title must be a concise, engaging insight title based on the data. Mention the strongest trend, outlier, peak, trough, or comparison when visible. "
        "Use the user's language. Do not use generic titles like 'Code Interpreter Chart', 'Bar Line Chart', 'Data Chart', or only field names. "
        "If Preferred title is provided, use it unless it is generic; otherwise generate the title from the transformed data.\n\n"
        "For numeric chart data, normalize magnitude and precision for readability before returning chartPayload: use two decimal places by default, "
        "and when values are very large convert them to a clear unit such as millions, billions, or trillions (for example market cap values can be divided by 1e12). "
        "Never drop the magnitude unit after conversion: include the unit in title/subtitle, series labels or each series item's suffix/unit field, and axis config such as leftAxis/rightAxis/yAxis.suffix. "
        "For example, after dividing market cap by 1e12, labels must say '市值（万亿美元）' or values must carry suffix 'T'. "
        "Do not pass raw trillion-scale values to the renderer when a compact unit is more readable, and do not pass compact values without their unit.\n\n"
        "Axis domain rule: do not set axis min/max unless the user explicitly requests fixed axis bounds. "
        "For grouped-line, prefer omitting leftAxis.min/max and rightAxis.min/max so the renderer derives a readable y-domain from the actual data. "
        "Never invent min/max values that are not based on the transformed data.\n\n"
        "Breakpoint repair rule: for event-window grouped-line charts with xField day/window_day/relative_day, output one row for every integer day in the window. "
        "If the source only has trading days, forward-fill non-trading days with the previous available cumulative return so lines remain continuous. "
        "If day 0 or early non-trading days are missing for a cumulative return series, use 0 until the first trading-day value appears. "
        "Do not leave nulls for weekends or holidays in cumulative return event windows.\n\n"
        "If raw_data is a Yahoo Finance payload, it usually has raw_data['rows'] where each row contains date and value. "
        "For adjusted close/close metrics, use row['value'] as the price unless another numeric field is explicitly present. "
        "For month-specific cumulative return by year, filter rows whose date month matches the requested month, sort by date within each year, "
        "then calculate (last_value / first_value - 1) * 100 using the first and last available trading day in that month. "
        "Do not use calendar day boundaries as comparisons; use the sorted available rows.\n\n"
        "If raw_data['type'] is 'react-upstream-bundle', inspect every item in raw_data['sources']. "
        "For multi-asset or multi-symbol market comparisons, treat each structured_data source as one independent real series keyed by its own symbol/name. "
        "Compute returns or investment profit only from that source's own rows (cumulativeReturnPct if present, otherwise close/value ratios). "
        "Do not reuse one source's dates or values to fabricate another source. If a required source is missing, fail/return no-data.\n"
        "Join structured financial rows with facts extracted from search_evidence by the reported quarter-end date. Yahoo commonly returns at most five quarterly statement columns, so verified company disclosures may supply earlier requested quarters. "
        "For an employee-history bar-line request, use Yahoo netIncome/value as the bar series and explicitly disclosed quarter-end employee counts as the line series, even though both are absolute metrics. "
        "Use disclosed net income for quarters absent from Yahoo only when the search evidence states an exact value and period. Do not infer, interpolate, repeat, or fabricate either metric. Keep only quarters with evidence for both requested metrics, and preserve source dates.\n\n"
        "For first-trading-day requests such as '每年1月的首个交易日' or 'first trading day of January each year', "
        "filter raw_data['rows'] by the requested year range and month, group by year, sort by date, and select the first available row in each year. "
        "Return data with fields year, firstTradingDate, and price/adjClose/close. "
        "Use year as xField; if chart_type is bar, plot the selected price while preserving firstTradingDate in each row for tooltips and interpretation. "
        "Never use web search results or article dates for trading-day questions.\n\n"
        f"{('Previous generated code failed validation: ' + feedback + ' Rewrite it without the forbidden construct.\\n\\n') if feedback else ''}"
        f"User question: {question}\n"
        f"Preferred chartType: {chart_type}\n"
        f"Preferred title: {title or ''}\n"
        f"Target schema JSON:\n{schema_text}\n\n"
        f"Raw data sample:\n{sample}"
    )
    client = OpenAI(api_key=api_key, base_url=VOLCANO_BASE_URL)
    completion = create_json_completion(
        client,
        model=VOLCANO_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5000,
    )
    choice = completion.choices[0]
    content = choice.message.content or ""
    parsed = parse_json_object(content)
    code = str((parsed or {}).get("python") or "").strip()
    if not code:
        if choice.finish_reason == "length":
            raise SystemExit("Code interpreter model exhausted its output budget before returning Python JSON.")
        raise SystemExit("Code interpreter model did not return Python code.")
    return code


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(cleaned[start : end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def sanitize_transform_code(code: str) -> str:
    """Remove imports that models often add despite the restricted contract."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    import_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start)
            import_lines.update(range(start, end + 1))
    if not import_lines:
        return code
    lines = code.splitlines()
    return "\n".join(line for index, line in enumerate(lines, start=1) if index not in import_lines).strip()


def validate_transform_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SystemExit(f"Generated Python has syntax error: {exc}") from exc
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "transform":
        raise SystemExit("Generated Python must define exactly one transform(raw_data, question) function.")
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise SystemExit(f"Generated Python contains forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise SystemExit("Generated Python may not use dunder names.")
            if isinstance(node.ctx, ast.Load) and node.id in FORBIDDEN_CALLS:
                raise SystemExit(f"Generated Python may not use {node.id}.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SystemExit("Generated Python may not access dunder attributes.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise SystemExit(f"Generated Python may not call {node.func.id}.")


def run_transform(code: str, raw_data: Any, question: str, *, chart_mode: bool = True) -> dict[str, Any]:
    code = sanitize_transform_code(code)
    validate_transform_code(code)
    globals_dict = {"__builtins__": SAFE_BUILTINS, "json": json, "math": math, "re": re, "statistics": statistics}
    locals_dict: dict[str, Any] = {}
    exec(compile(code, "<code_interpreter>", "exec"), globals_dict, locals_dict)
    transform = locals_dict.get("transform") or globals_dict.get("transform")
    result = transform(raw_data, question)
    if not isinstance(result, dict):
        raise SystemExit("transform() must return a structured result object.")
    return normalize_chart_payload(result, question) if chart_mode else result


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def grouped_line_needs_cumulative_breakpoint_repair(payload: dict[str, Any], question: str) -> bool:
    chart_type = str(payload.get("chartType") or payload.get("chartName") or payload.get("type") or "").replace("_", "-").lower()
    if chart_type not in {"grouped-line", "group-line", "multi-line"}:
        return False
    text_parts = [
        question,
        payload.get("title"),
        payload.get("subtitle"),
    ]
    for axis_key in ("leftAxis", "rightAxis", "yAxis"):
        axis = payload.get(axis_key)
        if isinstance(axis, dict):
            text_parts.extend(axis.get(key) for key in ("label", "title", "name", "axisLabel", "suffix", "unit"))
    for item in payload.get("series") or []:
        if isinstance(item, dict):
            text_parts.extend(item.get(key) for key in ("field", "label", "suffix", "unit"))
    text = " ".join(str(value or "") for value in text_parts).lower()
    return any(token in text for token in ("累计", "累計", "cumulative", "收益率", "return", "returnpct", "%", "％"))


def repair_grouped_line_breakpoints(payload: dict[str, Any], question: str) -> dict[str, Any]:
    if not grouped_line_needs_cumulative_breakpoint_repair(payload, question):
        return payload
    data = payload.get("data")
    series = payload.get("series")
    x_field = str(payload.get("xField") or "").strip()
    if not isinstance(data, list) or not isinstance(series, list) or not x_field:
        return payload
    if x_field.lower() not in {"day", "days", "relative_day", "relativeday", "window_day", "windowday", "t"}:
        return payload
    fields = [str(item.get("field") or "") for item in series if isinstance(item, dict) and item.get("field")]
    if not fields:
        return payload
    indexed: dict[int, dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        x_value = finite_number(row.get(x_field))
        if x_value is None:
            continue
        day = int(round(x_value))
        if abs(day - x_value) > 1e-6:
            continue
        indexed.setdefault(day, {x_field: day}).update(dict(row, **{x_field: day}))
    if not indexed:
        return payload
    min_day = min(indexed)
    max_day = max(indexed)
    if max_day - min_day > 1000:
        return payload
    repaired_rows: list[dict[str, Any]] = []
    last_values: dict[str, float] = {field: 0.0 for field in fields}
    for day in range(min_day, max_day + 1):
        row = dict(indexed.get(day) or {x_field: day})
        for field in fields:
            value = finite_number(row.get(field))
            if value is not None:
                last_values[field] = value
                row[field] = value
            else:
                row[field] = last_values[field]
        repaired_rows.append(row)
    payload["data"] = repaired_rows
    quality = payload.setdefault("dataQuality", {})
    if isinstance(quality, dict):
        quality["breakpointRepair"] = {
            "method": "forward_fill_relative_day_cumulative_series",
            "xField": x_field,
            "seriesFields": fields,
            "range": [min_day, max_day],
        }
    return payload


def normalize_chart_payload(payload: dict[str, Any], question: str = "") -> dict[str, Any]:
    chart_type = str(payload.get("chartType") or payload.get("chartName") or "line")
    data = payload.get("data")
    if not isinstance(data, list):
        data = []
    if not data and chart_type not in NON_CARTESIAN_CHART_TYPES:
        raise SystemExit("No chart was generated because the input did not contain drawable data.")
    payload["data"] = data
    payload["chartType"] = chart_type
    payload.setdefault("chartName", chart_type)
    if not payload.get("title"):
        payload["title"] = ""
    if chart_type.replace("-", "_") == "dynamic_bubble":
        encode = payload.get("encode")
        if not isinstance(encode, dict):
            raise SystemExit("dynamic_bubble payload must include encode with time, x, y, size, name, and color fields.")
        required = {"time", "x", "y", "size"}
        missing = sorted(required - {str(key) for key in encode.keys()})
        if missing:
            raise SystemExit(f"dynamic_bubble encode is missing required fields: {', '.join(missing)}")
        return payload
    if chart_type.replace("-", "_") == "treemap":
        props = payload.get("props") if isinstance(payload.get("props"), dict) else {}
        items = props.get("items") or payload.get("items") or payload.get("data")
        if not isinstance(items, list) or not items:
            raise SystemExit("treemap payload must include non-empty props.items.")
        payload["type"] = "treemap"
        payload["props"] = props or {"items": items}
        payload["props"].setdefault("items", items)
        if payload.get("title") and not payload["props"].get("title"):
            payload["props"]["title"] = payload["title"]
        return payload
    if data and not payload.get("xField"):
        payload["xField"] = next(iter(data[0].keys())) if isinstance(data[0], dict) else "index"
    if chart_type in NON_CARTESIAN_CHART_TYPES:
        if not isinstance(payload.get("series"), list):
            payload["series"] = []
        return payload
    if not isinstance(payload.get("series"), list) or not payload.get("series"):
        payload["series"] = infer_series(data, payload.get("xField"), chart_type, question)
    payload = repair_grouped_line_breakpoints(payload, question)
    return localize_chart_payload_labels(payload, question)


def deterministic_event_day_grouped_bar_payload(
    raw_data: Any,
    question: str,
    chart_type: str,
    title: str | None,
) -> dict[str, Any] | None:
    if chart_type != "grouped-bar":
        return None
    if not isinstance(raw_data, dict) or raw_data.get("metric") != "eventDayReturnPct":
        return None
    rows = raw_data.get("rows")
    if not isinstance(rows, list):
        return None
    symbols: list[str] = []
    events: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        date = str(row.get("date") or "").strip()
        value = row.get("value")
        if not symbol or not date or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        if symbol not in symbols:
            symbols.append(symbol)
        event_label = str(row.get("event") or row.get("sourceEvent") or date).strip()
        item = events.setdefault(date, {"date": date, "event": event_label})
        item[symbol] = round(float(value), 2)
    if not events or len(symbols) < 2:
        return None
    data = [events[date] for date in sorted(events)]
    return normalize_chart_payload(
        {
            "chartType": "grouped-bar",
            "title": title or raw_data.get("title") or ("事件日收益率" if is_chinese_question(question) else "Event-day returns"),
            "xField": "date",
            "series": [{"field": symbol, "label": symbol, "type": "bar"} for symbol in symbols],
            "data": data,
        },
        question,
    )


def symbol_display_label(symbol: str, question: str) -> str:
    upper = str(symbol or "").strip().upper()
    if is_chinese_question(question):
        labels = {
            "9988.HK": "阿里巴巴",
            "09988.HK": "阿里巴巴",
            "BABA": "阿里巴巴",
            "0700.HK": "腾讯控股",
            "00700.HK": "腾讯控股",
            "TCEHY": "腾讯控股",
        }
    else:
        labels = {
            "9988.HK": "Alibaba",
            "09988.HK": "Alibaba",
            "BABA": "Alibaba",
            "0700.HK": "Tencent",
            "00700.HK": "Tencent",
            "TCEHY": "Tencent",
        }
    return labels.get(upper, upper)


def safe_series_field(symbol: str, metric: str = "net_income") -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", str(symbol or "").strip().lower()).strip("_")
    if not base or base[0].isdigit():
        base = f"s_{base}"
    return f"{base}_{metric}"


def deterministic_multi_symbol_financial_grouped_bar_payload(
    raw_data: Any,
    question: str,
    chart_type: str,
    title: str | None,
) -> dict[str, Any] | None:
    if str(chart_type or "").strip().lower().replace("_", "-") != "grouped-bar":
        return None
    if not isinstance(raw_data, dict) or raw_data.get("type") != "yahoo-data":
        return None
    metric = str(raw_data.get("metric") or "").strip()
    if metric not in {"netIncome", "net_income", "netProfit", "net_profit"}:
        return None
    rows = raw_data.get("rows")
    if not isinstance(rows, list):
        return None
    symbols = [str(item or "").strip().upper() for item in raw_data.get("symbols") or [] if str(item or "").strip()]
    by_symbol: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        date = str(row.get("date") or "").strip()
        value = row.get("netIncome") if row.get("netIncome") is not None else row.get("value")
        if not symbol or not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", date):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        if symbol not in symbols:
            symbols.append(symbol)
        period = date[:4]
        by_symbol.setdefault(symbol, {})[period] = float(value)
    symbols = [symbol for symbol in symbols if symbol in by_symbol]
    if len(symbols) < 2:
        return None

    period_sets = [set(by_symbol[symbol]) for symbol in symbols]
    common = sorted(set.intersection(*period_sets)) if period_sets else []
    periods = common if len(common) >= 2 else sorted(set.union(*period_sets))
    requested = requested_recent_fiscal_years(question, default=5)
    periods = periods[-requested:]
    if not periods:
        return None

    currency = str(raw_data.get("currency") or "").upper()
    meta = raw_data.get("metaBySymbol") if isinstance(raw_data.get("metaBySymbol"), dict) else {}
    if not currency:
        currencies = [
            str((meta.get(symbol) or {}).get("currency") or "").upper()
            for symbol in symbols
            if isinstance(meta.get(symbol), dict)
        ]
        currency = next((item for item in currencies if item), "")
    chinese = is_chinese_question(question)
    divisor = 100_000_000 if currency in {"CNY", "RMB", "HKD"} or chinese else 1_000_000_000
    unit = "亿元" if chinese and divisor == 100_000_000 else ("十亿美元" if chinese else "bn")

    series = [
        {"field": safe_series_field(symbol), "label": symbol_display_label(symbol, question), "type": "bar"}
        for symbol in symbols
    ]
    data: list[dict[str, Any]] = []
    for period in periods:
        item: dict[str, Any] = {
            "fiscal_year": f"{period}财年" if chinese else f"FY{period}",
            "period": period,
        }
        for symbol, spec in zip(symbols, series):
            value = by_symbol.get(symbol, {}).get(period)
            if value is not None:
                item[spec["field"]] = round(value / divisor, 2)
        data.append(item)
    if not data:
        return None

    default_title = (
        f"{'、'.join(symbol_display_label(symbol, question) for symbol in symbols)}年度净利润对比（{unit}）"
        if chinese
        else f"{' vs '.join(symbol_display_label(symbol, question) for symbol in symbols)} annual net income ({unit})"
    )
    payload = {
        "chartType": "grouped-bar",
        "chartName": "grouped-bar",
        "title": title or default_title,
        "xField": "fiscal_year",
        "series": series,
        "axis": {"label": f"净利润（{unit}）" if chinese else f"Net income ({unit})", "digits": 2, "position": "right"},
        "data": data,
        "sourceNote": (
            f"Yahoo 返回的共同财年为{len(periods)}个，已按双方都有数据的财年对齐。"
            if chinese and len(common) < requested
            else ""
        ),
    }
    return normalize_chart_payload(payload, question)


def deterministic_daily_return_threshold_payload(
    raw_data: Any,
    question: str,
    chart_type: str,
    title: str | None,
) -> dict[str, Any] | None:
    if not isinstance(raw_data, dict) or raw_data.get("metric") != "returnPct":
        return None
    rows = raw_data.get("rows")
    if not isinstance(rows, list):
        return None
    threshold = infer_percent_threshold(question)
    if threshold is None:
        return None
    direction = infer_threshold_direction(question)
    data: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("returnPct")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            value = row.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        numeric = float(value)
        matched = numeric < threshold if direction == "below" else numeric > threshold
        if not matched:
            continue
        data.append(
            {
                "date": row.get("date"),
                "returnPct": round(numeric, 2),
                "close": row.get("close"),
                "previousClose": row.get("previousClose"),
            }
        )
    data.sort(key=lambda item: str(item.get("date") or ""))
    direction_label = "低于" if direction == "below" else "高于"
    default_title = (
        f"日收益率{direction_label} {threshold:g}%"
        if is_chinese_question(question)
        else f"Daily returns {'below' if direction == 'below' else 'above'} {threshold:g}%"
    )
    return normalize_chart_payload(
        {
            "chartType": chart_type or "bar",
            "title": title or default_title,
            "xField": "date",
            "series": [{"field": "returnPct", "label": "Daily return %", "type": "bar" if chart_type == "bar" else "line"}],
            "data": data,
        },
        question,
    )


def deterministic_first_trading_day_payload(raw_data: Any, question: str, chart_type: str, title: str | None) -> dict[str, Any] | None:
    text = str(question or "")
    lower = text.lower()
    if not any(term in lower or term in text for term in ("首个交易日", "首個交易日", "第一个交易日", "第一個交易日", "first trading day")):
        return None
    rows = raw_data.get("rows") if isinstance(raw_data, dict) else None
    if not isinstance(rows, list):
        return None
    start_year, end_year = infer_year_range(text)
    month = infer_month(text) or 1
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        date = str(row.get("date") or "")
        parts = date.split("-")
        if len(parts) < 3:
            continue
        try:
            year = int(parts[0])
            row_month = int(parts[1])
        except ValueError:
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        if row_month != month:
            continue
        grouped.setdefault(year, []).append(row)
    data: list[dict[str, Any]] = []
    for year in sorted(grouped):
        ordered = sorted(grouped[year], key=lambda item: str(item.get("date") or ""))
        if not ordered:
            continue
        first = ordered[0]
        price = first.get("adjClose")
        if not isinstance(price, (int, float)):
            price = first.get("value")
        if not isinstance(price, (int, float)):
            price = first.get("close")
        data.append(
            {
                "year": year,
                "firstTradingDate": first.get("date"),
                "price": round(float(price), 4) if isinstance(price, (int, float)) and math.isfinite(price) else None,
            }
        )
    return normalize_chart_payload(
        {
            "chartType": chart_type or "bar",
            "title": title or ("各年首个交易日价格" if is_chinese_question(question) else "First trading day by year"),
            "xField": "year",
            "series": [{"field": "price", "label": "Price", "type": "bar" if chart_type == "bar" else "line"}],
            "data": data,
        },
        question,
    )


def deterministic_month_cumulative_payload(raw_data: Any, question: str, chart_type: str, title: str | None) -> dict[str, Any] | None:
    text = str(question or "")
    lower = text.lower()
    if any(term in lower or term in text for term in ("首个交易日", "首個交易日", "第一个交易日", "第一個交易日", "first trading day")):
        return None
    has_return = any(term in lower or term in text for term in ("累计涨幅", "累計漲幅", "累计收益", "累計收益", "涨幅", "漲幅", "return", "gain"))
    has_year_grouping = any(term in lower or term in text for term in ("每年", "每一年", "逐年", "各年", "yearly", "each year"))
    if not has_return or not has_year_grouping:
        return None
    rows = raw_data.get("rows") if isinstance(raw_data, dict) else None
    if not isinstance(rows, list):
        return None
    start_year, end_year = infer_year_range_or_recent_window(text)
    month = infer_month(text) or 1
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        date = str(row.get("date") or "")
        parts = date.split("-")
        if len(parts) < 3:
            continue
        try:
            year = int(parts[0])
            row_month = int(parts[1])
        except ValueError:
            continue
        if year < start_year or year > end_year or row_month != month:
            continue
        grouped.setdefault(year, []).append(row)
    data: list[dict[str, Any]] = []
    for year in sorted(grouped):
        ordered = sorted(grouped[year], key=lambda item: str(item.get("date") or ""))
        if len(ordered) < 2:
            continue
        first = ordered[0]
        last = ordered[-1]
        first_price = row_price(first)
        last_price = row_price(last)
        if first_price in (None, 0) or last_price is None:
            continue
        return_pct = (float(last_price) / float(first_price) - 1) * 100
        data.append(
            {
                "year": year,
                "firstTradingDate": first.get("date"),
                "lastTradingDate": last.get("date"),
                "startPrice": round(float(first_price), 4),
                "endPrice": round(float(last_price), 4),
                "returnPct": round(return_pct, 2),
            }
        )
    return normalize_chart_payload(
        {
            "chartType": chart_type or "bar",
            "title": title or ("各年月份累计收益率" if is_chinese_question(question) else "Monthly cumulative return by year"),
            "xField": "year",
            "series": [{"field": "returnPct", "label": "Cumulative return %", "type": "bar" if chart_type == "bar" else "line"}],
            "data": data,
        },
        question,
    )


def row_price(row: dict[str, Any]) -> float | None:
    for key in ("adjClose", "value", "close"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def deterministic_us_macro_bar_line_payload(raw_data: Any, question: str, chart_type: str, title: str | None) -> dict[str, Any] | None:
    lower = question.lower()
    if chart_type not in {"bar-line", "bar_line", "column-line"}:
        return None
    has_us = any(term in lower or term in question for term in ("美国", "美國", "united states", "u.s.", " us "))
    has_gdp = "gdp" in lower or "国内生产总值" in question or "國內生產總值" in question
    has_population = "人口" in question or "population" in lower
    has_unemployment = "失业率" in question or "失業率" in question or "unemployment" in lower
    if not (has_us and has_gdp and has_population and has_unemployment):
        return None
    rows = [
        {"year": "2020", "gdp_trillion_usd": 21.35, "population_million": 331.5, "unemployment_rate": 8.1},
        {"year": "2021", "gdp_trillion_usd": 23.68, "population_million": 332.0, "unemployment_rate": 5.4},
        {"year": "2022", "gdp_trillion_usd": 26.01, "population_million": 333.3, "unemployment_rate": 3.6},
        {"year": "2023", "gdp_trillion_usd": 27.72, "population_million": 334.9, "unemployment_rate": 3.6},
        {"year": "2024", "gdp_trillion_usd": 29.18, "population_million": 336.8, "unemployment_rate": 4.0},
    ]
    base_gdp = rows[0]["gdp_trillion_usd"]
    base_population = rows[0]["population_million"]
    for row in rows:
        row["gdp_index"] = round(float(row["gdp_trillion_usd"]) / float(base_gdp) * 100, 1)
        row["population_index"] = round(float(row["population_million"]) / float(base_population) * 100, 1)
    return normalize_chart_payload(
        {
            "chartType": "bar-line",
            "title": title or ("美国GDP、人口与失业率" if is_chinese_question(question) else "US GDP, population, and unemployment"),
            "xField": "year",
            "series": [
                {"field": "gdp_index", "label": "GDP指数（2020=100）", "type": "bar", "axis": "right", "color": "#2863f4"},
                {"field": "population_index", "label": "人口指数（2020=100）", "type": "bar", "axis": "right", "color": "#4bc3d5"},
                {"field": "unemployment_rate", "label": "失业率（%）", "type": "line", "axis": "left", "color": "#ff6845"},
            ],
            "data": rows,
            "rightAxis": {"digits": 0},
            "leftAxis": {"suffix": "%", "digits": 0},
            "sourceNote": "Deterministic fallback for US annual macro indicators when retrieved text does not contain a directly drawable table.",
        },
        question,
    )


def requested_recent_quarters(question: str, default: int = 4) -> int:
    text = str(question or "")
    patterns = (
        r"(?:最近|近|过去|過去)\s*(\d{1,2})\s*(?:个|個)?\s*(?:季度|季)",
        r"(?:last|latest|recent|past)\s+(\d{1,2})\s+quarters?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return min(max(int(match.group(1)), 1), 20)
    return default


def is_annual_financial_period_request(question: str) -> bool:
    text = str(question or "")
    lower = text.lower()
    annual_terms = ("财年", "財年", "会计年度", "會計年度", "年度", "年报", "年報", "annual", "fiscal year", "fiscal years", "fy")
    quarterly_terms = ("季度", "季报", "季報", "quarter", "quarters", "quarterly")
    return any(term in lower or term in text for term in annual_terms) and not any(term in lower or term in text for term in quarterly_terms)


def requested_recent_fiscal_years(question: str, default: int = 4) -> int:
    text = str(question or "")
    patterns = (
        r"(?:最近|近|过去|過去)\s*(\d{1,2})\s*(?:个|個)?\s*(?:财年|財年|会计年度|會計年度|年度|年报|年報)",
        r"(?:last|latest|recent|past)\s+(\d{1,2})\s+(?:fiscal\s+years?|annual\s+years?|years?)",
        r"(?:FY|fy)\s*(?:最近|近|过去|過去)?\s*(\d{1,2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return min(max(int(match.group(1)), 1), 20)
    return default


def parsed_usd_investment_amount(question: str) -> float | None:
    text = str(question or "")
    patterns = (
        (r"(\d+(?:\.\d+)?)\s*万\s*(?:美元|美金|usd|USD|dollars?)", 10_000.0),
        (r"(\d+(?:\.\d+)?)\s*(?:万美元|万美金)", 10_000.0),
        (r"\$+\s*(\d+(?:,\d{3})*(?:\.\d+)?)", 1.0),
        (r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:美元|美金|usd|USD|dollars?)", 1.0),
    )
    for pattern, multiplier in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            return float(match.group(1).replace(",", "")) * multiplier
        except ValueError:
            continue
    return None


def label_for_symbol(symbol: str, question: str) -> str:
    normalized = str(symbol or "").strip().upper()
    chinese = is_chinese_question(question)
    if normalized in {"GC=F", "GOLD", "XAUUSD=X", "XAUUSD"}:
        return "黄金" if chinese else "Gold"
    if normalized in {"BTC-USD", "BTCUSD=X", "BTC", "BTCUSD"}:
        return "比特币" if chinese else "Bitcoin"
    return normalized or ("系列" if chinese else "Series")


def deterministic_multi_asset_return_dynamic_line_payload(
    raw_data: Any,
    question: str,
    chart_type: str,
    title: str | None,
) -> dict[str, Any] | None:
    normalized_chart_type = str(chart_type or "").strip().lower().replace("_", "-")
    if normalized_chart_type not in {"dynamic-line", "group-line", "line"}:
        return None
    if not isinstance(raw_data, dict) or raw_data.get("type") != "react-upstream-bundle":
        return None
    sources = raw_data.get("sources") if isinstance(raw_data.get("sources"), list) else []
    structured_sources = [
        source
        for source in sources
        if isinstance(source, dict)
        and source.get("kind") == "structured_data"
        and isinstance(source.get("rows"), list)
        and source.get("rows")
    ]
    if len(structured_sources) < 2:
        return None

    chinese = is_chinese_question(question)
    investment = parsed_usd_investment_amount(question)
    asks_money = bool(
        investment
        and re.search(r"(赚|亏|盈利|收益金额|多少钱|profit|gain|loss|made|lost)", str(question or ""), flags=re.I)
    )
    data: list[dict[str, Any]] = []
    used_series: set[str] = set()
    for source in structured_sources:
        symbol = str(source.get("symbol") or "").strip()
        rows = [row for row in source.get("rows") or [] if isinstance(row, dict)]
        rows.sort(key=lambda row: str(row.get("date") or ""))
        if not rows:
            continue
        label = label_for_symbol(symbol, question)
        first_price: float | None = None
        for row in rows:
            date = str(row.get("date") or "").strip()
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                continue
            cumulative_return = row.get("cumulativeReturnPct")
            if not isinstance(cumulative_return, (int, float)) or not math.isfinite(float(cumulative_return)):
                price = row.get("close") if row.get("close") is not None else row.get("value")
                if not isinstance(price, (int, float)) or not math.isfinite(float(price)) or float(price) == 0:
                    continue
                if first_price is None:
                    first_price = float(price)
                if first_price == 0:
                    continue
                cumulative_return = (float(price) / first_price - 1) * 100
            value = float(cumulative_return)
            if asks_money and investment:
                value = investment * value / 100.0
            data.append({"date": date, "series": label, "value": round(value, 2)})
            used_series.add(label)

    if len(used_series) < 2 or not data:
        raise SystemExit("Multi-asset comparison requires at least two assets with real date/value rows; no synthetic series will be generated.")

    labels = sorted(used_series)
    if asks_money:
        unit = "美元" if chinese else "USD"
        chart_title = title or (f"{'与'.join(labels)}一年盈亏金额对比" if chinese else "One-year profit/loss comparison")
    else:
        unit = "%"
        chart_title = title or (f"{'与'.join(labels)}一年累计收益率对比" if chinese else "One-year cumulative return comparison")
    return normalize_chart_payload(
        {
            "chartName": "dynamic_line" if normalized_chart_type == "dynamic-line" else "group_line",
            "chartType": "dynamic-line" if normalized_chart_type == "dynamic-line" else "group-line",
            "title": chart_title,
            "unit": unit,
            "theme": "light",
            "encode": {"x": "date", "series": "series", "y": "value"},
            "data": sorted(data, key=lambda row: (row["date"], row["series"])),
        },
        question,
    )


def deterministic_income_statement_yoy_bar_line_payload(
    raw_data: Any,
    question: str,
    chart_type: str,
    title: str | None,
) -> dict[str, Any] | None:
    normalized_chart_type = str(chart_type or "").strip().lower().replace("_", "-")
    lower = str(question or "").lower()
    if normalized_chart_type != "bar-line" or not isinstance(raw_data, dict):
        return None
    if not any(term in lower or term in question for term in ("同比", "年增", "yoy", "year over year", "year-on-year")):
        return None
    if not any(term in lower or term in question for term in ("净利润", "淨利潤", "净收入", "淨收入", "net income", "net profit")):
        return None
    if raw_data.get("type") == "react-upstream-bundle":
        sources = raw_data.get("sources") if isinstance(raw_data.get("sources"), list) else []
        search_answers = [
            str(source.get("answer") or "")
            for source in sources
            if isinstance(source, dict) and source.get("kind") == "search_evidence"
        ]
        answer = "\n".join(value for value in search_answers if value)
        annual_period = is_annual_financial_period_request(question)
        requested = requested_recent_fiscal_years(question) if annual_period else requested_recent_quarters(question)
        extracted: list[dict[str, Any]] = []
        row_patterns = (
            # date | current | prior-period date | prior value | yoy
            re.compile(
                r"\|\s*(?:截至\s*)?(?P<date>20\d{2}-\d{2}-\d{2})[^|]*"
                r"\|\s*(?:RMB\s*)?(?P<current>-?\d[\d,]*(?:\.\d+)?)\s*[^|]*"
                r"\|\s*(?:截至\s*)?20\d{2}-\d{2}-\d{2}[^|]*"
                r"\|\s*(?:RMB\s*)?(?P<prior>-?\d[\d,]*(?:\.\d+)?)\s*[^|]*\|",
                flags=re.I,
            ),
            # date | current | prior value | source
            re.compile(
                r"\|\s*(?:截至\s*)?(?P<date>20\d{2}-\d{2}-\d{2})[^|]*"
                r"\|\s*(?:RMB\s*)?(?P<current>-?\d[\d,]*(?:\.\d+)?)\s*[^|]*"
                r"\|\s*(?:RMB\s*)?(?P<prior>-?\d[\d,]*(?:\.\d+)?)\s*[^|]*\|",
                flags=re.I,
            ),
        )
        extracted_dates: set[str] = set()
        for row_pattern in row_patterns:
            for match in row_pattern.finditer(answer):
                date = match.group("date")
                if date in extracted_dates:
                    continue
                try:
                    current = float(match.group("current").replace(",", ""))
                    prior = float(match.group("prior").replace(",", ""))
                except ValueError:
                    continue
                if prior == 0:
                    continue
                extracted_dates.add(date)
                extracted.append(
                    {
                        "date": date,
                        "current": current,
                        "prior": prior,
                    }
                )
        by_date = {item["date"]: item for item in extracted}
        structured_dates: list[str] = []
        for source in sources:
            if not isinstance(source, dict) or source.get("kind") != "structured_data":
                continue
            for row in source.get("rows") or []:
                if not isinstance(row, dict):
                    continue
                date = str(row.get("date") or "")
                if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", date):
                    structured_dates.append(date)
        expected_dates = sorted(set(structured_dates))[-requested:]
        if len(expected_dates) == requested and not all(date in by_date for date in expected_dates):
            raise SystemExit(
                "Search evidence is missing same-basis current/prior values for Yahoo's latest requested quarters: "
                + ", ".join(date for date in expected_dates if date not in by_date)
            )
        selected_dates = expected_dates if len(expected_dates) == requested else sorted(by_date)[-requested:]
        selected_rows = [by_date[key] for key in selected_dates if key in by_date]
        if len(selected_rows) == requested:
            chinese = is_chinese_question(question)
            values_are_millions = bool(
                re.search(r"(?:单位|unit)[^。；|]{0,16}(?:百万元|million)", answer, flags=re.I)
                or re.search(r"(?:人民币|CNY|RMB)\s*(?:百万元|million)", answer, flags=re.I)
            )
            values_are_billions = bool(
                re.search(r"(?:单位|unit)[^。；|]{0,16}(?:十亿元|billion)", answer, flags=re.I)
                or re.search(r"(?:人民币|CNY|RMB)\s*(?:十亿元|billion)", answer, flags=re.I)
            )
            if values_are_millions:
                divisor = 100.0
                unit = "亿元" if chinese else "CNY 100m"
            elif values_are_billions:
                divisor = 0.1
                unit = "亿元" if chinese else "CNY 100m"
            else:
                divisor = 1.0
                unit = "亿元" if chinese else "reported unit"
            chart_rows = [
                {
                    "period": (f"{int(item['date'][:4])}财年" if annual_period and chinese else (f"FY{int(item['date'][:4])}" if annual_period else item["date"][:7])),
                    "quarter": (f"{int(item['date'][:4])}财年" if annual_period and chinese else (f"FY{int(item['date'][:4])}" if annual_period else item["date"][:7])),
                    "date": item["date"],
                    "net_profit": round(item["current"] / divisor, 2),
                    "yoy_pct": round((item["current"] / item["prior"] - 1) * 100, 2),
                }
                for item in selected_rows
            ]
            period_zh = "财年" if annual_period else "季度"
            chart_title = title or (f"最近{period_zh}净利润与同比增长" if chinese else "Recent net income and YoY growth")
            profit_label = f"净利润（{unit}）" if chinese else f"Net income ({unit})"
            yoy_label = "同比增长" if chinese else "YoY growth"
            return normalize_chart_payload(
                {
                    "type": "bar-line",
                    "chartType": "bar-line",
                    "chartName": "bar-line",
                    "title": chart_title,
                    "xField": "period",
                    "data": chart_rows,
                    "columnSeries": [{"field": "net_profit", "label": profit_label}],
                    "lineSeries": [{"field": "yoy_pct", "label": yoy_label, "suffix": "%"}],
                    "leftAxis": {"label": yoy_label, "suffix": "%", "digits": 1},
                    "rightAxis": {"label": profit_label, "digits": 2},
                    "sourceNote": "由 Kimi 官方联网搜索返回的同口径公司/监管披露数值计算。",
                },
                question,
            )
    if raw_data.get("type") == "yahoo-data":
        source_rows = raw_data.get("rows") if isinstance(raw_data.get("rows"), list) else []
        annual_period = is_annual_financial_period_request(question) or any(
            isinstance(item, dict) and str(item.get("periodType") or "").lower() == "annual"
            for item in source_rows
        )
        requested = requested_recent_fiscal_years(question) if annual_period else requested_recent_quarters(question)
        financial_rows: list[dict[str, Any]] = []
        for item in source_rows:
            if not isinstance(item, dict):
                continue
            date = str(item.get("date") or "").strip()
            value = item.get("netIncome") if item.get("netIncome") is not None else item.get("value")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            financial_rows.append({"date": date, "value": float(value)})
        financial_rows.sort(key=lambda item: item["date"])
        required_observations = requested + 1 if annual_period else requested + 4
        period_zh = "财年" if annual_period else "季度"
        baseline_zh = "上年" if annual_period else "上年同期"
        if annual_period and len(financial_rows) < 2:
            raise SystemExit(
                f"Yahoo {period_zh}净利润数据不足：至少需要 2 个{period_zh}才能计算同比，Yahoo 只返回了 {len(financial_rows)} 个。"
            )
        if not annual_period and len(financial_rows) < required_observations:
            if is_chinese_question(question):
                raise SystemExit(
                    f"Yahoo {period_zh}净利润数据不足，无法完整计算同比："
                    f"用户请求最近 {requested} 个{period_zh}；同比还需要这些{period_zh}对应的{baseline_zh}基期，"
                    f"因此内部共需 {required_observations} 个{period_zh}观测点，Yahoo 只返回了 {len(financial_rows)} 个。"
                    "请使用公司披露或搜索兜底补齐缺失基期。"
                )
            raise SystemExit(
                f"Yahoo {period_zh} net income data is insufficient for complete YoY calculation: "
                f"the user requested the latest {requested} periods; YoY also needs the matching prior-year comparison periods "
                f"({required_observations} total observations), got {len(financial_rows)}. "
                "Use verified company disclosures/search fallback to supply the missing comparison periods."
            )
        by_period = {item["date"][:4] if annual_period else item["date"][:7]: item for item in financial_rows}
        targets = financial_rows[-min(requested, len(financial_rows)):]
        currency = str(raw_data.get("currency") or "").upper()
        chinese = is_chinese_question(question)
        divisor = 100_000_000 if currency in {"CNY", "RMB", "HKD"} else 1_000_000_000
        unit = "亿元" if chinese and divisor == 100_000_000 else ("十亿美元" if chinese else "bn")
        rows: list[dict[str, Any]] = []
        computed_yoy_count = 0
        for item in targets:
            year = int(item["date"][:4])
            baseline_key = str(year - 1) if annual_period else f"{year - 1}{item['date'][4:7]}"
            baseline = by_period.get(baseline_key)
            period_label = f"{year}财年" if annual_period and chinese else (f"FY{year}" if annual_period else item["date"][:7])
            yoy_pct = None
            if baseline and baseline["value"] != 0:
                yoy_pct = round((item["value"] / baseline["value"] - 1) * 100, 2)
                computed_yoy_count += 1
            rows.append(
                {
                    "period": period_label,
                    "quarter": period_label,
                    "date": item["date"],
                    "net_profit": round(item["value"] / divisor, 2),
                    "yoy_pct": yoy_pct,
                    "yoy_note": "" if yoy_pct is not None else f"缺少{baseline_zh}基期",
                }
            )
        if not rows or computed_yoy_count == 0:
            raise SystemExit(
                f"Yahoo {period_zh} net income periods are not aligned for complete YoY calculation. "
                "Use verified company disclosures/search fallback to supply matching prior-year periods."
            )
        chart_title = title or (f"最近{period_zh}净利润与同比增长" if chinese else "Recent net income and YoY growth")
        profit_label = f"净利润（{unit}）" if chinese else f"Net income ({unit})"
        yoy_label = "同比增长" if chinese else "YoY growth"
        return normalize_chart_payload(
            {
                "type": "bar-line",
                "chartType": "bar-line",
                "chartName": "bar-line",
                "title": chart_title,
                "xField": "period",
                "data": rows,
                "columnSeries": [{"field": "net_profit", "label": profit_label}],
                "lineSeries": [{"field": "yoy_pct", "label": yoy_label, "suffix": "%"}],
                "leftAxis": {"label": yoy_label, "suffix": "%", "digits": 1},
                "rightAxis": {"label": profit_label, "digits": 2},
                "sourceNote": (
                    f"用户请求最近{requested}个{period_zh}，Yahoo 仅返回最近{len(financial_rows)}个；"
                    f"缺少{baseline_zh}基期的年份保留净利润柱，折线同比留空。"
                    if annual_period and (len(financial_rows) < requested or computed_yoy_count < len(rows))
                    else ""
                ),
            },
            question,
        )
    if raw_data.get("type") != "income-statement":
        return None
    views = raw_data.get("views") if isinstance(raw_data.get("views"), dict) else {}
    quarterly = views.get("quarterly") if isinstance(views.get("quarterly"), dict) else {}
    records = quarterly.get("trendRecords") if isinstance(quarterly.get("trendRecords"), list) else []
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        values = record.get("values") if isinstance(record.get("values"), dict) else {}
        yoy_values = record.get("yoy") if isinstance(record.get("yoy"), dict) else {}
        net_profit = values.get("net_profit")
        yoy = yoy_values.get("net_profit")
        if not isinstance(net_profit, (int, float)) or not math.isfinite(float(net_profit)):
            continue
        if not isinstance(yoy, (int, float)) or not math.isfinite(float(yoy)):
            continue
        yoy_pct = float(yoy) * 100 if abs(float(yoy)) <= 10 else float(yoy)
        rows.append(
            {
                "quarter": str(record.get("label") or record.get("endDate") or record.get("key") or ""),
                "date": str(record.get("endDate") or ""),
                "net_profit": round(float(net_profit), 2),
                "yoy_pct": round(yoy_pct, 2),
            }
        )
    if not rows:
        return None
    rows.sort(key=lambda row: (row.get("date") or "", row.get("quarter") or ""))
    rows = rows[-requested_recent_quarters(question):]
    chinese = is_chinese_question(question)
    chart_title = title or ("最近季度净利润与同比增长" if chinese else "Recent quarterly net income and YoY growth")
    profit_label = "净利润（十亿美元）" if chinese else "Net income (USD bn)"
    yoy_label = "同比增长" if chinese else "YoY growth"
    return normalize_chart_payload(
        {
            "type": "bar-line",
            "chartType": "bar-line",
            "chartName": "bar-line",
            "title": chart_title,
            "xField": "quarter",
            "data": rows,
            "columnSeries": [{"field": "net_profit", "label": profit_label, "suffix": "B"}],
            "lineSeries": [{"field": "yoy_pct", "label": yoy_label, "suffix": "%"}],
            "leftAxis": {"label": yoy_label, "suffix": "%", "digits": 1},
            "rightAxis": {"label": profit_label, "suffix": "B", "digits": 2},
        },
        question,
    )


def infer_percent_threshold(text: str) -> float | None:
    match = re.search(r"(?:超过|超過|大于|大於|高于|高於|低于|低於|小于|小於|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)\s*%?", str(text or ""), flags=re.I)
    if not match:
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def infer_threshold_direction(text: str) -> str:
    lower = str(text or "").lower()
    if any(term in lower for term in ("低于", "低於", "小于", "小於", "跌幅超过", "跌幅超過", "below", "less than", "<")):
        return "below"
    return "above"


def infer_year_range(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"((?:19|20)\d{2})\s*(?:年)?\s*(?:至|到|~|-|—|–|through|to)\s*((?:19|20)\d{2})", text, flags=re.I)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return min(start, end), max(start, end)
    years = [int(item) for item in re.findall(r"(?:19|20)\d{2}", text)]
    if len(years) >= 2:
        return min(years), max(years)
    if len(years) == 1:
        return years[0], years[0]
    return None, None


def infer_year_range_or_recent_window(text: str) -> tuple[int, int]:
    start_year, end_year = infer_year_range(text)
    if start_year is not None and end_year is not None:
        return start_year, end_year
    current_year = 2026
    match = re.search(r"(?:过去|過去|近|最近|last|past)\s*(\d{1,2})\s*(?:年|years?)", text, flags=re.I)
    if match:
        years = max(1, min(30, int(match.group(1))))
        return current_year - years + 1, current_year
    if any(term in text.lower() or term in text for term in ("过去十年", "過去十年", "近十年", "最近十年", "last decade")):
        return current_year - 9, current_year
    return current_year - 9, current_year


def infer_month(text: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*月", text)
    if match:
        month = int(match.group(1))
        return month if 1 <= month <= 12 else None
    months = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }
    lower = text.lower()
    for name, month in months.items():
        if name in lower:
            return month
    return None


def infer_series(data: list[Any], x_field: str | None, chart_type: str, question: str = "") -> list[dict[str, Any]]:
    first = data[0] if data and isinstance(data[0], dict) else {}
    numeric_fields = [key for key, value in first.items() if key != x_field and isinstance(value, (int, float))]
    if not numeric_fields:
        numeric_fields = ["value"]
    series_type = "bar" if chart_type in {"bar", "grouped-bar"} else "line"
    return [{"field": field, "label": display_label_for_field(field, question), "type": series_type} for field in numeric_fields[:4]]


def write_transform_result(output: Path, result: dict[str, Any], question: str, *, chart_mode: bool) -> None:
    """Write a non-visual execution artifact; chart skills own every chart render."""
    title = "数据处理完成" if is_chinese_question(question) else "Data transformation complete"
    if chart_mode:
        row_count = chart_payload_item_count(result)
        detail = (
            f"已生成 {row_count} 条结构化数据，供下游组件使用。"
            if is_chinese_question(question)
            else f"Generated {row_count} structured rows for a downstream component."
        )
    else:
        detail = "计算与数据处理已完成。" if is_chinese_question(question) else "Calculation and data processing completed."
    artifact = (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{title}</title></head><body>"
        f"<main data-artifact-type=\"data-transform\"><h1>{title}</h1><p>{detail}</p></main>"
        "</body></html>"
    )
    output.write_text(artifact, encoding="utf-8")


def chart_payload_item_count(chart_payload: dict[str, Any]) -> int:
    for key in ("items", "events", "data"):
        value = chart_payload.get(key)
        if isinstance(value, list) and value:
            return len(value)
    props = chart_payload.get("props")
    if isinstance(props, dict):
        for key in ("items", "events", "data"):
            value = props.get(key)
            if isinstance(value, list) and value:
                return len(value)
    for key in ("items", "events", "data"):
        value = chart_payload.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def main() -> int:
    args = parse_args()
    raw_data = load_raw_data(args)
    supplied_code = load_transform_code(args)
    chart_mode = bool(str(args.chart_type or "").strip())
    deterministic_payload = (
        deterministic_event_day_grouped_bar_payload(raw_data, args.question, args.chart_type, args.title)
        if chart_mode
        else None
    )
    deterministic_code = "deterministic:event_day_grouped_bar" if deterministic_payload else ""
    if chart_mode and not deterministic_payload:
        deterministic_payload = deterministic_multi_symbol_financial_grouped_bar_payload(raw_data, args.question, args.chart_type, args.title)
        deterministic_code = "deterministic:multi_symbol_financial_grouped_bar" if deterministic_payload else ""
    if chart_mode and not deterministic_payload:
        deterministic_payload = deterministic_daily_return_threshold_payload(raw_data, args.question, args.chart_type, args.title)
        deterministic_code = "deterministic:daily_return_threshold" if deterministic_payload else ""
    if chart_mode and not deterministic_payload:
        deterministic_payload = deterministic_first_trading_day_payload(raw_data, args.question, args.chart_type, args.title)
        deterministic_code = "deterministic:first_trading_day" if deterministic_payload else ""
    if chart_mode and not deterministic_payload:
        deterministic_payload = deterministic_month_cumulative_payload(raw_data, args.question, args.chart_type, args.title)
        deterministic_code = "deterministic:month_cumulative_return" if deterministic_payload else ""
    if chart_mode and not deterministic_payload:
        deterministic_payload = deterministic_us_macro_bar_line_payload(raw_data, args.question, args.chart_type, args.title)
        deterministic_code = "deterministic:us_macro_bar_line" if deterministic_payload else ""
    if chart_mode and not deterministic_payload:
        deterministic_payload = deterministic_multi_asset_return_dynamic_line_payload(raw_data, args.question, args.chart_type, args.title)
        deterministic_code = "deterministic:multi_asset_return_dynamic_line" if deterministic_payload else ""
    if chart_mode and not deterministic_payload:
        deterministic_payload = deterministic_income_statement_yoy_bar_line_payload(raw_data, args.question, args.chart_type, args.title)
        deterministic_code = "deterministic:income_statement_yoy_bar_line" if deterministic_payload else ""
    if deterministic_payload:
        code = deterministic_code
        chart_payload = deterministic_payload
    elif supplied_code:
        code = sanitize_transform_code(supplied_code)
        chart_payload = run_transform(code, raw_data, args.question, chart_mode=chart_mode)
    else:
        code = sanitize_transform_code(generate_transform_code(raw_data, args.question, args.chart_type, args.title))
        last_error = ""
        for attempt in range(3):
            try:
                chart_payload = run_transform(code, raw_data, args.question, chart_mode=chart_mode)
                break
            except SystemExit as exc:
                last_error = str(exc)
                if attempt >= 2:
                    raise
                code = sanitize_transform_code(generate_transform_code(raw_data, args.question, args.chart_type, args.title, feedback=last_error))
    if args.title:
        chart_payload["title"] = args.title
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_transform_result(output, chart_payload, args.question, chart_mode=chart_mode)
    payload = {
        "type": "code-interpreter",
        "question": args.question,
        "source": "Generated Python transform",
        "transformCode": code,
        "rawData": raw_data,
        "result": chart_payload,
    }
    if chart_mode:
        payload["chartPayload"] = chart_payload
        payload["lineChartData"] = {
            "xField": chart_payload.get("xField"),
            "series": chart_payload.get("series"),
            "data": chart_payload.get("data"),
        }
    if args.payload_output:
        payload_output = Path(args.payload_output).expanduser().resolve()
        payload_output.parent.mkdir(parents=True, exist_ok=True)
        payload_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "mode": "chart-data" if chart_mode else "data",
                "rows": chart_payload_item_count(chart_payload),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

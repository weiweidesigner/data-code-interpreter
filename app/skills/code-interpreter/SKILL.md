---
name: code-interpreter
description: Code transformation skill for auditable calculations, cleaning, aggregating, reshaping, filtering, sorting, and deriving structured payloads from raw JSON, CSV-like text, table data, or previous tool payloads. Select it with an upstream data source when the user explicitly asks to calculate, recompute, verify, or derive a result; it never renders charts itself.
---

# Code Interpreter

Use `scripts/code_interpreter.py` when an upstream payload or user-provided data must be cleaned, calculated, filtered, grouped, or reshaped.

## Script

- `code_interpreter.py`: generates and executes a constrained Python transform over raw data, then writes a structured payload and a non-visual execution artifact.

## Common Arguments

- `--question`: user intent for the transformation.
- `--data-json`, `--data`, `--raw-data`, or `--payload`: source data.
- `--chart-type`: optional downstream payload contract such as `line`, `bar`, `grouped-line`, `grouped-bar`, or `bar-line`; omit it for ordinary calculations and data processing.
- `--title`, `--output`, `--payload-output`.

## Routing Notes

Use this skill between data source tools and chart renderers when the requested output requires derived fields, grouping, filtering, ranking, threshold detection, date extraction, pivoting, or custom aggregation.

Distinguish a lookup from a computation by the user's intent. If the user only asks to retrieve an existing source metric, the data source can answer directly. If the user asks to calculate, recompute, verify, audit, or explain a derived result, fetch the necessary raw fields first and use this skill to perform the operation, even when a data source also exposes a similarly named precomputed field. The model must derive the operation from the question and upstream schema; do not use ticker-specific or metric-specific execution overrides.

Do not use this skill as a replacement for a real data source.
Do not select this skill merely to turn an already available metric or time series into a chart.
This skill never renders a chart, SVG, canvas, or visual preview. A visible chart requires an explicit downstream chart skill selected by the route.
When `--chart-type` is omitted, return the computation under `result` and do not emit `chartPayload` or `lineChartData`.
When `--chart-type` is present, it only controls the structured payload contract for a downstream chart skill; it still does not render a chart.
Do not use this skill for chart styling; downstream chart skills own all rendering contracts and styles.

## Standalone Schema Configuration

Set `CODE_INTERPRETER_SCHEMA_DIRS` to downstream schema directories separated by `os.pathsep`, or set `AICHART_SKILLS_DIR` to the host skill root. Schemas stay in their chart repositories. Without explicit configuration, lookup uses sibling skills at the logical CLI path and `app/skills` under the working directory.

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "app/skills/code-interpreter/scripts/code_interpreter.py"
spec = importlib.util.spec_from_file_location("standalone_interpreter", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_offline_cli_calculates_without_chart_or_model(tmp_path):
    target = tmp_path / "result.json"
    subprocess.run([sys.executable, str(SCRIPT), "--payload", str(ROOT / "examples/input.json"),
                    "--transform-file", str(ROOT / "examples/sum_transform.py"), "--question", "计算总和",
                    "--output", str(tmp_path / "result.html"), "--payload-output", str(target)],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    payload = json.loads(target.read_text())
    assert payload["result"] == {"total": 60, "count": 3}
    assert "chartPayload" not in payload
    assert "lineChartData" not in payload


def test_explicit_schema_directory(monkeypatch, tmp_path):
    expected = {"script": "line_chart.py", "payloadShape": {"data": "array"}}
    (tmp_path / "line_chart.schema.json").write_text(json.dumps(expected))
    monkeypatch.setenv("CODE_INTERPRETER_SCHEMA_DIRS", str(tmp_path))
    assert module.target_schema_for_chart_type("line") == expected


def test_host_skill_root_schema_lookup(monkeypatch, tmp_path):
    schemas = tmp_path / "generic-chart/schemas"
    schemas.mkdir(parents=True)
    (schemas / "bar_chart.schema.json").write_text('{"script":"bar_chart.py"}')
    monkeypatch.delenv("CODE_INTERPRETER_SCHEMA_DIRS", raising=False)
    monkeypatch.setenv("AICHART_SKILLS_DIR", str(tmp_path))
    assert module.target_schema_for_chart_type("bar")["script"] == "bar_chart.py"


def test_missing_schema_has_no_undefined_variable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert module.target_schema_for_chart_type("nonexistent_contract") == {}


def test_transform_rejects_filesystem_access():
    with pytest.raises(SystemExit, match="open"):
        module.run_transform('def transform(raw_data, question):\n    return {"data": open("secret")}', [], "", chart_mode=False)

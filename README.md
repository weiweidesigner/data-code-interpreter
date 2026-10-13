# Data Code Interpreter

Aime Chart 的独立数据计算与转换模块。支持清洗、聚合、排序、派生计算，以及按下游 Schema 生成图表 payload。输出结构化 JSON 和非图表的执行结果 HTML。

## 离线使用

建议 Python 3.11+：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app/skills/code-interpreter/scripts/code_interpreter.py --payload examples/input.json --transform-file examples/sum_transform.py --question '计算总和' --output outputs/result.html --payload-output outputs/result.json
```

上述调用无需模型凭据；预期结果为总和 60、数量 3。普通计算省略 `--chart-type`，结果在 `result` 中。只有下游需要图表 payload 时才指定 `--chart-type`；此模块不渲染图表。

不提供转换代码时，模块可调用兼容 OpenAI 的模型生成代码。通过进程环境配置 `DEEPSEEK_API_KEY`、`MODEL_BASE_URL` 和 `MODEL_INTERPRET_NAME`；也保留源码中的其他模型变量兼容入口。密钥不写入仓库。

## 下游 Schema

Schema 由图表仓库管理，此库不复制图表实现或 Schema。设置 `CODE_INTERPRETER_SCHEMA_DIRS` 为一个或多个 Schema 目录（以操作系统路径分隔符分隔）。也可通过 `AICHART_SKILLS_DIR` 指向包含 `generic-chart` 和 `business-chart-recall` 的宿主 Skill 目录。

```bash
export CODE_INTERPRETER_SCHEMA_DIRS=/path/to/data-visualization-charts/app/skills/generic-chart/schemas
```

未显式配置时，从逻辑脚本路径的兄弟 Skill 目录和当前工作目录的 `app/skills` 查找。找不到 Schema 时返回空契约；需要严格图表契约的调用方应先安装对应图表仓库并检查 Schema。完整调用规范见 `app/skills/code-interpreter/SKILL.md`。

转换代码使用 AST 限制和受限内置函数执行；它不是隔离进程沙箱，`--timeout` 当前不强制终止 Python 转换。仅执行可信输入与代码，资源隔离由宿主提供。

## 验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

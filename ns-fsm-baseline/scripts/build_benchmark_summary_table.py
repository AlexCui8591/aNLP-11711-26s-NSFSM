#!/usr/bin/env python3
"""Build the cross-benchmark summary table for the report.

Inputs are the figure-ready outputs from:
  - scripts/analyze_mctextworld_eval.py
  - scripts/analyze_robotouille_eval.py

The script writes CSV, Markdown, LaTeX, and a PNG/PDF table. It prefers computed
metrics from those analysis outputs. For paper drafting, it can fall back to the
static values currently used in the report table; rows using those defaults are
marked with source=paper_default in the CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent

DEFAULT_OUT_DIR = BASELINE_ROOT / "results/analysis/benchmark_summary_table"
DEFAULT_MCTEXTWORLD_OVERALL = (
    BASELINE_ROOT / "results/analysis/mctextworld_eval/data/overall_benchmark.csv"
)
DEFAULT_ROBOTOUILLE_SUITE = (
    BASELINE_ROOT / "results/analysis/robotouille_eval/data/suite_metrics.csv"
)

MCTEXTWORLD_AGENT_MAP = {
    "Qwen ReAct": "ReAct Qwen2.5-7B",
    "Qwen Reflexion": "Reflexion Qwen2.5-7B",
    "Qwen NS-FSM (projected)": r"NS-FSM Qwen2.5-7B^\dagger",
    "GPT-5-mini Reflexion": "Reflexion GPT-5-mini",
    "GPT-5-mini NS-FSM (projected)": r"NS-FSM GPT-5-mini^\dagger",
}

MCTEXTWORLD_ORDER = [
    "Qwen ReAct",
    "Qwen Reflexion",
    "Qwen NS-FSM (projected)",
    "GPT-5-mini Reflexion",
    "GPT-5-mini NS-FSM (projected)",
]

MCTEXTWORLD_DEFAULTS = {
    "Qwen ReAct": {"runs": 1005, "sr": "14.3%"},
    "Qwen Reflexion": {"runs": 1005, "sr": "26.5%"},
    "Qwen NS-FSM (projected)": {"runs": 1005, "sr": "39.7%"},
    "GPT-5-mini Reflexion": {"runs": 201, "sr": "99.0%"},
    "GPT-5-mini NS-FSM (projected)": {"runs": 164, "sr": "97.5%"},
}

ROBOTOUILLE_DEFAULT_ROWS = [
    {
        "benchmark": "Robotouille",
        "suite_setting": "Sync",
        "agent": "ReAct GPT-5-mini",
        "runs": 100,
        "sr": "0.51",
        "steps_over_lstar": "1.25",
        "budget_hit": "49%",
        "source": "paper_default",
    },
    {
        "benchmark": "Robotouille",
        "suite_setting": "Sync",
        "agent": r"NS-FSM GPT-5-mini^\dagger",
        "runs": 100,
        "sr": "0.67",
        "steps_over_lstar": "1.15",
        "budget_hit": "12%",
        "source": "paper_default",
    },
    {
        "benchmark": "Robotouille",
        "suite_setting": "Async",
        "agent": "ReAct GPT-5-mini",
        "runs": 100,
        "sr": "0.10",
        "steps_over_lstar": "1.46",
        "budget_hit": "90%",
        "source": "paper_default",
    },
    {
        "benchmark": "Robotouille",
        "suite_setting": "Async",
        "agent": r"NS-FSM GPT-5-mini^\dagger",
        "runs": 100,
        "sr": "0.39",
        "steps_over_lstar": "1.18",
        "budget_hit": "36%",
        "source": "paper_default",
    },
]

FIELDNAMES = [
    "benchmark",
    "suite_setting",
    "agent",
    "runs",
    "sr",
    "steps_over_lstar",
    "budget_hit",
    "source",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the combined MC-TextWorld + Robotouille summary table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mctextworld-overall", type=Path, default=DEFAULT_MCTEXTWORLD_OVERALL)
    parser.add_argument("--robotouille-suite", type=Path, default=DEFAULT_ROBOTOUILLE_SUITE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--no-paper-defaults",
        action="store_true",
        help="Fail instead of filling missing rows from the report's current table values.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Rendered table formats.",
    )
    parser.add_argument("--no-render", action="store_true", help="Skip PNG/PDF/SVG rendering.")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


def safe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_percent_from_rate(value: Any, digits: int = 1) -> str:
    rate = safe_float(value)
    if rate is None or math.isnan(rate):
        return "--"
    if rate > 1.0:
        rate /= 100.0
    return f"{rate * 100:.{digits}f}%"


def format_decimal(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    if number is None or math.isnan(number):
        return "--"
    return f"{number:.{digits}f}"


def build_mctextworld_rows(path: Path, allow_defaults: bool) -> list[dict[str, Any]]:
    raw_by_label: dict[str, dict[str, str]] = {}
    source = "computed"
    if path.exists():
        raw_by_label = {row.get("label", ""): row for row in read_csv(path)}
    elif not allow_defaults:
        raise FileNotFoundError(f"Missing MC-TextWorld overall CSV: {path}")
    else:
        source = "paper_default"

    rows: list[dict[str, Any]] = []
    for label in MCTEXTWORLD_ORDER:
        raw = raw_by_label.get(label, {})
        fallback = MCTEXTWORLD_DEFAULTS[label]
        runs = safe_int(raw.get("total_runs")) or fallback["runs"]
        sr = (
            format_percent_from_rate(raw.get("success_rate"), digits=1)
            if raw
            else fallback["sr"]
        )
        row_source = "computed" if raw else source
        if raw and not raw.get("total_runs"):
            row_source = "computed_with_default_runs"
        rows.append(
            {
                "benchmark": "MC-TextWorld",
                "suite_setting": "67 goals",
                "agent": MCTEXTWORLD_AGENT_MAP[label],
                "runs": runs,
                "sr": sr,
                "steps_over_lstar": "--",
                "budget_hit": "--",
                "source": row_source,
            }
        )
    return rows


def suite_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    suite = str(row.get("suite_setting", "")).lower()
    agent = str(row.get("agent", ""))
    suite_key = {"sync": 0, "async": 1}.get(suite, 9)
    agent_key = 0 if agent.startswith("ReAct") else 1
    return suite_key, agent_key


def robotouille_agent(method: str) -> str:
    if method == "NS-FSM":
        return r"NS-FSM GPT-5-mini^\dagger"
    return f"{method} GPT-5-mini"


def build_robotouille_rows(path: Path, allow_defaults: bool) -> list[dict[str, Any]]:
    if not path.exists():
        if not allow_defaults:
            raise FileNotFoundError(f"Missing Robotouille suite CSV: {path}")
        return [dict(row) for row in ROBOTOUILLE_DEFAULT_ROWS]

    rows: list[dict[str, Any]] = []
    for raw in read_csv(path):
        method = raw.get("method", "")
        suite = raw.get("suite_label") or raw.get("suite", "")
        if not method or not suite:
            continue
        rows.append(
            {
                "benchmark": "Robotouille",
                "suite_setting": "Sync" if suite.lower().startswith("sync") else "Async",
                "agent": robotouille_agent(method),
                "runs": safe_int(raw.get("n_runs")) or safe_int(raw.get("runs")) or "--",
                "sr": format_decimal(raw.get("success_rate"), digits=2),
                "steps_over_lstar": format_decimal(raw.get("steps_over_optimal"), digits=2),
                "budget_hit": format_percent_from_rate(raw.get("budget_hit_rate"), digits=0),
                "source": "computed",
            }
        )
    return sorted(rows, key=suite_sort_key)


def markdown_agent(agent: str) -> str:
    return agent.replace(r"^\dagger", "†")


def latex_escape(text: Any) -> str:
    raw = str(text)
    raw = raw.replace(r"^\dagger", r"$^\dagger$")
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
    }
    protected = raw.replace(r"$^\dagger$", "DAGGERMARK")
    for old, new in replacements.items():
        protected = protected.replace(old, new)
    return protected.replace("DAGGERMARK", r"$^\dagger$")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Benchmark | Suite / Setting | Agent | Runs | SR | Steps/L* | Budget-hit | Source |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["benchmark"]),
                    str(row["suite_setting"]),
                    markdown_agent(str(row["agent"])),
                    str(row["runs"]),
                    str(row["sr"]),
                    str(row["steps_over_lstar"]),
                    str(row["budget_hit"]),
                    str(row["source"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Benchmark & Suite / Setting & Agent & Runs & SR & Steps/$L^*$ & Budget-hit \\",
        r"\midrule",
    ]
    for idx, row in enumerate(rows):
        if idx == 5:
            lines.append(r"\midrule")
        lines.append(
            " & ".join(
                [
                    latex_escape(row["benchmark"]),
                    latex_escape(row["suite_setting"]),
                    latex_escape(row["agent"]),
                    latex_escape(row["runs"]),
                    latex_escape(row["sr"]),
                    latex_escape(row["steps_over_lstar"]),
                    latex_escape(row["budget_hit"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_table(path_stem: Path, rows: list[dict[str, Any]], formats: list[str]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for rendering. Use --no-render to skip.") from exc

    columns = ["Benchmark", "Suite / Setting", "Agent", "Runs", "SR", r"Steps/$L^*$", "Budget-hit"]
    cell_text = [
        [
            row["benchmark"],
            row["suite_setting"],
            markdown_agent(str(row["agent"])),
            row["runs"],
            row["sr"],
            row["steps_over_lstar"],
            row["budget_hit"],
        ]
        for row in rows
    ]

    fig_height = max(2.8, 0.43 * len(rows) + 0.9)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.16, 0.16, 0.28, 0.08, 0.08, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.35)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8 if row_idx in {0, 6} else 0.25)
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f3f3f3")
        elif row_idx >= 7:
            cell.set_facecolor("#fbfbfb")
        if col_idx >= 3:
            cell.get_text().set_ha("right")

    written: list[Path] = []
    for fmt in formats:
        out = path_stem.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    allow_defaults = not args.no_paper_defaults

    rows = build_mctextworld_rows(args.mctextworld_overall, allow_defaults)
    rows.extend(build_robotouille_rows(args.robotouille_suite, allow_defaults))

    out_dir = args.out_dir
    write_csv(out_dir / "benchmark_summary_table.csv", rows)
    write_json(out_dir / "benchmark_summary_table.json", rows)
    write_markdown(out_dir / "benchmark_summary_table.md", rows)
    write_latex(out_dir / "benchmark_summary_table.tex", rows)

    rendered: list[Path] = []
    if not args.no_render:
        rendered = render_table(out_dir / "benchmark_summary_table", rows, args.formats)

    print(f"Wrote benchmark summary table to: {out_dir}")
    for item in rendered:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

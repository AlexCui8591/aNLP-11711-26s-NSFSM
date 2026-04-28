#!/usr/bin/env python3
"""Build MC-TextWorld evaluation metrics and report figures.

This script complements scripts/error_analysis.py. The older script computes
per-run error analysis and a couple of basic plots; this script packages those
metrics into the report-style panels used for the NS-FSM writeup:

  A. Overall benchmark comparison
  B. Success rate by dependency depth
  C. ReAct failure attribution and NS-FSM mechanism

By default it reads the custom analysis files already produced in this repo:
results/full/custom/react_analysis.json and reflexion_analysis.json. Projected
NS-FSM and GPT-5-mini numbers can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent

DEFAULT_REACT_ANALYSIS = BASELINE_ROOT / "results/full/custom/react_analysis.json"
DEFAULT_REFLEXION_ANALYSIS = BASELINE_ROOT / "results/full/custom/reflexion_analysis.json"
DEFAULT_OUT_DIR = BASELINE_ROOT / "results/analysis/mctextworld_eval"

GROUP_ORDER = ["Wooden", "Stone", "Iron", "Redstone", "Golden", "Armor", "Diamond"]
GROUP_DEPTHS = {
    "Wooden": 3,
    "Stone": 7,
    "Iron": 12,
    "Redstone": 14,
    "Golden": 15,
    "Armor": 15,
    "Diamond": 18,
}

DEFAULT_OVERALL_PROJECTIONS = [
    {
        "label": "Qwen NS-FSM (projected)",
        "success_rate": 0.397,
        "status": "projected",
        "total_runs": 1005,
    },
    {
        "label": "GPT-5-mini Reflexion",
        "success_rate": 0.990,
        "status": "measured",
        "total_runs": 201,
    },
    {
        "label": "GPT-5-mini NS-FSM (projected)",
        "success_rate": 0.975,
        "status": "projected",
        "total_runs": 164,
    },
]

DEFAULT_NSFSM_DEPTH_PROJECTION = {
    "Wooden": 0.910,
    "Stone": 0.784,
    "Iron": 0.228,
    "Redstone": 0.186,
    "Golden": 0.238,
    "Armor": 0.246,
    "Diamond": 0.150,
}

ERROR_LABELS = {
    "tool_dependency_error": "tool_dependency_error",
    "plan_knowledge_error": "plan_knowledge_error",
    "resource_shortage": "resource_shortage",
    "sequencing_error": "sequencing_error",
    "invalid_action": "invalid_action",
    "none": "none",
}

MECHANISM_MAP = [
    (
        "tool_dependency_error",
        "Symbolic planner enforces\ndependency order",
    ),
    (
        "plan_knowledge_error",
        "Recipe library replaces\nin-context recall",
    ),
    (
        "resource_shortage",
        "Verifier blocks\npremature actions",
    ),
]


@dataclass
class MethodAnalysis:
    label: str
    status: str
    success_rate: float
    total_runs: int | None
    n_success: int | None
    n_fail: int | None
    group_rows: list[dict[str, Any]]
    error_distribution: dict[str, int]
    termination_distribution: dict[str, int]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_rate(value: Any) -> float:
    """Return a rate in [0, 1] when possible.

    Inputs above 1 are treated as percentages, so 39.7 becomes 0.397.
    """

    if value is None:
        return float("nan")
    rate = float(value)
    if rate > 1.0:
        rate /= 100.0
    return rate


def pct(rate: float) -> float:
    return rate * 100.0


def first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalize_group_rows(raw_group_stats: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if isinstance(raw_group_stats, list):
        iterable = raw_group_stats
    elif isinstance(raw_group_stats, dict):
        iterable = []
        for group, value in raw_group_stats.items():
            if isinstance(value, dict):
                item = {"group": group, **value}
            else:
                item = {"group": group, "success_rate": value}
            iterable.append(item)
    else:
        iterable = []

    for item in iterable:
        if not isinstance(item, dict):
            continue
        group = str(first_present(item, ["group", "name", "label"]) or "")
        if not group:
            continue
        success_rate = normalize_rate(
            first_present(item, ["success_rate", "sr", "rate", "overall_success_rate"])
        )
        if math.isnan(success_rate):
            n_runs = first_present(item, ["n_runs", "total_runs", "total", "count"])
            n_success = first_present(item, ["n_success", "successes", "success"])
            if n_runs:
                success_rate = float(n_success or 0) / float(n_runs)
        depth = safe_int(first_present(item, ["depth", "dependency_depth"]))
        if depth is None:
            depth = GROUP_DEPTHS.get(group)
        rows.append(
            {
                "group": group,
                "depth": depth,
                "success_rate": success_rate,
                "success_rate_pct": pct(success_rate),
                "n_runs": safe_int(first_present(item, ["n_runs", "total_runs", "total"])),
                "n_success": safe_int(first_present(item, ["n_success", "successes"])),
            }
        )

    return sorted(rows, key=lambda r: GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99)


def parse_analysis(path: Path, label: str, status: str) -> MethodAnalysis:
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis file: {path}")

    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    overall = data.get("overall") if isinstance(data.get("overall"), dict) else data
    success_rate = normalize_rate(
        first_present(overall, ["success_rate", "overall_success_rate", "sr"])
    )

    total_runs = safe_int(
        first_present(overall, ["total_runs", "total_episodes", "n_runs", "episodes"])
    )
    n_success = safe_int(
        first_present(overall, ["n_success", "total_success", "successes"])
    )
    if math.isnan(success_rate) and total_runs:
        success_rate = float(n_success or 0) / float(total_runs)
    if n_success is None and total_runs is not None and not math.isnan(success_rate):
        n_success = round(total_runs * success_rate)

    n_fail = safe_int(first_present(overall, ["n_fail", "total_fail", "failures"]))
    if n_fail is None and total_runs is not None and n_success is not None:
        n_fail = total_runs - n_success

    group_rows = normalize_group_rows(data.get("group_stats"))
    error_distribution = {
        str(k): int(v) for k, v in (data.get("error_distribution") or {}).items()
    }
    termination_distribution = {
        str(k): int(v) for k, v in (data.get("termination_distribution") or {}).items()
    }

    return MethodAnalysis(
        label=label,
        status=status,
        success_rate=success_rate,
        total_runs=total_runs,
        n_success=n_success,
        n_fail=n_fail,
        group_rows=group_rows,
        error_distribution=error_distribution,
        termination_distribution=termination_distribution,
    )


def parse_projection_spec(spec: str) -> dict[str, Any]:
    parts = spec.split(":")
    if len(parts) not in {2, 3, 4}:
        raise argparse.ArgumentTypeError(
            "Projection must be LABEL:SUCCESS_RATE[:STATUS[:N_RUNS]]"
        )
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError("Projection label cannot be empty")
    status = parts[2].strip() if len(parts) >= 3 else "projected"
    return {
        "label": label,
        "success_rate": normalize_rate(parts[1]),
        "status": status,
        "total_runs": safe_int(parts[3]) if len(parts) == 4 else None,
    }


def parse_group_rate_spec(spec: str) -> tuple[str, float]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Depth projection must be GROUP=SUCCESS_RATE")
    group, rate = spec.split("=", 1)
    group = group.strip()
    if group not in GROUP_DEPTHS:
        raise argparse.ArgumentTypeError(
            f"Unknown group {group!r}. Expected one of: {', '.join(GROUP_ORDER)}"
        )
    return group, normalize_rate(rate)


def build_overall_rows(
    react: MethodAnalysis,
    reflexion: MethodAnalysis,
    projections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for analysis in [react, reflexion]:
        rows.append(
            {
                "label": analysis.label,
                "status": analysis.status,
                "success_rate": analysis.success_rate,
                "success_rate_pct": pct(analysis.success_rate),
                "total_runs": analysis.total_runs,
                "n_success": analysis.n_success,
                "n_fail": analysis.n_fail,
            }
        )
    for proj in projections:
        rate = normalize_rate(proj["success_rate"])
        rows.append(
            {
                "label": proj["label"],
                "status": proj.get("status", "projected"),
                "success_rate": rate,
                "success_rate_pct": pct(rate),
                "total_runs": proj.get("total_runs"),
                "n_success": None,
                "n_fail": None,
            }
        )
    return rows


def build_depth_rows(
    react: MethodAnalysis,
    reflexion: MethodAnalysis,
    nsfsm_depth_projection: dict[str, float],
    nsfsm_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis in [react, reflexion]:
        for item in analysis.group_rows:
            if item["group"] not in GROUP_ORDER:
                continue
            rows.append(
                {
                    "method": analysis.label,
                    "status": analysis.status,
                    "group": item["group"],
                    "depth": item["depth"],
                    "success_rate": item["success_rate"],
                    "success_rate_pct": item["success_rate_pct"],
                    "n_runs": item.get("n_runs"),
                    "n_success": item.get("n_success"),
                }
            )

    for group in GROUP_ORDER:
        if group not in nsfsm_depth_projection:
            continue
        rate = normalize_rate(nsfsm_depth_projection[group])
        rows.append(
            {
                "method": nsfsm_label,
                "status": "projected",
                "group": group,
                "depth": GROUP_DEPTHS[group],
                "success_rate": rate,
                "success_rate_pct": pct(rate),
                "n_runs": None,
                "n_success": None,
            }
        )

    return sorted(
        rows,
        key=lambda r: (
            [react.label, reflexion.label, nsfsm_label].index(r["method"])
            if r["method"] in [react.label, reflexion.label, nsfsm_label]
            else 99,
            GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99,
        ),
    )


def build_failure_rows(react: MethodAnalysis) -> list[dict[str, Any]]:
    total_failures = react.n_fail
    if total_failures is None:
        total_failures = sum(
            count
            for error, count in react.error_distribution.items()
            if error not in {"success", "none"}
        )

    rows = []
    for error, count in sorted(
        react.error_distribution.items(),
        key=lambda kv: (-kv[1], kv[0]),
    ):
        rows.append(
            {
                "method": react.label,
                "error_type": ERROR_LABELS.get(error, error),
                "count": count,
                "denominator": total_failures,
                "percent_of_failures": (100.0 * count / total_failures) if total_failures else 0.0,
                "included_in_main_stack": error not in {"none", "success"},
            }
        )
    return rows


def build_termination_rows(react: MethodAnalysis) -> list[dict[str, Any]]:
    total = sum(react.termination_distribution.values())
    rows = []
    for reason, count in sorted(
        react.termination_distribution.items(),
        key=lambda kv: (-kv[1], kv[0]),
    ):
        rows.append(
            {
                "method": react.label,
                "termination_reason": reason,
                "count": count,
                "percent": (100.0 * count / total) if total else 0.0,
            }
        )
    return rows


def write_markdown_summary(
    path: Path,
    overall_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    termination_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MC-TextWorld Evaluation Summary",
        "",
        "## Overall Benchmark Comparison",
        "",
        "| Method | Status | Success Rate | Runs | Successes |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['label']} | {row['status']} | {row['success_rate_pct']:.1f}% | "
            f"{row.get('total_runs') or ''} | {row.get('n_success') or ''} |"
        )

    lines.extend(
        [
            "",
            "## Success Rate by Dependency Depth",
            "",
            "| Method | Group | Depth | Success Rate |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in depth_rows:
        lines.append(
            f"| {row['method']} | {row['group']} | {row['depth']} | "
            f"{row['success_rate_pct']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## ReAct Failure Attribution",
            "",
            "| Error Type | Count | Percent of Failures |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in failure_rows:
        lines.append(
            f"| {row['error_type']} | {row['count']} | "
            f"{row['percent_of_failures']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## ReAct Termination Reasons",
            "",
            "| Termination Reason | Count | Percent |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in termination_rows:
        lines.append(
            f"| {row['termination_reason']} | {row['count']} | {row['percent']:.1f}% |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Re-run with --no-plots to only export metrics."
        ) from exc
    return plt, patches


def method_color(label: str) -> str:
    if "ReAct" in label:
        return "#8f8f8f"
    if "Reflexion" in label and "GPT-5" not in label:
        return "#169c2f"
    if "NS-FSM" in label and "GPT-5" not in label:
        return "#1458ff"
    if "GPT-5" in label and "Reflexion" in label:
        return "#ff6a00"
    if "GPT-5" in label and "NS-FSM" in label:
        return "#0d2c82"
    return "#4c78a8"


def short_label(label: str) -> str:
    label = label.replace(" (projected)", "\n(projected)")
    label = label.replace("Qwen ", "Qwen\n")
    label = label.replace("GPT-5-mini ", "GPT-5-mini\n")
    return label


def save_figure(fig: Any, out_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    out_paths = []
    for fmt in formats:
        out_path = out_dir / f"{stem}.{fmt}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        out_paths.append(out_path)
    return out_paths


def draw_overall_panel(ax: Any, overall_rows: list[dict[str, Any]]) -> None:
    plt, _ = require_matplotlib()
    labels = [row["label"] for row in overall_rows]
    values = [row["success_rate_pct"] for row in overall_rows]
    colors = [method_color(label) for label in labels]
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=1.2)

    for bar, row in zip(bars, overall_rows):
        if row["status"] == "projected":
            bar.set_hatch("///")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{row['success_rate_pct']:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    measured_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#c0c0c0", edgecolor="black")
    projected_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="#1458ff", edgecolor="black", hatch="///"
    )
    ax.legend(
        [measured_patch, projected_patch],
        ["Measured", "Projected"],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        fontsize=8,
    )
    ax.set_title("Overall benchmark comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Success rate (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([short_label(label) for label in labels], fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def draw_depth_panel(
    ax: Any,
    depth_rows: list[dict[str, Any]],
    method_order: list[str],
) -> None:
    x_positions = {group: i for i, group in enumerate(GROUP_ORDER)}
    styles = {
        method_order[0]: {"marker": "o", "linestyle": "-", "color": method_color(method_order[0])},
        method_order[1]: {"marker": "s", "linestyle": "-", "color": method_color(method_order[1])},
        method_order[2]: {"marker": "^", "linestyle": "--", "color": method_color(method_order[2])},
    }

    for method in method_order:
        rows = [row for row in depth_rows if row["method"] == method]
        rows = sorted(rows, key=lambda row: x_positions.get(row["group"], 99))
        xs = [x_positions[row["group"]] for row in rows]
        ys = [row["success_rate_pct"] for row in rows]
        ax.plot(xs, ys, linewidth=1.8, markersize=5.5, label=method, **styles[method])
        for x, y in zip(xs, ys):
            if x == 0 and method == method_order[2]:
                offset = -7.0
                va = "top"
            elif x == 0 and method == method_order[1]:
                offset = 6.0
                va = "bottom"
            else:
                offset = 3.2 if y < 92 else -7.0
                va = "bottom" if y < 92 else "top"
            ax.text(
                x,
                y + offset,
                f"{y:.1f}",
                color=styles[method]["color"],
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold" if method == method_order[2] else "normal",
            )

    tick_labels = [f"{GROUP_DEPTHS[group]}\n{group}" for group in GROUP_ORDER]
    ax.set_title("Success rate vs. dependency depth", fontsize=13, fontweight="bold")
    ax.set_ylabel("Success rate (%)", fontsize=11)
    ax.set_xlabel("Dependency depth", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(GROUP_ORDER) - 0.5)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, frameon=True, edgecolor="black")


def draw_failure_panel(
    ax: Any,
    failure_rows: list[dict[str, Any]],
    termination_rows: list[dict[str, Any]],
    title: str = "Qwen ReAct failure attribution and NS-FSM mechanism",
) -> None:
    _, patches = require_matplotlib()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=13, fontweight="bold")

    visible_rows = [
        row
        for row in failure_rows
        if row["included_in_main_stack"] and row["count"] > 0
    ]
    order = ["tool_dependency_error", "plan_knowledge_error", "resource_shortage"]
    visible_rows.sort(
        key=lambda row: order.index(row["error_type"])
        if row["error_type"] in order
        else len(order)
    )

    denominator = visible_rows[0]["denominator"] if visible_rows else 0
    total_visible = sum(row["count"] for row in visible_rows) or 1
    ax.text(
        0.24,
        0.84,
        f"Qwen2.5-7B ReAct failures (n = {denominator})",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    colors = {
        "tool_dependency_error": "#ef1d2d",
        "plan_knowledge_error": "#ff6f00",
        "resource_shortage": "#ffb000",
        "sequencing_error": "#db3a34",
        "invalid_action": "#ad1457",
    }
    x0 = 0.03
    y0 = 0.58
    width = 0.46
    height = 0.18
    cursor = x0
    for row in visible_rows:
        segment_width = width * row["count"] / total_visible
        rect = patches.FancyBboxPatch(
            (cursor, y0),
            segment_width,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            linewidth=0,
            facecolor=colors.get(row["error_type"], "#999999"),
        )
        ax.add_patch(rect)
        percent = 100.0 * row["count"] / denominator if denominator else 0.0
        ax.text(
            cursor + segment_width / 2,
            y0 + height / 2,
            f"{percent:.0f}%\n({row['count']})",
            color="white",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        cursor += segment_width

    legend_y = 0.42
    legend_x = 0.04
    for i, row in enumerate(visible_rows):
        lx = legend_x + i * 0.16
        ax.add_patch(
            patches.Rectangle(
                (lx, legend_y),
                0.015,
                0.04,
                color=colors.get(row["error_type"], "#999999"),
                transform=ax.transAxes,
            )
        )
        ax.text(
            lx + 0.022,
            legend_y + 0.02,
            row["error_type"],
            ha="left",
            va="center",
            fontsize=7,
        )

    termination_map = {row["termination_reason"]: row["count"] for row in termination_rows}
    term_text = (
        "Termination: "
        f"{termination_map.get('dead_loop', 0)} dead_loop / "
        f"{termination_map.get('max_steps', 0)} max_steps / "
        f"{termination_map.get('llm_error', 0)} llm_error"
    )
    ax.text(
        0.24,
        0.24,
        term_text,
        ha="center",
        va="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#555555"),
    )

    ax.plot([0.53, 0.53], [0.18, 0.88], color="#777777", linestyle="--", linewidth=1)
    ax.text(0.74, 0.86, "Why NS-FSM should help", ha="center", va="center", fontsize=10, fontweight="bold")

    for i, (error_type, mechanism) in enumerate(MECHANISM_MAP):
        y = 0.72 - i * 0.20
        ax.text(
            0.61,
            y,
            error_type,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#fff5f5",
                edgecolor=colors.get(error_type, "#ef1d2d"),
                linewidth=1.1,
            ),
        )
        ax.annotate(
            "",
            xy=(0.72, y),
            xytext=(0.67, y),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2),
        )
        ax.text(
            0.84,
            y,
            mechanism,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#f4f7ff",
                edgecolor="#1458ff",
                linewidth=1.1,
            ),
        )


def add_panel_tag(ax: Any, tag: str) -> None:
    ax.text(
        -0.075,
        1.03,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="black", edgecolor="black", pad=3),
        clip_on=False,
        zorder=10,
    )


def make_plots(
    out_dir: Path,
    overall_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    termination_rows: list[dict[str, Any]],
    formats: list[str],
    method_order: list[str],
) -> list[Path]:
    plt, _ = require_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )

    figure_dir = out_dir / "figures"
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    draw_overall_panel(ax, overall_rows)
    written.extend(save_figure(fig, figure_dir, "overall_benchmark_comparison", formats))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_depth_panel(ax, depth_rows, method_order)
    written.extend(save_figure(fig, figure_dir, "success_rate_vs_dependency_depth", formats))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    draw_failure_panel(ax, failure_rows, termination_rows)
    written.extend(save_figure(fig, figure_dir, "react_failure_attribution", formats))
    plt.close(fig)

    fig = plt.figure(figsize=(14.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.42, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    draw_overall_panel(ax_a, overall_rows)
    draw_depth_panel(ax_b, depth_rows, method_order)
    draw_failure_panel(ax_c, failure_rows, termination_rows)
    add_panel_tag(ax_a, "A")
    add_panel_tag(ax_b, "B")
    add_panel_tag(ax_c, "C")
    written.extend(save_figure(fig, figure_dir, "mctextworld_eval_dashboard", formats))
    plt.close(fig)

    return written


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MC-TextWorld metrics tables and report figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--react-analysis", type=Path, default=DEFAULT_REACT_ANALYSIS)
    parser.add_argument("--reflexion-analysis", type=Path, default=DEFAULT_REFLEXION_ANALYSIS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--projection",
        action="append",
        type=parse_projection_spec,
        default=[],
        help="Add/override an overall projected row: LABEL:SUCCESS_RATE[:STATUS[:N_RUNS]].",
    )
    parser.add_argument(
        "--skip-default-projections",
        action="store_true",
        help="Do not include the report default projected/GPT rows.",
    )
    parser.add_argument(
        "--depth-projection",
        action="append",
        type=parse_group_rate_spec,
        default=[],
        help="Override projected NS-FSM depth rate as GROUP=SUCCESS_RATE.",
    )
    parser.add_argument(
        "--depth-projection-json",
        type=Path,
        default=None,
        help="Optional JSON object mapping group names to projected NS-FSM success rates.",
    )
    parser.add_argument(
        "--nsfsm-depth-label",
        default="Qwen NS-FSM (projected)",
        help="Label used for the depth-curve projected NS-FSM series.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to write.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Only export metrics tables.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    out_dir = args.out_dir
    data_dir = out_dir / "data"

    react = parse_analysis(args.react_analysis, "Qwen ReAct", "measured")
    reflexion = parse_analysis(args.reflexion_analysis, "Qwen Reflexion", "measured")

    projections = []
    if not args.skip_default_projections:
        projections.extend(DEFAULT_OVERALL_PROJECTIONS)
    projections.extend(args.projection)

    nsfsm_depth_projection = dict(DEFAULT_NSFSM_DEPTH_PROJECTION)
    if args.depth_projection_json is not None:
        loaded = load_json(args.depth_projection_json)
        if not isinstance(loaded, dict):
            raise ValueError("--depth-projection-json must contain a JSON object")
        for group, rate in loaded.items():
            if group not in GROUP_DEPTHS:
                raise ValueError(f"Unknown group in depth projection JSON: {group}")
            nsfsm_depth_projection[group] = normalize_rate(rate)
    for group, rate in args.depth_projection:
        nsfsm_depth_projection[group] = rate

    overall_rows = build_overall_rows(react, reflexion, projections)
    depth_rows = build_depth_rows(
        react,
        reflexion,
        nsfsm_depth_projection,
        args.nsfsm_depth_label,
    )
    failure_rows = build_failure_rows(react)
    termination_rows = build_termination_rows(react)
    method_order = [react.label, reflexion.label, args.nsfsm_depth_label]

    write_json(data_dir / "overall_benchmark.json", overall_rows)
    write_csv(
        data_dir / "overall_benchmark.csv",
        overall_rows,
        ["label", "status", "success_rate", "success_rate_pct", "total_runs", "n_success", "n_fail"],
    )
    write_json(data_dir / "success_rate_by_depth.json", depth_rows)
    write_csv(
        data_dir / "success_rate_by_depth.csv",
        depth_rows,
        ["method", "status", "group", "depth", "success_rate", "success_rate_pct", "n_runs", "n_success"],
    )
    write_json(data_dir / "react_failure_attribution.json", failure_rows)
    write_csv(
        data_dir / "react_failure_attribution.csv",
        failure_rows,
        ["method", "error_type", "count", "denominator", "percent_of_failures", "included_in_main_stack"],
    )
    write_json(data_dir / "react_termination_reasons.json", termination_rows)
    write_csv(
        data_dir / "react_termination_reasons.csv",
        termination_rows,
        ["method", "termination_reason", "count", "percent"],
    )
    write_markdown_summary(
        out_dir / "metrics_summary.md",
        overall_rows,
        depth_rows,
        failure_rows,
        termination_rows,
    )

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = make_plots(
            out_dir,
            overall_rows,
            depth_rows,
            failure_rows,
            termination_rows,
            args.formats,
            method_order,
        )

    print(f"Wrote metrics to: {data_dir}")
    print(f"Wrote summary to: {out_dir / 'metrics_summary.md'}")
    if plot_paths:
        print("Wrote figures:")
        for path in plot_paths:
            print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

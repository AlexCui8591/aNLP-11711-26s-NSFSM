#!/usr/bin/env python3
"""Generate CSV, Markdown, and PNG visualizations for an NS-FSM rollout."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    result_root = Path(args.results_dir) if args.results_dir else ROOT / "results" / "full" / args.tag
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "analysis" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(result_root)
    rows = [flatten_result(result) for result in results]
    write_csv(output_dir / "rollout_runs.csv", rows, RUN_FIELDS)
    write_csv(output_dir / "summary_by_task.csv", summarize(rows, "task_id"), SUMMARY_FIELDS("task_id"))
    write_csv(output_dir / "summary_by_group.csv", summarize(rows, "group"), SUMMARY_FIELDS("group"))
    write_csv(output_dir / "summary_by_termination.csv", termination_rows(rows), ["termination", "runs"])

    plot_paths = make_plots(rows, output_dir)
    report_path = write_report(output_dir / "rollout_visual_report.md", rows, plot_paths, result_root)
    log_wandb(args, rows, plot_paths, report_path)
    print(output_dir)


RUN_FIELDS = [
    "task_id",
    "run_id",
    "group",
    "task_source",
    "success",
    "termination",
    "total_steps",
    "blocked_action_count",
    "fallback_action_count",
    "llm_action_count",
    "llm_valid_decision_count",
    "planner_action_count",
]


def SUMMARY_FIELDS(key: str) -> list[str]:
    return [
        key,
        "runs",
        "successes",
        "success_rate",
        "avg_steps",
        "blocked_action_rate",
        "fallback_usage_rate",
        "llm_valid_decision_rate",
        "planner_decision_rate",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="nsfsm-mctextworld")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", default="online")
    return parser.parse_args()


def load_results(root: Path) -> list[dict[str, Any]]:
    results = []
    for path in root.rglob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if {"dataset", "task_id", "success", "trajectory"}.issubset(payload):
            payload["_path"] = str(path)
            results.append(payload)
    return sorted(results, key=lambda item: (str(item.get("task_id", "")), int(item.get("run_id", 0))))


def flatten_result(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata", {})
    task_metadata = metadata.get("task_spec", {}).get("metadata", {})
    metrics = metadata.get("execution_metrics") or infer_execution_metrics(result)
    return {
        "task_id": result.get("task_id", ""),
        "run_id": result.get("run_id", ""),
        "group": task_metadata.get("group", ""),
        "task_source": task_metadata.get("task_source", ""),
        "success": bool(result.get("success", False)),
        "termination": result.get("termination", ""),
        "total_steps": int(result.get("total_steps", 0)),
        "blocked_action_count": int(result.get("blocked_action_count", 0)),
        "fallback_action_count": int(result.get("fallback_action_count", 0)),
        "llm_action_count": int(metrics.get("llm_action_count", 0)),
        "llm_valid_decision_count": int(metrics.get("llm_valid_decision_count", 0)),
        "planner_action_count": int(metrics.get("planner_action_count", 0)),
    }


def infer_execution_metrics(result: Mapping[str, Any]) -> dict[str, int]:
    trajectory = list(result.get("trajectory", []))
    return {
        "llm_action_count": sum(
            1
            for step in trajectory
            if dict(step.get("proposal", {})).get("source") == "llm"
            and dict(step.get("proposal", {})).get("action")
        ),
        "llm_valid_decision_count": sum(1 for step in trajectory if step.get("decision_source") == "llm"),
        "planner_action_count": sum(1 for step in trajectory if step.get("decision_source") == "planner"),
    }


def summarize(rows: list[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)

    output = []
    for value, items in sorted(grouped.items()):
        runs = len(items)
        successes = sum(1 for item in items if item.get("success"))
        steps = sum(int(item.get("total_steps", 0)) for item in items)
        blocked = sum(int(item.get("blocked_action_count", 0)) for item in items)
        fallback = sum(int(item.get("fallback_action_count", 0)) for item in items)
        llm = sum(int(item.get("llm_valid_decision_count", 0)) for item in items)
        planner = sum(int(item.get("planner_action_count", 0)) for item in items)
        output.append(
            {
                key: value,
                "runs": runs,
                "successes": successes,
                "success_rate": successes / runs if runs else 0.0,
                "avg_steps": steps / runs if runs else 0.0,
                "blocked_action_rate": blocked / steps if steps else 0.0,
                "fallback_usage_rate": fallback / steps if steps else 0.0,
                "llm_valid_decision_rate": llm / steps if steps else 0.0,
                "planner_decision_rate": planner / steps if steps else 0.0,
            }
        )
    return output


def termination_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("termination", "")) for row in rows)
    return [{"termination": key, "runs": value} for key, value in sorted(counts.items())]


def make_plots(rows: list[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    if not rows:
        return []
    plot_paths = []
    by_group = summarize(rows, "group")
    plot_paths.append(bar_plot(
        by_group,
        x_key="group",
        y_key="success_rate",
        title="Success Rate by Group",
        ylabel="success rate",
        path=output_dir / "success_rate_by_group.png",
        ylim=(0, 1.05),
    ))
    plot_paths.append(bar_plot(
        by_group,
        x_key="group",
        y_key="avg_steps",
        title="Average Steps by Group",
        ylabel="avg steps",
        path=output_dir / "avg_steps_by_group.png",
    ))
    plot_paths.append(hist_plot(
        [int(row.get("total_steps", 0)) for row in rows],
        title="Episode Step Distribution",
        xlabel="total steps",
        path=output_dir / "step_histogram.png",
    ))
    plot_paths.append(bar_plot(
        [
            {"metric": "blocked", "count": sum(int(row.get("blocked_action_count", 0)) for row in rows)},
            {"metric": "fallback", "count": sum(int(row.get("fallback_action_count", 0)) for row in rows)},
            {"metric": "planner", "count": sum(int(row.get("planner_action_count", 0)) for row in rows)},
        ],
        x_key="metric",
        y_key="count",
        title="Control Events",
        ylabel="count",
        path=output_dir / "control_event_counts.png",
    ))
    return plot_paths


def bar_plot(
    rows: list[Mapping[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    ylabel: str,
    path: Path,
    ylim: tuple[float, float] | None = None,
) -> Path:
    labels = [str(row.get(x_key, "")) or "(blank)" for row in rows]
    values = [float(row.get(y_key, 0.0)) for row in rows]
    fig_width = max(8, min(18, len(labels) * 0.6))
    plt.figure(figsize=(fig_width, 4.5))
    plt.bar(labels, values, color="#2f6f73")
    plt.title(title)
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(*ylim)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def hist_plot(values: list[int], title: str, xlabel: str, path: Path) -> Path:
    plt.figure(figsize=(8, 4.5))
    plt.hist(values, bins=min(30, max(5, len(set(values)))), color="#8a5a44", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("runs")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_report(
    path: Path,
    rows: list[Mapping[str, Any]],
    plot_paths: list[Path],
    result_root: Path,
) -> Path:
    runs = len(rows)
    successes = sum(1 for row in rows if row.get("success"))
    steps = sum(int(row.get("total_steps", 0)) for row in rows)
    blocked = sum(int(row.get("blocked_action_count", 0)) for row in rows)
    fallback = sum(int(row.get("fallback_action_count", 0)) for row in rows)
    planner = sum(int(row.get("planner_action_count", 0)) for row in rows)
    lines = [
        "# NS-FSM Rollout Visual Report",
        "",
        f"- Result root: `{result_root}`",
        f"- Runs: `{runs}`",
        f"- Successes: `{successes}`",
        f"- Success rate: `{successes / runs if runs else 0:.3f}`",
        f"- Average steps: `{steps / runs if runs else 0:.3f}`",
        f"- Blocked actions: `{blocked}`",
        f"- Fallback actions: `{fallback}`",
        f"- Planner decisions: `{planner}`",
        "",
        "## Figures",
        "",
    ]
    for plot_path in plot_paths:
        lines.append(f"- [{plot_path.name}]({plot_path.name})")
    lines.append("")
    lines.append("## CSV Outputs")
    lines.append("")
    lines.append("- `rollout_runs.csv`")
    lines.append("- `summary_by_task.csv`")
    lines.append("- `summary_by_group.csv`")
    lines.append("- `summary_by_termination.csv`")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def log_wandb(
    args: argparse.Namespace,
    rows: list[Mapping[str, Any]],
    plot_paths: list[Path],
    report_path: Path,
) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed. Install it or rerun without --wandb.") from exc
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or f"{args.tag}-analysis",
        job_type="analysis",
        config={"tag": args.tag, "rows": len(rows)},
    )
    for plot_path in plot_paths:
        run.log({f"figure/{plot_path.stem}": wandb.Image(str(plot_path))})
    artifact = wandb.Artifact(f"{args.tag}-analysis", type="analysis")
    artifact.add_file(str(report_path))
    for csv_name in ["rollout_runs.csv", "summary_by_task.csv", "summary_by_group.csv", "summary_by_termination.csv"]:
        csv_path = report_path.parent / csv_name
        if csv_path.exists():
            artifact.add_file(str(csv_path))
    run.log_artifact(artifact)
    run.finish()


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    main()

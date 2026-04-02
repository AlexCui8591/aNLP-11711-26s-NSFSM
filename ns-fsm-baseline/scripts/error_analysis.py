"""
Phase 5 error analysis driver.

Consumes saved trajectory JSONs, computes the README metrics, and writes:
  - tables/*.csv, *.json
  - figures/*.png
  - tables/baseline_report.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ground_truth import GroundTruth
from metrics import (
    EpisodeRecord,
    ReflexionRunRecord,
    build_group_depths,
    compute_reflexion_fix_rate,
    select_case_studies,
    summarize_episodes,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


GOALS_PATH = ROOT / "config" / "goals_67.json"
TRAJECTORIES_DIR = ROOT / "results" / "trajectories"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
FULL_V1_REACT_INPUT = ROOT.parent / "full_v1_results" / "full_v1" / "react"
FULL_V1_REACT_OUTPUT = ROOT.parent / "full_v1_results" / "analysis" / "react_error2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 error analysis")
    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(TRAJECTORIES_DIR)],
        help="JSON files or directories to analyze",
    )
    parser.add_argument(
        "--tables-dir",
        default=None,
        help="Directory for CSV/JSON/Markdown outputs",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Directory for plot outputs",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=5,
        help="Maximum number of representative failure case studies",
    )
    return parser.parse_args()


def resolve_output_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    input_paths = [Path(p).resolve() for p in args.input]
    full_v1_react = FULL_V1_REACT_INPUT.resolve()

    if args.tables_dir is None and args.figures_dir is None:
        if len(input_paths) == 1 and input_paths[0] == full_v1_react:
            return FULL_V1_REACT_OUTPUT, FULL_V1_REACT_OUTPUT
        return TABLES_DIR, FIGURES_DIR

    tables_dir = Path(args.tables_dir) if args.tables_dir else TABLES_DIR
    figures_dir = Path(args.figures_dir) if args.figures_dir else FIGURES_DIR
    return tables_dir, figures_dir


def load_goals_config() -> dict:
    with GOALS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_goal_index(goals_config: dict) -> dict[str, dict]:
    index = {}
    for group, payload in goals_config.items():
        for entry in payload.get("goals", []):
            index[entry["goal"]] = {
                "group": group,
                "goal_type": entry.get("type", ""),
                "instruction": entry.get("instruction", ""),
                "id": entry.get("id"),
            }
    return index


def discover_json_files(paths: list[str]) -> list[Path]:
    files = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".json":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
    return sorted(set(files))


def infer_group(goal: str, payload: dict, goal_index: dict[str, dict]) -> str:
    if payload.get("group"):
        return payload["group"]
    if goal in goal_index:
        return goal_index[goal]["group"]
    return "Unknown"


def build_episode_record(
    payload: dict,
    *,
    agent: str,
    goal: str,
    group: str,
    source: str,
    attempt_index: int | None = None,
    inherited_metadata: dict | None = None,
) -> EpisodeRecord:
    metadata = dict(inherited_metadata or {})
    for key, value in payload.items():
        if key in {"goal", "success", "total_steps", "termination", "trajectory"}:
            continue
        metadata.setdefault(key, value)
    return EpisodeRecord(
        agent=agent,
        goal=goal,
        group=group,
        success=bool(payload.get("success", False)),
        total_steps=int(payload.get("total_steps", len(payload.get("trajectory", [])))),
        termination=payload.get("termination", "unknown"),
        trajectory=list(payload.get("trajectory", [])),
        source=source,
        attempt_index=attempt_index,
        metadata=metadata,
    )


def build_reflexion_run(
    payload: dict,
    *,
    goal: str,
    group: str,
    source: str,
    inherited_metadata: dict | None = None,
) -> ReflexionRunRecord | None:
    attempts_payload = payload.get("attempts", [])
    attempts = []
    for attempt_payload in attempts_payload:
        result_payload = attempt_payload.get("result", {})
        if not result_payload:
            continue
        attempt_idx = attempt_payload.get("attempt") or len(attempts) + 1
        metadata = dict(inherited_metadata or {})
        metadata["reflection"] = attempt_payload.get("reflection", "")
        attempts.append(
            build_episode_record(
                result_payload,
                agent="reflexion",
                goal=goal,
                group=group,
                source=source,
                attempt_index=attempt_idx,
                inherited_metadata=metadata,
            )
        )

    if not attempts:
        return None

    metadata = dict(inherited_metadata or {})
    for key, value in payload.items():
        if key in {
            "goal",
            "success",
            "winning_attempt",
            "total_attempts",
            "attempts",
        }:
            continue
        metadata.setdefault(key, value)

    return ReflexionRunRecord(
        goal=goal,
        group=group,
        success=bool(payload.get("success", False)),
        winning_attempt=payload.get("winning_attempt"),
        total_attempts=int(payload.get("total_attempts", len(attempts))),
        attempts=attempts,
        source=source,
        metadata=metadata,
    )


def extract_records_from_payload(
    payload: dict,
    *,
    source: str,
    goal_index: dict[str, dict],
    inherited_metadata: dict | None = None,
) -> tuple[list[EpisodeRecord], list[ReflexionRunRecord]]:
    episodes: list[EpisodeRecord] = []
    reflexion_runs: list[ReflexionRunRecord] = []

    if "entries" in payload:
        inherited = dict(inherited_metadata or {})
        for key in ("timestamp", "skip_reflexion", "react_success_rate"):
            if key in payload:
                inherited[key] = payload[key]
        for entry in payload.get("entries", []):
            entry_episodes, entry_runs = extract_records_from_payload(
                entry,
                source=source,
                goal_index=goal_index,
                inherited_metadata=inherited,
            )
            episodes.extend(entry_episodes)
            reflexion_runs.extend(entry_runs)
        return episodes, reflexion_runs

    if "results" in payload:
        inherited = dict(inherited_metadata or {})
        for key in ("timestamp", "mode"):
            if key in payload:
                inherited[key] = payload[key]
        for result in payload.get("results", []):
            result_episodes, result_runs = extract_records_from_payload(
                result,
                source=source,
                goal_index=goal_index,
                inherited_metadata=inherited,
            )
            episodes.extend(result_episodes)
            reflexion_runs.extend(result_runs)
        return episodes, reflexion_runs

    if "react" in payload:
        goal = payload.get("goal") or payload["react"].get("goal", "")
        group = infer_group(goal, payload, goal_index)
        metadata = dict(inherited_metadata or {})
        for key in ("tier", "chain", "optimal_steps"):
            if key in payload:
                metadata[key] = payload[key]
        episodes.append(
            build_episode_record(
                payload["react"],
                agent="react",
                goal=goal,
                group=group,
                source=source,
                inherited_metadata=metadata,
            )
        )
        if "reflexion" in payload:
            run = build_reflexion_run(
                payload["reflexion"],
                goal=goal,
                group=group,
                source=source,
                inherited_metadata=metadata,
            )
            if run is not None:
                reflexion_runs.append(run)
        return episodes, reflexion_runs

    if "attempts" in payload and "trajectory" not in payload:
        goal = payload.get("goal", "")
        group = infer_group(goal, payload, goal_index)
        run = build_reflexion_run(
            payload,
            goal=goal,
            group=group,
            source=source,
            inherited_metadata=inherited_metadata,
        )
        if run is not None:
            reflexion_runs.append(run)
        return episodes, reflexion_runs

    if "trajectory" in payload and payload.get("goal"):
        goal = payload["goal"]
        group = infer_group(goal, payload, goal_index)
        agent = payload.get("agent")
        if not agent:
            source_name = Path(source).name.lower()
            agent = "reflexion" if "reflexion" in source_name else "react"
        episodes.append(
            build_episode_record(
                payload,
                agent=agent,
                goal=goal,
                group=group,
                source=source,
                inherited_metadata=inherited_metadata,
            )
        )
    return episodes, reflexion_runs


def load_analysis_records(paths: list[str], goal_index: dict[str, dict]):
    react_episodes = []
    reflexion_runs = []
    for path in discover_json_files(paths):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        episodes, runs = extract_records_from_payload(
            payload,
            source=str(path),
            goal_index=goal_index,
        )
        react_episodes.extend([episode for episode in episodes if episode.agent == "react"])
        reflexion_runs.extend(runs)

    react_episodes = dedupe_episodes(react_episodes)
    reflexion_runs = dedupe_reflexion_runs(reflexion_runs)
    reflexion_final_episodes = [
        run.attempts[-1] for run in reflexion_runs if run.attempts
    ]
    return react_episodes, reflexion_runs, reflexion_final_episodes


def dedupe_episodes(episodes: list[EpisodeRecord]) -> list[EpisodeRecord]:
    seen = set()
    deduped = []
    for episode in episodes:
        fingerprint = episode_fingerprint(episode)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(episode)
    return deduped


def dedupe_reflexion_runs(runs: list[ReflexionRunRecord]) -> list[ReflexionRunRecord]:
    seen = set()
    deduped = []
    for run in runs:
        fingerprint = reflexion_fingerprint(run)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(run)
    return deduped


def episode_fingerprint(episode: EpisodeRecord) -> str:
    run_id = episode.metadata.get("run_id")
    if run_id is not None:
        payload = {
            "agent": episode.agent,
            "goal": episode.goal,
            "group": episode.group,
            "run_id": run_id,
            "attempt_index": episode.attempt_index,
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    payload = {
        "agent": episode.agent,
        "goal": episode.goal,
        "group": episode.group,
        "success": episode.success,
        "total_steps": episode.total_steps,
        "termination": episode.termination,
        "attempt_index": episode.attempt_index,
        "trajectory": episode.trajectory,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def reflexion_fingerprint(run: ReflexionRunRecord) -> str:
    run_id = run.metadata.get("run_id")
    if run_id is not None:
        payload = {
            "goal": run.goal,
            "group": run.group,
            "run_id": run_id,
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    payload = {
        "goal": run.goal,
        "group": run.group,
        "success": run.success,
        "winning_attempt": run.winning_attempt,
        "total_attempts": run.total_attempts,
        "attempts": [episode_fingerprint(attempt) for attempt in run.attempts],
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_episode_lookup(episodes: list[EpisodeRecord]) -> dict[tuple, EpisodeRecord]:
    lookup = {}
    for episode in episodes:
        lookup[
            (
                episode.agent,
                episode.goal,
                episode.group,
                episode.source,
                episode.attempt_index,
                episode.total_steps,
            )
        ] = episode
    return lookup


def write_episode_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "agent",
        "goal",
        "group",
        "success",
        "total_steps",
        "termination",
        "attempt_index",
        "optimal_depth",
        "invalid_action_steps",
        "invalid_action_rate",
        "repeated_action_transitions",
        "repeated_action_rate",
        "failed_steps",
        "successful_steps",
        "dead_loop",
        "cascade_failure",
        "plan_knowledge_error_steps",
        "sequencing_error_steps",
        "dominant_error_type",
        "first_failure_step",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_group_success_table(path: Path, agent_summaries: dict[str, dict]) -> None:
    groups = sorted({
        group
        for summary in agent_summaries.values()
        for group in summary.get("by_group", {})
    })
    fieldnames = [
        "group",
        "avg_dependency_depth",
        "react_success_rate",
        "react_average_steps_success",
        "react_invalid_action_rate",
        "react_dead_loop_rate",
        "react_plan_knowledge_error_rate",
        "react_sequencing_error_rate",
        "reflexion_success_rate",
        "reflexion_average_steps_success",
        "reflexion_invalid_action_rate",
        "reflexion_dead_loop_rate",
        "reflexion_plan_knowledge_error_rate",
        "reflexion_sequencing_error_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            react_group = agent_summaries.get("react", {}).get("by_group", {}).get(group, {})
            reflexion_group = agent_summaries.get("reflexion", {}).get("by_group", {}).get(group, {})
            writer.writerow({
                "group": group,
                "avg_dependency_depth": react_group.get(
                    "avg_dependency_depth",
                    reflexion_group.get("avg_dependency_depth", 0.0),
                ),
                "react_success_rate": react_group.get("success_rate", 0.0),
                "react_average_steps_success": react_group.get("average_steps_success", 0.0),
                "react_invalid_action_rate": react_group.get("invalid_action_rate", 0.0),
                "react_dead_loop_rate": react_group.get("dead_loop_rate", 0.0),
                "react_plan_knowledge_error_rate": react_group.get(
                    "plan_knowledge_error_rate", 0.0
                ),
                "react_sequencing_error_rate": react_group.get(
                    "sequencing_error_rate", 0.0
                ),
                "reflexion_success_rate": reflexion_group.get("success_rate", 0.0),
                "reflexion_average_steps_success": reflexion_group.get(
                    "average_steps_success", 0.0
                ),
                "reflexion_invalid_action_rate": reflexion_group.get(
                    "invalid_action_rate", 0.0
                ),
                "reflexion_dead_loop_rate": reflexion_group.get("dead_loop_rate", 0.0),
                "reflexion_plan_knowledge_error_rate": reflexion_group.get(
                    "plan_knowledge_error_rate", 0.0
                ),
                "reflexion_sequencing_error_rate": reflexion_group.get(
                    "sequencing_error_rate", 0.0
                ),
            })


def plot_success_rate_by_group(path: Path, agent_summaries: dict[str, dict]) -> None:
    if plt is None:
        return
    groups = sorted({
        group
        for summary in agent_summaries.values()
        for group in summary.get("by_group", {})
    })
    if not groups:
        return

    react_values = [
        agent_summaries.get("react", {}).get("by_group", {}).get(group, {}).get("success_rate", 0.0)
        for group in groups
    ]
    reflexion_values = [
        agent_summaries.get("reflexion", {}).get("by_group", {}).get(group, {}).get("success_rate", 0.0)
        for group in groups
    ]

    positions = list(range(len(groups)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([p - width / 2 for p in positions], react_values, width=width, label="ReAct")
    if any(value > 0 for value in reflexion_values):
        ax.bar([p + width / 2 for p in positions], reflexion_values, width=width, label="Reflexion")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Goal Group")
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_performance_decay(path: Path, agent_summaries: dict[str, dict]) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False

    for agent_name, summary in agent_summaries.items():
        points = summary.get("performance_decay", {}).get("points", [])
        if not points:
            continue
        xs = [point["avg_dependency_depth"] for point in points]
        ys = [point["success_rate"] for point in points]
        labels = [point["group"] for point in points]
        ax.plot(xs, ys, marker="o", label=agent_name.title())
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 6), ha="center")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Average dependency depth")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Performance Decay vs. Plan Length")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_case_studies(
    case_rows: list[dict],
    episode_lookup: dict[tuple, EpisodeRecord],
) -> str:
    if not case_rows:
        return "No failed episodes were available for case studies.\n"

    lines = []
    for idx, row in enumerate(case_rows, start=1):
        key = (
            row["agent"],
            row["goal"],
            row["group"],
            row["source"],
            row["attempt_index"],
            row["total_steps"],
        )
        episode = episode_lookup.get(key)
        failing_steps = []
        if episode is not None:
            failing_steps = [step for step in episode.trajectory if not step.get("success", False)]

        lines.append(f"## Case {idx}: {row['agent'].title()} on `{row['goal']}`")
        lines.append(f"- Group: `{row['group']}`")
        lines.append(f"- Termination: `{row['termination']}`")
        lines.append(f"- Steps: `{row['total_steps']}`")
        lines.append(f"- Dominant error: `{row['dominant_error_type']}`")
        if row.get("first_failure_step") is not None:
            lines.append(f"- First failure step: `{row['first_failure_step']}`")
        if row.get("first_failure_message"):
            lines.append(f"- First failure: `{row['first_failure_message']}`")
        if failing_steps:
            lines.append("- Failure trace:")
            for step in failing_steps[:3]:
                lines.append(
                    f"  - Step {step['step']}: `{step['action']}` -> {step.get('message', '')}"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_report(
    *,
    input_paths: list[str],
    agent_summaries: dict[str, dict],
    reflexion_fix_summary: dict | None,
    case_markdown: str,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Baseline Reproduction",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Inputs: `{', '.join(input_paths)}`",
        f"- ReAct episodes analyzed: `{agent_summaries.get('react', {}).get('n_episodes', 0)}`",
        f"- Reflexion runs analyzed: `{reflexion_fix_summary.get('n_runs', 0) if reflexion_fix_summary else 0}`",
        "",
        "# Error Analysis",
        "",
    ]

    for agent_name, summary in agent_summaries.items():
        overall = summary.get("overall", {})
        lines.extend([
            f"## {agent_name.title()}",
            "",
            f"- Success rate: `{overall.get('n_success', 0)}/{overall.get('n_episodes', 0)}` ({overall.get('success_rate', 0.0):.1%})",
            f"- Average steps on success: `{overall.get('average_steps_success', 0.0):.1f}`",
            f"- Invalid action rate: `{overall.get('invalid_action_rate', 0.0):.1%}`",
            f"- Repeated action rate: `{overall.get('repeated_action_rate', 0.0):.1%}`",
            f"- Dead loop rate: `{overall.get('dead_loop_rate', 0.0):.1%}`",
            f"- Cascade failure rate: `{overall.get('cascade_failure_rate', 0.0):.1%}`",
            f"- Plan knowledge error rate: `{overall.get('plan_knowledge_error_rate', 0.0):.1%}`",
            f"- Sequencing error rate: `{overall.get('sequencing_error_rate', 0.0):.1%}`",
            "",
        ])
        decay = summary.get("performance_decay", {})
        if decay.get("slope") is not None:
            lines.append(
                f"- Performance decay slope: `{decay['slope']:.4f}` success-rate points per dependency-depth unit"
            )
            lines.append("")

    if reflexion_fix_summary:
        lines.extend([
            "## Reflexion Fix Rate",
            "",
            f"- Eligible runs: `{reflexion_fix_summary['eligible_runs']}`",
            f"- Fixed runs: `{reflexion_fix_summary['fixed_runs']}`",
            f"- Fix rate: `{reflexion_fix_summary['fix_rate']:.1%}`",
            "",
        ])

    lines.extend([
        "# Reflection",
        "",
        "- The best evidence for structural weakness is the combination of declining success rate with rising dependency depth.",
        "- Plan-knowledge errors indicate recipe confusion or unsupported actions.",
        "- Sequencing errors indicate the agent knows the right recipe family but attempts late-stage actions before satisfying prerequisites.",
        "- Dead loops and cascade failures isolate where plain prompt-based control keeps wasting budget after the first mistake.",
        "",
        "# Representative Failures",
        "",
        case_markdown.rstrip(),
        "",
    ])
    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()

    tables_dir, figures_dir = resolve_output_dirs(args)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    goals_config = load_goals_config()
    goal_index = build_goal_index(goals_config)
    ground_truth = GroundTruth()
    group_depths = build_group_depths(goals_config, ground_truth)

    react_episodes, reflexion_runs, reflexion_final_episodes = load_analysis_records(
        args.input,
        goal_index,
    )

    if not react_episodes and not reflexion_runs:
        print("No trajectory artifacts found to analyze.")
        return 1

    agent_summaries = {}
    all_episodes = []

    if react_episodes:
        react_summary = summarize_episodes(react_episodes, ground_truth, group_depths)
        agent_summaries["react"] = react_summary
        all_episodes.extend(react_episodes)

    reflexion_fix_summary = None
    if reflexion_final_episodes:
        reflexion_summary = summarize_episodes(
            reflexion_final_episodes,
            ground_truth,
            group_depths,
        )
        agent_summaries["reflexion"] = reflexion_summary
        all_episodes.extend(reflexion_final_episodes)
        reflexion_fix_summary = compute_reflexion_fix_rate(reflexion_runs)

    episode_lookup = build_episode_lookup(all_episodes)
    combined_rows = []
    for summary in agent_summaries.values():
        combined_rows.extend(summary.get("episode_metrics", []))
    case_rows = select_case_studies(combined_rows, limit=args.case_limit)
    case_markdown = render_case_studies(case_rows, episode_lookup)

    aggregate_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": args.input,
        "group_depths": group_depths,
        "agents": agent_summaries,
        "reflexion_fix_rate": reflexion_fix_summary,
        "case_studies": case_rows,
    }

    write_json(tables_dir / "aggregate_metrics.json", aggregate_payload)
    write_episode_metrics_csv(tables_dir / "episode_metrics.csv", combined_rows)
    write_group_success_table(tables_dir / "group_success_rate.csv", agent_summaries)

    case_study_path = tables_dir / "case_studies.md"
    case_study_path.write_text(case_markdown, encoding="utf-8")

    report_text = render_report(
        input_paths=args.input,
        agent_summaries=agent_summaries,
        reflexion_fix_summary=reflexion_fix_summary,
        case_markdown=case_markdown,
    )
    report_path = tables_dir / "baseline_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    plot_success_rate_by_group(figures_dir / "success_rate_by_group.png", agent_summaries)
    plot_performance_decay(figures_dir / "performance_decay.png", agent_summaries)

    print("Saved analysis artifacts:")
    print(f"  - {tables_dir / 'aggregate_metrics.json'}")
    print(f"  - {tables_dir / 'episode_metrics.csv'}")
    print(f"  - {tables_dir / 'group_success_rate.csv'}")
    print(f"  - {case_study_path}")
    print(f"  - {report_path}")
    if plt is None:
        print("  - matplotlib not installed; figure generation skipped")
    else:
        print(f"  - {figures_dir / 'success_rate_by_group.png'}")
        print(f"  - {figures_dir / 'performance_decay.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

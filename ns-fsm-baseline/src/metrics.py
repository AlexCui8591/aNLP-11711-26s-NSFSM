"""
Metric helpers for Phase 5 error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable, Optional


_UNKNOWN_ACTION_MARKERS = (
    "not in the action library",
    "unknown action",
    "does not exist",
    "not exist",
)


@dataclass
class EpisodeRecord:
    agent: str
    goal: str
    group: str
    success: bool
    total_steps: int
    termination: str
    trajectory: list[dict]
    source: str = ""
    attempt_index: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflexionRunRecord:
    goal: str
    group: str
    success: bool
    winning_attempt: Optional[int]
    total_attempts: int
    attempts: list[EpisodeRecord]
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def split_action(action: str) -> tuple[str, str]:
    if "_" not in action:
        return action, ""
    return action.split("_", 1)


def is_unknown_action_message(message: str, action: str = "") -> bool:
    lowered = (message or "").lower()
    if action == "__invalid__":
        return True
    return any(marker in lowered for marker in _UNKNOWN_ACTION_MARKERS)


def has_missing_tool(message: str) -> bool:
    return "missing tool:" in (message or "").lower()


def has_missing_material(message: str) -> bool:
    return "missing materials:" in (message or "").lower()


def build_group_depths(goals_config: dict, ground_truth) -> dict[str, float]:
    group_depths = {}
    for group, payload in goals_config.items():
        depths = []
        for goal_entry in payload.get("goals", []):
            depth = ground_truth.get_dependency_depth(goal_entry["goal"])
            if depth >= 0:
                depths.append(depth)
        group_depths[group] = mean(depths) if depths else 0.0
    return group_depths


def classify_episode_errors(
    trajectory: list[dict],
    goal: str,
    ground_truth,
) -> dict[str, Any]:
    required_items = set(ground_truth.get_all_required_items(goal))
    required_items.add(goal)
    required_actions = set(ground_truth.get_required_actions(goal))
    optimal_depth = ground_truth.get_dependency_depth(goal)

    invalid_action_steps = 0
    repeated_action_transitions = 0
    max_repeat_streak = 1 if trajectory else 0
    current_repeat_streak = 1 if trajectory else 0
    repeated_failed_loop = False
    failed_steps = 0
    successful_steps = 0
    plan_knowledge_error_steps = 0
    sequencing_error_steps = 0
    first_failure_step = None
    first_failure_message = ""
    post_failure_successes = 0
    first_post_failure_success = None

    prev_action = None
    prev_failed_action = None
    failed_action_streak = 0

    for step in trajectory:
        action = step.get("action", "")
        message = step.get("message", "")
        success = bool(step.get("success", False))
        act_type, item = split_action(action)
        unknown_action = is_unknown_action_message(message, action)
        missing_tool = has_missing_tool(message)
        missing_material = has_missing_material(message)
        is_required_item = item in required_items if item else False
        is_required_action = action in required_actions

        if prev_action is not None and action == prev_action:
            repeated_action_transitions += 1
            current_repeat_streak += 1
            max_repeat_streak = max(max_repeat_streak, current_repeat_streak)
        else:
            current_repeat_streak = 1
        prev_action = action

        if success:
            successful_steps += 1
            prev_failed_action = None
            failed_action_streak = 0
            if first_failure_step is not None:
                post_failure_successes += 1
                if first_post_failure_success is None:
                    first_post_failure_success = step.get("step")
            continue

        failed_steps += 1
        if first_failure_step is None:
            first_failure_step = step.get("step")
            first_failure_message = message

        if unknown_action:
            invalid_action_steps += 1
            plan_knowledge_error_steps += 1
        elif not is_required_item and not is_required_action:
            plan_knowledge_error_steps += 1
        elif missing_tool or missing_material:
            sequencing_error_steps += 1

        if action == prev_failed_action:
            failed_action_streak += 1
        else:
            prev_failed_action = action
            failed_action_streak = 1
        if failed_action_streak >= 5:
            repeated_failed_loop = True

    cascade_failure = bool(
        failed_steps > 0 and first_failure_step is not None and post_failure_successes == 0
    )
    dead_loop = repeated_failed_loop or max_repeat_streak >= 5

    if dead_loop:
        dominant_error_type = "dead_loop"
    elif plan_knowledge_error_steps > sequencing_error_steps and plan_knowledge_error_steps > 0:
        dominant_error_type = "plan_knowledge_error"
    elif sequencing_error_steps > 0:
        dominant_error_type = "sequencing_error"
    elif invalid_action_steps > 0:
        dominant_error_type = "invalid_action"
    elif cascade_failure:
        dominant_error_type = "cascade_failure"
    elif failed_steps == 0:
        dominant_error_type = "none"
    else:
        dominant_error_type = "other_failure"

    return {
        "optimal_depth": optimal_depth,
        "required_action_count": len(required_actions),
        "required_item_count": len(required_items),
        "invalid_action_steps": invalid_action_steps,
        "failed_steps": failed_steps,
        "successful_steps": successful_steps,
        "repeated_action_transitions": repeated_action_transitions,
        "max_repeat_streak": max_repeat_streak,
        "repeated_failed_loop": repeated_failed_loop,
        "dead_loop": dead_loop,
        "cascade_failure": cascade_failure,
        "plan_knowledge_error_steps": plan_knowledge_error_steps,
        "sequencing_error_steps": sequencing_error_steps,
        "first_failure_step": first_failure_step,
        "first_failure_message": first_failure_message,
        "first_post_failure_success": first_post_failure_success,
        "dominant_error_type": dominant_error_type,
        "invalid_action_rate": safe_ratio(invalid_action_steps, len(trajectory)),
        "repeated_action_rate": safe_ratio(
            repeated_action_transitions, max(len(trajectory) - 1, 0)
        ),
    }


def summarize_episodes(
    episodes: Iterable[EpisodeRecord],
    ground_truth,
    group_depths: dict[str, float],
) -> dict[str, Any]:
    materialized = list(episodes)
    episode_metrics = []
    for episode in materialized:
        episode_metric = {
            "agent": episode.agent,
            "goal": episode.goal,
            "group": episode.group,
            "success": episode.success,
            "total_steps": episode.total_steps,
            "termination": episode.termination,
            "source": episode.source,
            "attempt_index": episode.attempt_index,
            **classify_episode_errors(episode.trajectory, episode.goal, ground_truth),
        }
        episode_metrics.append(episode_metric)

    overall = _aggregate_episode_rows(episode_metrics)

    by_group = {}
    groups = sorted({row["group"] for row in episode_metrics})
    for group in groups:
        rows = [row for row in episode_metrics if row["group"] == group]
        summary = _aggregate_episode_rows(rows)
        summary["avg_dependency_depth"] = group_depths.get(group, 0.0)
        by_group[group] = summary

    performance_decay = compute_performance_decay(by_group)

    return {
        "n_episodes": len(episode_metrics),
        "overall": overall,
        "by_group": by_group,
        "performance_decay": performance_decay,
        "episode_metrics": episode_metrics,
    }


def _aggregate_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_episodes": 0,
            "n_success": 0,
            "success_rate": 0.0,
            "average_steps_success": 0.0,
            "average_steps_all": 0.0,
            "invalid_action_rate": 0.0,
            "repeated_action_rate": 0.0,
            "dead_loop_rate": 0.0,
            "cascade_failure_rate": 0.0,
            "plan_knowledge_error_rate": 0.0,
            "sequencing_error_rate": 0.0,
            "average_optimal_depth": 0.0,
            "termination_counts": {},
        }

    successful_rows = [row for row in rows if row["success"]]
    failed_rows = [row for row in rows if not row["success"]]
    total_steps = sum(row["total_steps"] for row in rows)
    total_transitions = sum(max(row["total_steps"] - 1, 0) for row in rows)
    total_invalid_steps = sum(row["invalid_action_steps"] for row in rows)
    total_repeated_transitions = sum(row["repeated_action_transitions"] for row in rows)

    termination_counts = {}
    for row in rows:
        termination = row["termination"]
        termination_counts[termination] = termination_counts.get(termination, 0) + 1

    return {
        "n_episodes": len(rows),
        "n_success": len(successful_rows),
        "success_rate": safe_ratio(len(successful_rows), len(rows)),
        "average_steps_success": mean(row["total_steps"] for row in successful_rows)
        if successful_rows else 0.0,
        "average_steps_all": mean(row["total_steps"] for row in rows),
        "invalid_action_rate": safe_ratio(total_invalid_steps, total_steps),
        "repeated_action_rate": safe_ratio(total_repeated_transitions, total_transitions),
        "dead_loop_rate": safe_ratio(sum(1 for row in rows if row["dead_loop"]), len(rows)),
        "cascade_failure_rate": safe_ratio(
            sum(1 for row in failed_rows if row["cascade_failure"]),
            len(failed_rows),
        ),
        "plan_knowledge_error_rate": safe_ratio(
            sum(1 for row in failed_rows if row["plan_knowledge_error_steps"] > 0),
            len(failed_rows),
        ),
        "sequencing_error_rate": safe_ratio(
            sum(1 for row in failed_rows if row["sequencing_error_steps"] > 0),
            len(failed_rows),
        ),
        "average_optimal_depth": mean(
            row["optimal_depth"] for row in rows if row["optimal_depth"] >= 0
        ) if any(row["optimal_depth"] >= 0 for row in rows) else 0.0,
        "termination_counts": termination_counts,
    }


def compute_performance_decay(by_group: dict[str, dict[str, Any]]) -> dict[str, Any]:
    points = []
    for group, payload in by_group.items():
        points.append({
            "group": group,
            "avg_dependency_depth": payload.get("avg_dependency_depth", 0.0),
            "success_rate": payload.get("success_rate", 0.0),
        })
    points.sort(key=lambda point: point["avg_dependency_depth"])

    xs = [point["avg_dependency_depth"] for point in points]
    ys = [point["success_rate"] for point in points]
    fit = linear_regression(xs, ys)
    return {
        "points": points,
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "r_squared": fit["r_squared"],
    }


def linear_regression(xs: list[float], ys: list[float]) -> dict[str, Optional[float]]:
    if len(xs) < 2 or len(xs) != len(ys):
        return {"slope": None, "intercept": None, "r_squared": None}

    mean_x = mean(xs)
    mean_y = mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return {"slope": 0.0, "intercept": mean_y, "r_squared": 0.0}

    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}


def compute_reflexion_fix_rate(runs: Iterable[ReflexionRunRecord]) -> dict[str, Any]:
    materialized = list(runs)
    eligible = [
        run for run in materialized
        if run.attempts and not run.attempts[0].success
    ]
    fixed = [run for run in eligible if run.success]

    by_group = {}
    for group in sorted({run.group for run in materialized}):
        group_eligible = [
            run for run in materialized
            if run.group == group and run.attempts and not run.attempts[0].success
        ]
        group_fixed = [run for run in group_eligible if run.success]
        by_group[group] = {
            "eligible_runs": len(group_eligible),
            "fixed_runs": len(group_fixed),
            "fix_rate": safe_ratio(len(group_fixed), len(group_eligible)),
        }

    return {
        "n_runs": len(materialized),
        "eligible_runs": len(eligible),
        "fixed_runs": len(fixed),
        "fix_rate": safe_ratio(len(fixed), len(eligible)),
        "by_group": by_group,
    }


def select_case_studies(
    episode_metrics: Iterable[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    failed_rows = [row for row in episode_metrics if not row["success"]]
    if not failed_rows:
        return []

    def sort_key(row: dict[str, Any]) -> tuple:
        return (
            row.get("optimal_depth", -1),
            row.get("total_steps", 0),
            row.get("failed_steps", 0),
        )

    failed_rows.sort(key=sort_key, reverse=True)

    chosen = []
    seen_error_types = set()
    for row in failed_rows:
        error_type = row.get("dominant_error_type", "other_failure")
        if error_type not in seen_error_types:
            chosen.append(row)
            seen_error_types.add(error_type)
        if len(chosen) >= limit:
            return chosen

    for row in failed_rows:
        if row in chosen:
            continue
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen

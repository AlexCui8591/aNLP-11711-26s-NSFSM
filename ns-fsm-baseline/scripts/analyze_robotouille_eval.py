#!/usr/bin/env python3
"""Analyze Robotouille eval outputs and generate figure-ready metrics.

The script accepts multiple runs with this format:

    --run METHOD:SUITE:PATH

Examples:

    python scripts/analyze_robotouille_eval.py \
      --run ReAct:sync:/path/to/react_sync/results.json \
      --run ReAct:async:/path/to/react_async/results.json \
      --run NS-FSM:async:results/full/robotouille_async_qwen14b/robotouille/nsfsm \
      --out-dir results/analysis/robotouille_eval

Supported inputs:
  - Official Robotouille results.json files with entries like
    {"asynchronous/5_potato_soup_42": {"done": false, "steps": 29, "max_steps": 28.5}}.
  - NS-FSM per-run JSON files produced by scripts/run_nsfsm_experiment.py.
  - NS-FSM summary JSON files with a top-level "results" list.

Outputs:
  - suite_metrics.csv/json: SR, Steps/Optimal, Budget-hit rate, Tokens(M).
  - task_difficulty.csv: per-task SR against optimal steps.
  - failure_attribution.csv: budget-hit failure buckets.
  - case_studies.md: representative failed trajectories.
  - figures/*.png: suite-level bars, task difficulty curve, attribution bars,
    case-study flows when trajectory actions are available, and a dashboard.

Notes:
  - Tokens are taken from JSON usage fields when present, or from log files that
    contain "Prompt Tokens:" / "Completion Tokens:" lines. If token usage is not
    present, the token metric is left empty in CSV and annotated as n/a in plots.
  - Failure attribution is most reliable with --failure-labels. Without labels,
    the script uses conservative keyword heuristics and marks uncertain cases as
    step inflation / inefficient execution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import statistics
import textwrap
from typing import Any, Iterable, Mapping


DEFAULT_OPTIMAL_STEPS: dict[str, dict[str, int]] = {
    "sync": {
        "synchronous/0_cheese_sandwich": 10,
        "synchronous/1_lettuce_sandwich": 14,
        "synchronous/2_lettuce_tomato_sandwich": 24,
        "synchronous/3_burger": 10,
        "synchronous/4_cheeseburger": 15,
        "synchronous/5_double_cheeseburger": 23,
        "synchronous/6_lettuce_tomato_cheeseburger": 36,
        "synchronous/7_two_lettuce_chicken_sandwich": 44,
        "synchronous/8_two_lettuce_tomato_burger": 63,
        "synchronous/9_onion_cheese_burger_and_lettuce_tomato_chicken_sandwich": 57,
    },
    "async": {
        "asynchronous/0_cheese_chicken_sandwich": 21,
        "asynchronous/1_lettuce_chicken_sandwich": 27,
        "asynchronous/2_lettuce_tomato_fried_chicken_sandwich": 37,
        "asynchronous/3_tomato_burger_and_fries": 42,
        "asynchronous/4_onion_cheese_burger_and_fried_onion": 46,
        "asynchronous/5_potato_soup": 19,
        "asynchronous/6_onion_soup": 42,
        "asynchronous/7_tomato_soup_and_lettuce_chicken_sandwich": 46,
        "asynchronous/8_onion_tomato_soup_and_two_chicken_sandwich": 68,
        "asynchronous/9_onion_potato_soup_and_fried_onion_ring_lettuce_burger_and_onion_cheese_sandwich": 82,
    },
}

FAILURE_CATEGORIES = [
    "Step inflation / inefficient execution",
    "Action-vocabulary mismatch",
    "Spec-literal / goal grounding error",
    "Other / unknown",
]

METHOD_COLORS = {
    "ReAct": "#8c8c8c",
    "NS-FSM": "#1f5cff",
    "Reflexion": "#d65f5f",
    "IO": "#4c9f70",
    "IO-CoT": "#8f65d8",
}


@dataclass(frozen=True)
class RunSpec:
    method: str
    suite: str
    path: Path


@dataclass
class EpisodeRecord:
    method: str
    suite: str
    task_id: str
    task_name: str
    seed: str
    success: bool
    steps: int | None
    max_steps: float | None
    optimal_steps: int | None
    termination: str
    source_path: str
    controller_steps: int | None = None
    simulator_steps: int | None = None
    total_tokens: int | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def budget_hit(self) -> bool:
        if self.success or self.steps is None:
            return False
        if self.max_steps is None:
            return self.termination in {"max_steps", "max_steps_or_incomplete", "budget_exhausted"}
        return self.steps >= math.floor(float(self.max_steps))

    @property
    def step_ratio(self) -> float | None:
        if self.steps is None or not self.optimal_steps:
            return None
        return float(self.steps) / float(self.optimal_steps)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    fig_dir = out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    optimal_lookup = build_optimal_lookup(args)
    specs = [parse_run_spec(raw) for raw in args.run]
    if not specs:
        raise SystemExit("Provide at least one --run METHOD:SUITE:PATH input.")

    all_records: list[EpisodeRecord] = []
    token_totals_by_spec: dict[tuple[str, str, str], int] = {}
    warnings: list[str] = []
    for spec in specs:
        records = load_records(spec, optimal_lookup, warnings)
        all_records.extend(records)
        log_tokens = scan_log_tokens(spec.path)
        if log_tokens:
            token_totals_by_spec[(spec.method, spec.suite, str(spec.path))] = log_tokens
        if not records:
            warnings.append(f"No episode records found under {spec.path}")

    failure_labels = load_failure_labels(args.failure_labels)

    suite_rows = compute_suite_metrics(all_records, specs, token_totals_by_spec)
    task_rows = compute_task_difficulty(all_records)
    attribution_rows = compute_failure_attribution(all_records, failure_labels)
    case_rows = select_case_studies(all_records, failure_labels, limit=args.case_limit)

    write_csv(data_dir / "episode_records.csv", episode_rows(all_records))
    write_csv(data_dir / "suite_metrics.csv", suite_rows)
    write_json(data_dir / "suite_metrics.json", suite_rows)
    write_csv(data_dir / "task_difficulty.csv", task_rows)
    write_json(data_dir / "task_difficulty.json", task_rows)
    write_csv(data_dir / "failure_attribution.csv", attribution_rows)
    write_json(data_dir / "failure_attribution.json", attribution_rows)
    write_case_studies(data_dir / "case_studies.md", case_rows)

    generated = []
    if not args.no_plots:
        generated.extend(plot_suite_metrics(suite_rows, fig_dir))
        generated.extend(plot_task_difficulty(task_rows, fig_dir))
        generated.extend(plot_failure_attribution(attribution_rows, fig_dir))
        generated.extend(plot_case_studies(case_rows, fig_dir))
        generated.extend(plot_dashboard(suite_rows, task_rows, attribution_rows, fig_dir))

    write_json(
        data_dir / "analysis_manifest.json",
        {
            "runs": [
                {"method": spec.method, "suite": spec.suite, "path": str(spec.path)}
                for spec in specs
            ],
            "n_records": len(all_records),
            "outputs": {
                "data_dir": str(data_dir),
                "figures": [str(path) for path in generated],
            },
            "warnings": warnings,
        },
    )

    print(f"Wrote Robotouille analysis to {out_dir}")
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="METHOD:SUITE:PATH",
        help="Run input. SUITE can be sync, synchronous, async, or asynchronous.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/analysis/robotouille_eval",
        help="Directory where data and figures will be written.",
    )
    parser.add_argument(
        "--failure-labels",
        default="",
        help=(
            "Optional CSV with columns method,suite,task_id,seed,category. "
            "Use this for paper-quality failure attribution."
        ),
    )
    parser.add_argument(
        "--task-metadata",
        action="append",
        default=[],
        help=(
            "Optional JSON/YAML file containing Robotouille tasks or Hydra "
            "evaluation environment_names/optimal_steps."
        ),
    )
    parser.add_argument("--case-limit", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_run_spec(raw: str) -> RunSpec:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"--run must be METHOD:SUITE:PATH, got: {raw}")
    method, suite, path = parts
    method = method.strip()
    suite = normalize_suite(suite)
    if not method:
        raise ValueError(f"Missing method in --run {raw}")
    return RunSpec(method=method, suite=suite, path=Path(path).expanduser())


def normalize_suite(value: str) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"sync", "synchronous"}:
        return "sync"
    if lowered in {"async", "asynchronous"}:
        return "async"
    if "asynchronous" in lowered:
        return "async"
    if "synchronous" in lowered:
        return "sync"
    return lowered or "unknown"


def suite_display(suite: str) -> str:
    return {"sync": "Sync", "async": "Async"}.get(suite, suite)


def build_optimal_lookup(args: argparse.Namespace) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for suite_map in DEFAULT_OPTIMAL_STEPS.values():
        for env_name, steps in suite_map.items():
            add_task_aliases(lookup, env_name, steps)

    for raw_path in args.task_metadata:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        payload = load_structured_file(path)
        for env_name, steps in iter_task_optimal_pairs(payload):
            add_task_aliases(lookup, env_name, steps)
    return lookup


def add_task_aliases(lookup: dict[str, int], env_name: str, optimal_steps: int) -> None:
    env_name = normalize_env_name(env_name)
    task_id = f"robotouille/{env_name}"
    stem = env_name.rsplit("/", 1)[-1]
    lookup[env_name] = int(optimal_steps)
    lookup[task_id] = int(optimal_steps)
    lookup[stem] = int(optimal_steps)


def load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return {}


def iter_task_optimal_pairs(payload: Any) -> Iterable[tuple[str, int]]:
    if not isinstance(payload, Mapping):
        return
    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            env = task.get("environment_name") or task.get("task_id")
            optimal = task.get("optimal_steps")
            if env and optimal is not None:
                yield str(env), int(float(optimal))

    evaluation = payload.get("evaluation", payload)
    if isinstance(evaluation, Mapping):
        names = evaluation.get("environment_names") or []
        steps = evaluation.get("optimal_steps") or []
        if isinstance(names, list) and isinstance(steps, list):
            for env, optimal in zip(names, steps):
                yield str(env), int(float(optimal))


def load_records(
    spec: RunSpec,
    optimal_lookup: Mapping[str, int],
    warnings: list[str],
) -> list[EpisodeRecord]:
    path = spec.path
    if not path.exists():
        warnings.append(f"Input path does not exist: {path}")
        return []

    json_files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    records: list[EpisodeRecord] = []
    for json_file in json_files:
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"Could not read {json_file}: {exc}")
            continue
        records.extend(records_from_payload(payload, json_file, spec, optimal_lookup))
    return records


def records_from_payload(
    payload: Any,
    path: Path,
    spec: RunSpec,
    optimal_lookup: Mapping[str, int],
) -> list[EpisodeRecord]:
    if not isinstance(payload, Mapping):
        return []
    if is_nsfsm_episode(payload):
        return [record_from_nsfsm_episode(payload, path, spec, optimal_lookup)]
    if isinstance(payload.get("results"), list):
        return [
            record_from_nsfsm_summary_row(row, path, spec, optimal_lookup)
            for row in payload["results"]
            if isinstance(row, Mapping) and row.get("task_id")
        ]
    if looks_like_official_robotouille_results(payload):
        return records_from_official_results(payload, path, spec, optimal_lookup)
    return []


def is_nsfsm_episode(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("dataset") == "robotouille"
        and bool(payload.get("task_id"))
        and ("trajectory" in payload or "metadata" in payload)
        and "success" in payload
    )


def looks_like_official_robotouille_results(payload: Mapping[str, Any]) -> bool:
    return any(
        isinstance(value, Mapping) and {"done", "steps"}.issubset(value.keys())
        for key, value in payload.items()
        if key not in {"accuracy", "average_steps"}
    )


def records_from_official_results(
    payload: Mapping[str, Any],
    path: Path,
    spec: RunSpec,
    optimal_lookup: Mapping[str, int],
) -> list[EpisodeRecord]:
    records = []
    for key, value in payload.items():
        if key in {"accuracy", "average_steps"} or not isinstance(value, Mapping):
            continue
        if not {"done", "steps"}.issubset(value.keys()):
            continue
        env_name, seed = split_env_seed(str(key))
        suite = spec.suite if spec.suite != "unknown" else suite_from_env_name(env_name)
        task_id = f"robotouille/{normalize_env_name(env_name)}"
        optimal = find_optimal(task_id, optimal_lookup)
        steps = as_int(value.get("steps"))
        max_steps = as_float(value.get("max_steps"))
        if max_steps is None and optimal:
            max_steps = optimal * 1.5
        records.append(
            EpisodeRecord(
                method=spec.method,
                suite=suite,
                task_id=task_id,
                task_name=task_display_name(task_id),
                seed=seed,
                success=bool(value.get("done")),
                steps=steps,
                max_steps=max_steps,
                optimal_steps=optimal,
                termination="success" if value.get("done") else "max_steps_or_incomplete",
                source_path=str(path),
                simulator_steps=steps,
                total_tokens=None,
                raw=dict(value),
            )
        )
    return records


def record_from_nsfsm_episode(
    payload: Mapping[str, Any],
    path: Path,
    spec: RunSpec,
    optimal_lookup: Mapping[str, int],
) -> EpisodeRecord:
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), Mapping) else {}
    task_spec = metadata.get("task_spec", {}) if isinstance(metadata.get("task_spec"), Mapping) else {}
    task_meta = task_spec.get("metadata", {}) if isinstance(task_spec.get("metadata"), Mapping) else {}
    adapter_summary = (
        metadata.get("adapter_summary", {}) if isinstance(metadata.get("adapter_summary"), Mapping) else {}
    )
    task_id = str(payload.get("task_id") or task_spec.get("task_id") or "robotouille/unknown")
    env_name = (
        task_meta.get("environment_name")
        or adapter_summary.get("environment_name")
        or task_id.replace("robotouille/", "")
    )
    suite = spec.suite if spec.suite != "unknown" else suite_from_env_name(str(env_name))
    optimal = (
        as_int(task_meta.get("ground_truth_task", {}).get("optimal_steps"))
        if isinstance(task_meta.get("ground_truth_task"), Mapping)
        else None
    )
    if optimal is None:
        optimal = find_optimal(task_id, optimal_lookup)
    simulator_steps = as_int(adapter_summary.get("total_steps"))
    controller_steps = as_int(payload.get("total_steps"))
    steps = simulator_steps if simulator_steps is not None else controller_steps
    max_steps = as_float(task_spec.get("max_steps"))
    if max_steps is None and optimal:
        max_steps = optimal * 1.5
    seed = str(payload.get("seed") or task_meta.get("task_seed") or "")
    return EpisodeRecord(
        method=spec.method,
        suite=suite,
        task_id=normalize_task_id(task_id, str(env_name)),
        task_name=task_display_name(task_id),
        seed=seed,
        success=bool(payload.get("success")),
        steps=steps,
        max_steps=max_steps,
        optimal_steps=optimal,
        termination=str(payload.get("termination") or adapter_summary.get("termination") or "unknown"),
        source_path=str(path),
        controller_steps=controller_steps,
        simulator_steps=simulator_steps,
        total_tokens=sum_token_usage(payload),
        trajectory=list(payload.get("trajectory", [])) if isinstance(payload.get("trajectory"), list) else [],
        raw=dict(payload),
    )


def record_from_nsfsm_summary_row(
    row: Mapping[str, Any],
    path: Path,
    spec: RunSpec,
    optimal_lookup: Mapping[str, int],
) -> EpisodeRecord:
    task_id = str(row.get("task_id") or "robotouille/unknown")
    suite = spec.suite if spec.suite != "unknown" else suite_from_env_name(task_id)
    optimal = find_optimal(task_id, optimal_lookup)
    simulator_steps = as_int(row.get("simulator_steps"))
    controller_steps = as_int(row.get("total_steps"))
    steps = simulator_steps if simulator_steps is not None else controller_steps
    max_steps = optimal * 1.5 if optimal else None
    return EpisodeRecord(
        method=spec.method,
        suite=suite,
        task_id=task_id,
        task_name=task_display_name(task_id),
        seed=str(row.get("seed") or ""),
        success=bool(row.get("success")),
        steps=steps,
        max_steps=max_steps,
        optimal_steps=optimal,
        termination=str(row.get("termination") or "unknown"),
        source_path=str(path),
        controller_steps=controller_steps,
        simulator_steps=simulator_steps,
        raw=dict(row),
    )


def split_env_seed(key: str) -> tuple[str, str]:
    match = re.match(r"(.+)_([0-9]+)$", key)
    if match:
        return normalize_env_name(match.group(1)), match.group(2)
    return normalize_env_name(key), ""


def normalize_env_name(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("robotouille/", "")
    text = text.strip("/")
    return text


def normalize_task_id(task_id: str, env_name: str = "") -> str:
    task = str(task_id or "").strip()
    if task.startswith("robotouille/"):
        return task
    env = normalize_env_name(env_name or task)
    return f"robotouille/{env}"


def suite_from_env_name(value: str) -> str:
    lowered = str(value).lower()
    if "asynchronous" in lowered:
        return "async"
    if "synchronous" in lowered:
        return "sync"
    return "unknown"


def find_optimal(task_id: str, lookup: Mapping[str, int]) -> int | None:
    candidates = [
        str(task_id),
        normalize_env_name(str(task_id)),
        normalize_env_name(str(task_id)).rsplit("/", 1)[-1],
        str(task_id).replace("robotouille/", ""),
    ]
    for candidate in candidates:
        if candidate in lookup:
            return int(lookup[candidate])
    return None


def task_display_name(task_id: str) -> str:
    stem = normalize_env_name(task_id).rsplit("/", 1)[-1]
    stem = re.sub(r"^[0-9]+(?:\.[0-9]+)?_", "", stem)
    return stem.replace("_", " ")


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sum_token_usage(payload: Any) -> int | None:
    total = 0
    found = False

    def walk(node: Any) -> None:
        nonlocal total, found
        if isinstance(node, Mapping):
            usage = node.get("usage")
            if isinstance(usage, Mapping):
                usage_total = as_int(usage.get("total_tokens"))
                if usage_total is None:
                    prompt = as_int(usage.get("prompt_tokens")) or 0
                    completion = as_int(usage.get("completion_tokens")) or 0
                    usage_total = prompt + completion if prompt or completion else None
                if usage_total is not None:
                    total += usage_total
                    found = True
                    return
            direct_total = as_int(node.get("total_tokens"))
            if direct_total is not None:
                total += direct_total
                found = True
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return total if found else None


def scan_log_tokens(path: Path) -> int:
    root = path if path.is_dir() else path.parent
    total = 0
    for file_path in root.rglob("*"):
        if file_path.suffix.lower() not in {".txt", ".log"}:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in re.finditer(r"(?:Prompt|Completion|Total)\s+Tokens:\s*([0-9]+)", text, re.I):
            total += int(match.group(1))
    return total


def load_failure_labels(path_value: str) -> dict[tuple[str, str, str, str], str]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    labels: dict[tuple[str, str, str, str], str] = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            suite = normalize_suite(row.get("suite", ""))
            task_id = row.get("task_id", "").strip()
            seed = row.get("seed", "").strip()
            category = row.get("category", "").strip()
            if method and suite and task_id and category:
                labels[(method, suite, task_id, seed)] = category
                labels[(method, suite, task_id, "")] = category
    return labels


def failure_category(
    record: EpisodeRecord,
    labels: Mapping[tuple[str, str, str, str], str],
) -> str:
    for key in [
        (record.method, record.suite, record.task_id, record.seed),
        (record.method, record.suite, record.task_id, ""),
    ]:
        label = labels.get(key)
        if label:
            return normalize_failure_category(label)

    text = " ".join(
        [
            record.termination,
            json.dumps(record.trajectory[-5:], ensure_ascii=False) if record.trajectory else "",
        ]
    ).lower()
    if any(token in text for token in ["vocabulary", "invalid action", "not executable", "does not exist", "unknown action"]):
        return "Action-vocabulary mismatch"
    if any(token in text for token in ["goal", "satisfied", "omitted", "ingredient", "order", "serving"]):
        return "Spec-literal / goal grounding error"
    if record.budget_hit:
        return "Step inflation / inefficient execution"
    return "Other / unknown"


def normalize_failure_category(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"step", "step inflation", "inefficient", "inefficient execution"}:
        return "Step inflation / inefficient execution"
    if lowered in {"vocab", "vocabulary", "action vocabulary", "action-vocabulary mismatch"}:
        return "Action-vocabulary mismatch"
    if lowered in {"goal", "grounding", "spec", "spec-literal", "spec-literal / goal grounding error"}:
        return "Spec-literal / goal grounding error"
    for category in FAILURE_CATEGORIES:
        if lowered == category.lower():
            return category
    return value.strip() or "Other / unknown"


def compute_suite_metrics(
    records: list[EpisodeRecord],
    specs: list[RunSpec],
    token_totals_by_spec: Mapping[tuple[str, str, str], int],
) -> list[dict[str, Any]]:
    rows = []
    groups = sorted(
        {(record.method, record.suite) for record in records},
        key=lambda item: (suite_sort_key(item[1]), item[0]),
    )
    for method, suite in groups:
        items = [record for record in records if record.method == method and record.suite == suite]
        if not items:
            continue
        step_ratios = [record.step_ratio for record in items if record.step_ratio is not None]
        json_tokens = [record.total_tokens for record in items if record.total_tokens is not None]
        log_tokens = sum(
            total
            for (spec_method, spec_suite, _path), total in token_totals_by_spec.items()
            if spec_method == method and spec_suite == suite
        )
        total_tokens = sum(json_tokens) if json_tokens else (log_tokens or None)
        token_source = "json_usage" if json_tokens else ("log_files" if log_tokens else "missing")
        rows.append(
            {
                "method": method,
                "suite": suite,
                "suite_label": suite_display(suite),
                "runs": len(items),
                "successes": sum(1 for item in items if item.success),
                "success_rate": safe_div(sum(1 for item in items if item.success), len(items)),
                "steps_over_optimal": mean_or_none(step_ratios),
                "budget_hits": sum(1 for item in items if item.budget_hit),
                "budget_hit_rate": safe_div(sum(1 for item in items if item.budget_hit), len(items)),
                "total_tokens": total_tokens,
                "tokens_m": (total_tokens / 1_000_000.0) if total_tokens is not None else None,
                "token_source": token_source,
            }
        )

    # Preserve empty requested method/suite combinations in the manifest-friendly CSV.
    seen = {(row["method"], row["suite"]) for row in rows}
    for spec in specs:
        if (spec.method, spec.suite) not in seen:
            rows.append(
                {
                    "method": spec.method,
                    "suite": spec.suite,
                    "suite_label": suite_display(spec.suite),
                    "runs": 0,
                    "successes": 0,
                    "success_rate": None,
                    "steps_over_optimal": None,
                    "budget_hits": 0,
                    "budget_hit_rate": None,
                    "total_tokens": None,
                    "tokens_m": None,
                    "token_source": "missing",
                }
            )
    return sorted(rows, key=lambda row: (suite_sort_key(row["suite"]), row["method"]))


def compute_task_difficulty(records: list[EpisodeRecord]) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[tuple[str, str, str], list[EpisodeRecord]] = {}
    for record in records:
        grouped.setdefault((record.method, record.suite, record.task_id), []).append(record)

    for (method, suite, task_id), items in grouped.items():
        optimal_values = [item.optimal_steps for item in items if item.optimal_steps is not None]
        optimal = int(statistics.median(optimal_values)) if optimal_values else None
        rows.append(
            {
                "method": method,
                "suite": suite,
                "suite_label": suite_display(suite),
                "task_id": task_id,
                "task_name": items[0].task_name,
                "optimal_steps": optimal,
                "runs": len(items),
                "successes": sum(1 for item in items if item.success),
                "success_rate": safe_div(sum(1 for item in items if item.success), len(items)),
                "avg_steps": mean_or_none([item.steps for item in items if item.steps is not None]),
                "budget_hit_rate": safe_div(sum(1 for item in items if item.budget_hit), len(items)),
            }
        )
    return sorted(rows, key=lambda row: (suite_sort_key(row["suite"]), row["optimal_steps"] or 0, row["method"]))


def compute_failure_attribution(
    records: list[EpisodeRecord],
    labels: Mapping[tuple[str, str, str, str], str],
) -> list[dict[str, Any]]:
    rows = []
    groups = sorted(
        {(record.method, record.suite) for record in records},
        key=lambda item: (suite_sort_key(item[1]), item[0]),
    )
    for method, suite in groups:
        items = [record for record in records if record.method == method and record.suite == suite]
        budget_items = [record for record in items if record.budget_hit]
        counts = {category: 0 for category in FAILURE_CATEGORIES}
        for record in budget_items:
            category = failure_category(record, labels)
            if category not in counts:
                counts[category] = 0
            counts[category] += 1
        for category, count in counts.items():
            rows.append(
                {
                    "method": method,
                    "suite": suite,
                    "suite_label": suite_display(suite),
                    "category": category,
                    "count": count,
                    "total_runs": len(items),
                    "budget_hits": len(budget_items),
                    "rate_of_all_runs": safe_div(count, len(items)),
                    "rate_of_budget_hits": safe_div(count, len(budget_items)),
                }
            )
    return rows


def select_case_studies(
    records: list[EpisodeRecord],
    labels: Mapping[tuple[str, str, str, str], str],
    limit: int,
) -> list[dict[str, Any]]:
    failed = [record for record in records if not record.success]
    failed.sort(
        key=lambda item: (
            item.budget_hit,
            item.optimal_steps or 0,
            item.steps or 0,
            len(item.trajectory),
        ),
        reverse=True,
    )
    selected = []
    seen: set[tuple[str, str, str]] = set()
    for record in failed:
        key = (record.method, record.suite, record.task_id)
        if key in seen:
            continue
        seen.add(key)
        selected.append(case_study_row(record, failure_category(record, labels)))
        if len(selected) >= limit:
            break
    return selected


def case_study_row(record: EpisodeRecord, category: str) -> dict[str, Any]:
    actions = []
    for step in record.trajectory:
        action = step.get("action") or step.get("fsm_action")
        if action:
            actions.append(str(action))
    if len(actions) > 10:
        actions = actions[:5] + ["..."] + actions[-4:]
    return {
        "method": record.method,
        "suite": record.suite,
        "suite_label": suite_display(record.suite),
        "task_id": record.task_id,
        "task_name": record.task_name,
        "seed": record.seed,
        "success": record.success,
        "termination": record.termination,
        "steps": record.steps,
        "max_steps": record.max_steps,
        "optimal_steps": record.optimal_steps,
        "budget_hit": record.budget_hit,
        "failure_category": category,
        "actions": actions,
        "source_path": record.source_path,
    }


def episode_rows(records: list[EpisodeRecord]) -> list[dict[str, Any]]:
    return [
        {
            "method": record.method,
            "suite": record.suite,
            "suite_label": suite_display(record.suite),
            "task_id": record.task_id,
            "task_name": record.task_name,
            "seed": record.seed,
            "success": record.success,
            "termination": record.termination,
            "steps": record.steps,
            "controller_steps": record.controller_steps,
            "simulator_steps": record.simulator_steps,
            "max_steps": record.max_steps,
            "optimal_steps": record.optimal_steps,
            "steps_over_optimal": record.step_ratio,
            "budget_hit": record.budget_hit,
            "total_tokens": record.total_tokens,
            "source_path": record.source_path,
        }
        for record in records
    ]


def safe_div(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


def mean_or_none(values: Iterable[Any]) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    if not materialized:
        return None
    return statistics.mean(materialized)


def suite_sort_key(suite: str) -> int:
    return {"sync": 0, "async": 1}.get(suite, 99)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_case_studies(path: Path, rows: list[Mapping[str, Any]]) -> None:
    lines = ["# Robotouille Trajectory Case Studies", ""]
    if not rows:
        lines.append("No failed case studies were available.")
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {idx}. {row['task_name']} ({row['suite_label']}, {row['method']})",
                "",
                f"- task_id: `{row['task_id']}`",
                f"- seed: `{row.get('seed') or 'unknown'}`",
                f"- termination: `{row['termination']}`",
                f"- budget_hit: `{row['budget_hit']}`",
                f"- failure_category: `{row['failure_category']}`",
                f"- steps / optimal: `{row.get('steps')}` / `{row.get('optimal_steps')}`",
                f"- source: `{row['source_path']}`",
                "",
            ]
        )
        actions = row.get("actions") or []
        if actions:
            lines.append("Action flow:")
            lines.append("")
            lines.append("```text")
            lines.append(" -> ".join(str(action) for action in actions))
            lines.append("```")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plots. Re-run with --no-plots or install matplotlib."
        ) from exc
    return plt


def plot_suite_metrics(rows: list[Mapping[str, Any]], fig_dir: Path) -> list[Path]:
    plt = require_matplotlib()
    methods = sorted({str(row["method"]) for row in rows})
    suites = [suite for suite in ["sync", "async"] if any(row["suite"] == suite for row in rows)]
    metrics = [
        ("success_rate", "SR", "Success rate", 1.0, False),
        ("steps_over_optimal", "Steps/Optimal", "Ratio", None, False),
        ("budget_hit_rate", "Budget-hit rate", "Rate (%)", 1.0, True),
        ("tokens_m", "Tokens (M)", "Millions", None, False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    width = 0.8 / max(len(methods), 1)
    x_base = list(range(len(suites)))
    for ax, (metric, title, ylabel, ymax, percent) in zip(axes, metrics):
        for method_idx, method in enumerate(methods):
            vals = []
            missing = []
            for suite in suites:
                row = find_suite_row(rows, method, suite)
                value = row.get(metric) if row else None
                missing.append(value is None or value == "")
                vals.append(float(value) if value not in {None, ""} else 0.0)
            xs = [x + (method_idx - (len(methods) - 1) / 2) * width for x in x_base]
            color = METHOD_COLORS.get(method, None)
            bars = ax.bar(xs, [v * 100 if percent else v for v in vals], width, label=method, color=color)
            for bar, raw_value, is_missing in zip(bars, vals, missing):
                if is_missing:
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "n/a", ha="center", va="bottom", fontsize=8)
                else:
                    display_value = raw_value * 100 if percent else raw_value
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{display_value:.2f}" if metric == "tokens_m" else f"{display_value:.2f}".rstrip("0").rstrip("."),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_base)
        ax.set_xticklabels([suite_display(suite) for suite in suites])
        if ymax is not None:
            ax.set_ylim(0, 100 if percent else ymax)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Suite-level performance")
    fig.tight_layout()
    out = fig_dir / "suite_level_performance.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return [out]


def find_suite_row(rows: list[Mapping[str, Any]], method: str, suite: str) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("method") == method and row.get("suite") == suite:
            return row
    return None


def plot_task_difficulty(rows: list[Mapping[str, Any]], fig_dir: Path) -> list[Path]:
    plt = require_matplotlib()
    filtered = [row for row in rows if row.get("optimal_steps") not in {None, ""}]
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = sorted({str(row["method"]) for row in filtered})
    for method in methods:
        for suite, marker in [("sync", "o"), ("async", "^")]:
            subset = [row for row in filtered if row["method"] == method and row["suite"] == suite]
            if not subset:
                continue
            xs = [float(row["optimal_steps"]) for row in subset]
            ys = [float(row["success_rate"]) for row in subset]
            ax.scatter(
                xs,
                ys,
                marker=marker,
                s=55,
                color=METHOD_COLORS.get(method),
                label=f"{method} {suite_display(suite)}",
                alpha=0.9,
            )
    ax.set_title("Task difficulty curve")
    ax.set_xlabel("Optimal steps (lower is easier)")
    ax.set_ylabel("Success rate (SR)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "task_difficulty_curve.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return [out]


def plot_failure_attribution(rows: list[Mapping[str, Any]], fig_dir: Path) -> list[Path]:
    plt = require_matplotlib()
    groups = sorted(
        {(row["method"], row["suite"]) for row in rows},
        key=lambda item: (suite_sort_key(str(item[1])), str(item[0])),
    )
    if not groups:
        return []
    x = list(range(len(groups)))
    bottoms = [0.0] * len(groups)
    colors = ["#f39c9c", "#d7191c", "#f28e2b", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for category, color in zip(FAILURE_CATEGORIES, colors):
        vals = []
        for method, suite in groups:
            row = next(
                (
                    item
                    for item in rows
                    if item["method"] == method and item["suite"] == suite and item["category"] == category
                ),
                None,
            )
            vals.append(float(row["rate_of_all_runs"]) * 100 if row else 0.0)
        ax.bar(x, vals, bottom=bottoms, label=category, color=color)
        bottoms = [a + b for a, b in zip(bottoms, vals)]
    for xpos, total in zip(x, bottoms):
        ax.text(xpos, total, f"{total:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Failure-mode attribution of budget-hit runs")
    ax.set_ylabel("Budget-hit rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{suite_display(str(suite))}\n{method}" for method, suite in groups])
    ax.set_ylim(0, max([100.0] + bottoms) * 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out = fig_dir / "failure_mode_attribution.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return [out]


def plot_case_studies(rows: list[Mapping[str, Any]], fig_dir: Path) -> list[Path]:
    rows_with_actions = [row for row in rows if row.get("actions")]
    if not rows_with_actions:
        return []
    plt = require_matplotlib()
    n = min(len(rows_with_actions), 4)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(2.2 * n, 3)), squeeze=False)
    for ax, row in zip([item for sub in axes for item in sub], rows_with_actions[:n]):
        ax.axis("off")
        actions = [str(action) for action in row.get("actions", [])][:9]
        title = f"{row['task_name']} ({row['suite_label']}, {row['method']})"
        ax.text(0.0, 0.95, title, ha="left", va="top", fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(
            0.0,
            0.78,
            f"{row['failure_category']} | termination={row['termination']}",
            ha="left",
            va="top",
            fontsize=9,
            color="#444444",
            transform=ax.transAxes,
        )
        if not actions:
            continue
        y = 0.35
        box_w = min(0.13, 0.86 / max(len(actions), 1))
        gap = 0.015
        for idx, action in enumerate(actions):
            x = 0.02 + idx * (box_w + gap)
            label = shorten(action.replace("|", "\n"), 24)
            rect = plt.Rectangle((x, y), box_w, 0.22, fill=False, edgecolor="#1f5cff", linewidth=1.2, transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(x + box_w / 2, y + 0.11, label, ha="center", va="center", fontsize=7, transform=ax.transAxes)
            if idx < len(actions) - 1:
                ax.annotate(
                    "",
                    xy=(x + box_w + gap * 0.7, y + 0.11),
                    xytext=(x + box_w, y + 0.11),
                    arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                )
    fig.tight_layout()
    out = fig_dir / "trajectory_case_studies.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return [out]


def shorten(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, max_lines=2, placeholder="..."))


def plot_dashboard(
    suite_rows: list[Mapping[str, Any]],
    task_rows: list[Mapping[str, Any]],
    attribution_rows: list[Mapping[str, Any]],
    fig_dir: Path,
) -> list[Path]:
    # Keep the dashboard intentionally compact; the separate figures are the
    # paper-ready versions.
    plt = require_matplotlib()
    if not suite_rows and not task_rows and not attribution_rows:
        return []
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 2)
    ax_suite = fig.add_subplot(grid[0, 0])
    ax_diff = fig.add_subplot(grid[0, 1])
    ax_attr = fig.add_subplot(grid[1, :])

    plot_suite_sr_on_axis(ax_suite, suite_rows)
    plot_task_difficulty_on_axis(ax_diff, task_rows)
    plot_failure_attribution_on_axis(ax_attr, attribution_rows)

    fig.suptitle("Robotouille evaluation dashboard")
    fig.tight_layout()
    out = fig_dir / "robotouille_eval_dashboard.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return [out]


def plot_suite_sr_on_axis(ax: Any, rows: list[Mapping[str, Any]]) -> None:
    groups = sorted(
        {(row["method"], row["suite"]) for row in rows},
        key=lambda item: (suite_sort_key(str(item[1])), str(item[0])),
    )
    vals = [
        float(next(row["success_rate"] for row in rows if row["method"] == method and row["suite"] == suite) or 0.0)
        for method, suite in groups
    ]
    ax.bar(range(len(groups)), vals, color=[METHOD_COLORS.get(str(method), "#777777") for method, _suite in groups])
    ax.set_title("SR")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{suite_display(str(suite))}\n{method}" for method, suite in groups], fontsize=8)
    ax.grid(axis="y", alpha=0.25)


def plot_task_difficulty_on_axis(ax: Any, rows: list[Mapping[str, Any]]) -> None:
    filtered = [row for row in rows if row.get("optimal_steps") not in {None, ""}]
    for method in sorted({str(row["method"]) for row in filtered}):
        for suite, marker in [("sync", "o"), ("async", "^")]:
            subset = [row for row in filtered if row["method"] == method and row["suite"] == suite]
            if subset:
                ax.scatter(
                    [float(row["optimal_steps"]) for row in subset],
                    [float(row["success_rate"]) for row in subset],
                    marker=marker,
                    color=METHOD_COLORS.get(method),
                    label=f"{method} {suite_display(suite)}",
                )
    ax.set_title("Difficulty curve")
    ax.set_xlabel("Optimal steps")
    ax.set_ylabel("SR")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)


def plot_failure_attribution_on_axis(ax: Any, rows: list[Mapping[str, Any]]) -> None:
    groups = sorted(
        {(row["method"], row["suite"]) for row in rows},
        key=lambda item: (suite_sort_key(str(item[1])), str(item[0])),
    )
    x = list(range(len(groups)))
    bottoms = [0.0] * len(groups)
    colors = ["#f39c9c", "#d7191c", "#f28e2b", "#bdbdbd"]
    for category, color in zip(FAILURE_CATEGORIES, colors):
        vals = []
        for method, suite in groups:
            row = next(
                (
                    item
                    for item in rows
                    if item["method"] == method and item["suite"] == suite and item["category"] == category
                ),
                None,
            )
            vals.append(float(row["rate_of_all_runs"]) * 100 if row else 0.0)
        ax.bar(x, vals, bottom=bottoms, color=color, label=category)
        bottoms = [a + b for a, b in zip(bottoms, vals)]
    ax.set_title("Budget-hit attribution")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{suite_display(str(suite))}\n{method}" for method, suite in groups], fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)


if __name__ == "__main__":
    main()

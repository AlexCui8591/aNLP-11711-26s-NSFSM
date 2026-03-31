"""
Full-Scale Experiment — 67 goals × 15 runs × {ReAct, Reflexion}

Features:
  - Incremental save: each (goal, run) result is saved immediately
  - Resume: skips already-completed (goal, run) pairs on restart
  - Error classification and aggregate statistics
  - Supports running ReAct-only, Reflexion-only, or both
  - Supports filtering by group or goal subset
  - Progress tracking with ETA

Usage:
  cd ns-fsm-baseline

  # Full experiment (both agents, all 67 goals, 15 runs)
  python scripts/run_full_experiment.py

  # ReAct only
  python scripts/run_full_experiment.py --agent react

  # Reflexion only
  python scripts/run_full_experiment.py --agent reflexion

  # Specific groups
  python scripts/run_full_experiment.py --groups Wooden Stone

  # Fewer runs (e.g. for testing)
  python scripts/run_full_experiment.py --runs 3

  # Specific goals
  python scripts/run_full_experiment.py --goals stick wooden_pickaxe iron_pickaxe

  # Resume interrupted run (auto-detects from output dir)
  python scripts/run_full_experiment.py --resume

  # Custom output tag (default: timestamp)
  python scripts/run_full_experiment.py --tag v1
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from collections import OrderedDict

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "..", "MC-TextWorld"))

from react_agent import ReactAgent
from reflexion_agent import ReflexionAgent

CONFIG_PATH = os.path.join(ROOT, "config", "hyperparams.yaml")
GOALS_PATH  = os.path.join(ROOT, "config", "goals_67.json")
RESULTS_BASE = os.path.join(ROOT, "results", "full")


# ═════════════════════════════════════════════════════════════════════════════
# Error classification (reused from pilot_stress)
# ═════════════════════════════════════════════════════════════════════════════

def classify_errors(result: dict) -> dict:
    """Classify the dominant error type from a single episode trajectory."""
    traj = result.get("trajectory", [])
    unknown_action = 0
    missing_tool = 0
    missing_material = 0
    total_fail = 0
    total_ok = 0

    repeated_same_fail = 0
    prev_action = None
    streak = 0

    for step in traj:
        if step.get("success"):
            total_ok += 1
            prev_action = None
            streak = 0
        else:
            total_fail += 1
            msg = step.get("message", "")
            if "NOT exist" in msg or "not in the action library" in msg:
                unknown_action += 1
            elif "missing tool" in msg:
                missing_tool += 1
            elif "missing material" in msg:
                missing_material += 1

            if step["action"] == prev_action:
                streak += 1
                if streak >= 2:
                    repeated_same_fail += 1
            else:
                prev_action = step["action"]
                streak = 1

    total = total_ok + total_fail
    efficiency = total_ok / total if total > 0 else 0.0

    if unknown_action > 0:
        dominant = "plan_knowledge_error"
    elif missing_tool >= missing_material and missing_tool > 0:
        dominant = "tool_dependency_error"
    elif missing_material > 0:
        dominant = "resource_shortage"
    elif repeated_same_fail > 0:
        dominant = "dead_loop"
    elif total_fail == 0:
        dominant = "none"
    else:
        dominant = "sequencing_error"

    return {
        "unknown_action": unknown_action,
        "missing_tool": missing_tool,
        "missing_material": missing_material,
        "repeated_same_fail": repeated_same_fail,
        "total_failed_steps": total_fail,
        "total_successful_steps": total_ok,
        "efficiency": round(efficiency, 3),
        "dominant_error_type": dominant,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Goal loading
# ═════════════════════════════════════════════════════════════════════════════

def load_goals(groups_filter=None, goals_filter=None):
    """
    Load goals from goals_67.json.
    Returns list of {"goal": str, "group": str, "type": str, "instruction": str}.
    """
    with open(GOALS_PATH, encoding="utf-8") as f:
        all_goals = json.load(f)

    result = []
    for group_name, group_data in all_goals.items():
        if groups_filter and group_name not in groups_filter:
            continue
        for entry in group_data["goals"]:
            if goals_filter and entry["goal"] not in goals_filter:
                continue
            result.append({
                "goal": entry["goal"],
                "group": group_name,
                "type": entry["type"],
                "instruction": entry["instruction"],
            })
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Result file management (for incremental save + resume)
# ═════════════════════════════════════════════════════════════════════════════

def result_path(out_dir: str, agent_name: str, goal: str, run_id: int) -> str:
    """Path for a single (agent, goal, run) result file."""
    return os.path.join(out_dir, agent_name, f"{goal}_run{run_id:02d}.json")


def find_completed(out_dir: str, agent_name: str, goals: list, num_runs: int) -> set:
    """Find already-completed (goal, run_id) pairs by scanning existing result files."""
    completed = set()
    agent_dir = os.path.join(out_dir, agent_name)
    if not os.path.isdir(agent_dir):
        return completed
    for goal_entry in goals:
        goal = goal_entry["goal"]
        for run_id in range(1, num_runs + 1):
            path = result_path(out_dir, agent_name, goal, run_id)
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    if "success" in data or "goal" in data:
                        completed.add((goal, run_id))
                except (json.JSONDecodeError, KeyError):
                    pass  # corrupt file, will re-run
    return completed


def save_result(out_dir: str, agent_name: str, goal: str, run_id: int, data: dict):
    """Save a single result incrementally."""
    path = result_path(out_dir, agent_name, goal, run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# Run one agent across all goals × runs
# ═════════════════════════════════════════════════════════════════════════════

def run_agent(agent_name: str, goals: list, num_runs: int,
              out_dir: str, resume: bool, verbose: bool):
    """
    Run either 'react' or 'reflexion' on all (goal, run) pairs.
    Saves results incrementally; supports resume.
    """
    # Find completed runs (if resuming)
    if resume:
        completed = find_completed(out_dir, agent_name, goals, num_runs)
        if completed:
            print(f"  [Resume] Found {len(completed)} completed {agent_name} runs, skipping them.")
    else:
        completed = set()

    # Build task list
    tasks = []
    for goal_entry in goals:
        for run_id in range(1, num_runs + 1):
            if (goal_entry["goal"], run_id) not in completed:
                tasks.append((goal_entry, run_id))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print(f"  [{agent_name}] All runs already completed. Nothing to do.")
        return

    total_possible = len(goals) * num_runs
    print(f"\n{'='*70}")
    print(f"  [{agent_name.upper()}] {total_tasks} runs to execute "
          f"({total_possible - total_tasks} already done / {total_possible} total)")
    print(f"  Goals: {len(goals)}  |  Runs per goal: {num_runs}")
    print(f"{'='*70}\n")

    # Create agent (single instance, reused across episodes)
    if agent_name == "react":
        agent = ReactAgent(config_path=CONFIG_PATH, verbose=verbose)
    else:
        agent = ReflexionAgent(config_path=CONFIG_PATH, verbose=verbose)

    # Run
    times = []
    for idx, (goal_entry, run_id) in enumerate(tasks):
        goal = goal_entry["goal"]
        group = goal_entry["group"]

        print(f"\n{'─'*70}")
        print(f"  [{agent_name.upper()}] ({idx+1}/{total_tasks}) "
              f"{goal} (run {run_id}/{num_runs})  [{group}]")
        if times:
            avg_t = sum(times) / len(times)
            remaining = (total_tasks - idx) * avg_t
            eta_min = remaining / 60
            if eta_min > 60:
                print(f"  ETA: ~{eta_min/60:.1f}h ({avg_t:.1f}s/episode avg)")
            else:
                print(f"  ETA: ~{eta_min:.0f}min ({avg_t:.1f}s/episode avg)")
        print(f"{'─'*70}")

        t0 = time.time()
        try:
            if agent_name == "react":
                result = agent.run_episode(goal)
            else:
                result = agent.run(goal)
            elapsed = time.time() - t0
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  [ERROR] {agent_name} crashed on {goal} run {run_id}: {exc}")
            result = {
                "goal": goal,
                "success": False,
                "total_steps": 0,
                "termination": "crash",
                "error": str(exc),
                "trajectory": [],
            }

        # Annotate result
        result["run_id"] = run_id
        result["group"] = group
        result["elapsed_sec"] = round(elapsed, 2)

        # Error classification
        if agent_name == "react":
            result["errors"] = classify_errors(result)
        else:
            # For reflexion, classify the last attempt
            for attempt in result.get("attempts", []):
                attempt["result"]["errors"] = classify_errors(attempt["result"])
            last_attempt = (result.get("attempts", [{}])[-1] or {}).get("result", {})
            result["final_errors"] = classify_errors(last_attempt)

        # Save immediately
        save_result(out_dir, agent_name, goal, run_id, result)
        times.append(elapsed)

        tag = "OK" if result.get("success") else "FAIL"
        if agent_name == "react":
            steps = result.get("total_steps", "?")
            term = result.get("termination", "?")
            print(f"  => [{tag}] {steps} steps, {term}, {elapsed:.1f}s")
        else:
            n_att = result.get("total_attempts", "?")
            win = result.get("winning_attempt")
            print(f"  => [{tag}] {n_att} attempts, "
                  f"winning={win or 'none'}, {elapsed:.1f}s")

    total_time = sum(times)
    print(f"\n  [{agent_name.upper()}] Finished {total_tasks} runs in "
          f"{total_time/3600:.1f}h ({total_time/60:.0f}min)")


# ═════════════════════════════════════════════════════════════════════════════
# Aggregate results and produce summary
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_results(out_dir: str, agent_name: str, goals: list, num_runs: int) -> dict:
    """Load all saved results for one agent, compute aggregate stats."""
    all_results = []
    for goal_entry in goals:
        goal = goal_entry["goal"]
        group = goal_entry["group"]
        goal_results = []
        for run_id in range(1, num_runs + 1):
            path = result_path(out_dir, agent_name, goal, run_id)
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                goal_results.append(data)
        if goal_results:
            all_results.append({
                "goal": goal,
                "group": group,
                "runs": goal_results,
            })

    # Per-goal success rate
    goal_stats = []
    for entry in all_results:
        runs = entry["runs"]
        n_success = sum(1 for r in runs if r.get("success"))
        n_total = len(runs)
        avg_steps = sum(r.get("total_steps", 0) for r in runs) / max(n_total, 1)

        # Collect error type distribution across failed runs
        error_dist = {}
        for r in runs:
            if not r.get("success"):
                if agent_name == "react":
                    err_type = r.get("errors", {}).get("dominant_error_type", "unknown")
                else:
                    err_type = r.get("final_errors", {}).get("dominant_error_type", "unknown")
                error_dist[err_type] = error_dist.get(err_type, 0) + 1

        # Termination distribution
        term_dist = {}
        if agent_name == "react":
            for r in runs:
                t = r.get("termination", "unknown")
                term_dist[t] = term_dist.get(t, 0) + 1
        else:
            for r in runs:
                for att in r.get("attempts", []):
                    t = att.get("result", {}).get("termination", "unknown")
                    term_dist[t] = term_dist.get(t, 0) + 1

        goal_stats.append({
            "goal": entry["goal"],
            "group": entry["group"],
            "n_runs": n_total,
            "n_success": n_success,
            "success_rate": round(n_success / max(n_total, 1), 3),
            "avg_steps": round(avg_steps, 1),
            "error_distribution": error_dist,
            "termination_distribution": term_dist,
        })

    # Group-level stats
    group_stats = {}
    for gs in goal_stats:
        g = gs["group"]
        if g not in group_stats:
            group_stats[g] = {"goals": 0, "total_runs": 0, "total_success": 0}
        group_stats[g]["goals"] += 1
        group_stats[g]["total_runs"] += gs["n_runs"]
        group_stats[g]["total_success"] += gs["n_success"]
    for g in group_stats:
        s = group_stats[g]
        s["success_rate"] = round(s["total_success"] / max(s["total_runs"], 1), 3)

    # Overall
    total_runs = sum(gs["n_runs"] for gs in goal_stats)
    total_success = sum(gs["n_success"] for gs in goal_stats)

    return {
        "agent": agent_name,
        "total_goals": len(goal_stats),
        "total_runs": total_runs,
        "total_success": total_success,
        "overall_success_rate": round(total_success / max(total_runs, 1), 3),
        "group_stats": group_stats,
        "goal_stats": goal_stats,
    }


def print_agent_summary(stats: dict):
    """Print formatted summary for one agent."""
    agent = stats["agent"].upper()
    print(f"\n{'═'*70}")
    print(f"  {agent} — RESULTS SUMMARY")
    print(f"{'═'*70}")
    print(f"  Overall: {stats['total_success']}/{stats['total_runs']} "
          f"({stats['overall_success_rate']:.1%})")

    # Group breakdown
    print(f"\n  {'Group':<12} {'Goals':>5} {'Runs':>5} {'Success':>7} {'Rate':>6}")
    print(f"  {'─'*12} {'─'*5} {'─'*5} {'─'*7} {'─'*6}")
    for group, gs in stats["group_stats"].items():
        print(f"  {group:<12} {gs['goals']:>5} {gs['total_runs']:>5} "
              f"{gs['total_success']:>7} {gs['success_rate']:>6.1%}")

    # Per-goal table
    print(f"\n  {'Goal':<22} {'Group':<10} {'SR':>8} {'AvgSteps':>8} {'TopError':<24}")
    print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*8} {'─'*24}")
    for gs in stats["goal_stats"]:
        sr = f"{gs['n_success']}/{gs['n_runs']}"
        top_err = max(gs["error_distribution"], key=gs["error_distribution"].get) \
            if gs["error_distribution"] else "none"
        print(f"  {gs['goal']:<22} {gs['group']:<10} {sr:>8} "
              f"{gs['avg_steps']:>8.1f} {top_err:<24}")

    # Error distribution across all failed runs
    print(f"\n  Error type distribution (failed runs):")
    all_errs = {}
    for gs in stats["goal_stats"]:
        for e, c in gs["error_distribution"].items():
            all_errs[e] = all_errs.get(e, 0) + c
    if all_errs:
        total_errs = sum(all_errs.values())
        for e, c in sorted(all_errs.items(), key=lambda x: -x[1]):
            print(f"    {e:<28} {c:>4} ({100*c/total_errs:.1f}%)")
    else:
        print(f"    (none — all runs succeeded)")


def print_comparison(react_stats: dict, refl_stats: dict):
    """Side-by-side comparison of ReAct vs Reflexion."""
    print(f"\n\n{'═'*70}")
    print(f"  REACT vs REFLEXION — COMPARISON")
    print(f"{'═'*70}")

    print(f"\n  Overall success rate:")
    print(f"    ReAct:     {react_stats['overall_success_rate']:.1%} "
          f"({react_stats['total_success']}/{react_stats['total_runs']})")
    print(f"    Reflexion: {refl_stats['overall_success_rate']:.1%} "
          f"({refl_stats['total_success']}/{refl_stats['total_runs']})")

    # Group comparison
    all_groups = list(react_stats["group_stats"].keys())
    print(f"\n  {'Group':<12} {'React SR':>10} {'Reflexion SR':>14} {'Delta':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*14} {'─'*8}")
    for g in all_groups:
        r_sr = react_stats["group_stats"].get(g, {}).get("success_rate", 0)
        x_sr = refl_stats["group_stats"].get(g, {}).get("success_rate", 0)
        delta = x_sr - r_sr
        sign = "+" if delta > 0 else ""
        print(f"  {g:<12} {r_sr:>10.1%} {x_sr:>14.1%} {sign}{delta:>7.1%}")

    # Per-goal comparison (show only where they differ)
    print(f"\n  Per-goal comparison (showing goals with different outcomes):")
    react_goals = {gs["goal"]: gs for gs in react_stats["goal_stats"]}
    refl_goals = {gs["goal"]: gs for gs in refl_stats["goal_stats"]}

    print(f"  {'Goal':<22} {'React SR':>10} {'Reflexion SR':>14} {'Delta':>8}")
    print(f"  {'─'*22} {'─'*10} {'─'*14} {'─'*8}")
    n_diff = 0
    for goal in react_goals:
        r_sr = react_goals[goal]["success_rate"]
        x_sr = refl_goals.get(goal, {}).get("success_rate", 0)
        if abs(r_sr - x_sr) > 0.05:  # >5% difference
            delta = x_sr - r_sr
            sign = "+" if delta > 0 else ""
            print(f"  {goal:<22} {r_sr:>10.1%} {x_sr:>14.1%} {sign}{delta:>7.1%}")
            n_diff += 1
    if n_diff == 0:
        print(f"  (no significant differences found)")


# ═════════════════════════════════════════════════════════════════════════════
# Find latest experiment directory for --resume
# ═════════════════════════════════════════════════════════════════════════════

def find_latest_run(base_dir: str) -> str:
    """Find the most recent experiment directory under base_dir."""
    if not os.path.isdir(base_dir):
        return None
    dirs = [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None
    dirs.sort(reverse=True)
    return os.path.join(base_dir, dirs[0])


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full-scale experiment: ReAct & Reflexion on 67 goals × N runs"
    )
    parser.add_argument(
        "--agent", choices=["react", "reflexion", "both"], default="both",
        help="Which agent(s) to run (default: both)"
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help="Number of runs per goal (default: from config, typically 15)"
    )
    parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Only run these groups (e.g. --groups Wooden Stone Iron)"
    )
    parser.add_argument(
        "--goals", nargs="+", default=None,
        help="Only run these specific goals (e.g. --goals stick iron_pickaxe)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume the most recent interrupted experiment"
    )
    parser.add_argument(
        "--tag", default=None,
        help="Experiment tag for output directory (default: timestamp)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step verbose output (only show per-episode summary)"
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Don't run experiments, just regenerate summary from existing results"
    )
    args = parser.parse_args()

    # ── Load goals ───────────────────────────────────────────────────────
    goals = load_goals(groups_filter=args.groups, goals_filter=args.goals)
    if not goals:
        print("No goals matched the filters. Check --groups or --goals.")
        return

    # ── Determine num_runs ───────────────────────────────────────────────
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    num_runs = args.runs or config.get("env", {}).get("num_runs", 15)

    # ── Output directory ─────────────────────────────────────────────────
    if args.resume:
        out_dir = find_latest_run(RESULTS_BASE)
        if not out_dir:
            print("No previous experiment found to resume.")
            return
        print(f"Resuming from: {out_dir}")
    else:
        tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(RESULTS_BASE, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Save experiment config
    exp_config = {
        "timestamp": datetime.now().isoformat(),
        "agent": args.agent,
        "num_runs": num_runs,
        "num_goals": len(goals),
        "goals": [g["goal"] for g in goals],
        "groups": list(set(g["group"] for g in goals)),
        "config_path": CONFIG_PATH,
        "llm_config": config.get("llm", {}),
    }
    config_out = os.path.join(out_dir, "experiment_config.json")
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2, ensure_ascii=False)

    # ── Print banner ─────────────────────────────────────────────────────
    agents_str = args.agent if args.agent != "both" else "react + reflexion"
    groups_str = ", ".join(sorted(set(g["group"] for g in goals)))

    print(f"\n{'#'*70}")
    print(f"  FULL EXPERIMENT")
    print(f"  Agent(s):  {agents_str}")
    print(f"  Goals:     {len(goals)} ({groups_str})")
    print(f"  Runs/goal: {num_runs}")
    print(f"  Total episodes: {len(goals) * num_runs} per agent")
    print(f"  Output:    {out_dir}")
    print(f"  Resume:    {args.resume}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    verbose = not args.quiet

    # ── Run experiments ──────────────────────────────────────────────────
    if not args.summary_only:
        if args.agent in ("react", "both"):
            run_agent("react", goals, num_runs, out_dir,
                      resume=args.resume, verbose=verbose)

        if args.agent in ("reflexion", "both"):
            run_agent("reflexion", goals, num_runs, out_dir,
                      resume=args.resume, verbose=verbose)

    # ── Aggregate and print summary ──────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  GENERATING SUMMARY")
    print(f"{'#'*70}")

    react_stats = None
    refl_stats = None

    if args.agent in ("react", "both"):
        react_stats = aggregate_results(out_dir, "react", goals, num_runs)
        print_agent_summary(react_stats)
        summary_path = os.path.join(out_dir, "react_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(react_stats, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {summary_path}")

    if args.agent in ("reflexion", "both"):
        refl_stats = aggregate_results(out_dir, "reflexion", goals, num_runs)
        print_agent_summary(refl_stats)
        summary_path = os.path.join(out_dir, "reflexion_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(refl_stats, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {summary_path}")

    if react_stats and refl_stats:
        print_comparison(react_stats, refl_stats)

    # Save combined summary
    combined = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": out_dir,
        "num_goals": len(goals),
        "num_runs": num_runs,
    }
    if react_stats:
        combined["react"] = {
            "overall_success_rate": react_stats["overall_success_rate"],
            "group_stats": react_stats["group_stats"],
        }
    if refl_stats:
        combined["reflexion"] = {
            "overall_success_rate": refl_stats["overall_success_rate"],
            "group_stats": refl_stats["group_stats"],
        }
    combined_path = os.path.join(out_dir, "combined_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n\n  All results saved to: {out_dir}")
    print(f"  Combined summary:    {combined_path}")
    print(f"\n{'═'*70}\n")


if __name__ == "__main__":
    main()

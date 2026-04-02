"""
Reflexion Results Analysis — Full experiment on 67 goals.

Produces:
  1. Overall success rate
  2. Per-group success rate table + bar chart
  3. Per-goal success rate table (sorted by difficulty)
  4. Final error type distribution (failed runs only)
  5. Final termination reason distribution
  6. Winning-attempt distribution
  7. Step efficiency analysis (total steps across attempts)
  8. Performance decay by group complexity
  9. Detailed failure case studies

Usage:
  cd ns-fsm-baseline

  # Analyze from results directory
  python scripts/analyze_reflexion.py --tag full_v1

  # Or specify path directly
  python scripts/analyze_reflexion.py --path results/full/full_v1/reflexion

  # Save figures
  python scripts/analyze_reflexion.py --tag full_v1 --save-figs

  # Export CSV tables
  python scripts/analyze_reflexion.py --tag full_v1 --export-csv
"""

import sys
import os
import json
import argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("[WARN] matplotlib not installed — skipping figures")


GROUP_ORDER = ["Wooden", "Stone", "Iron", "Golden", "Diamond", "Redstone", "Armor"]
GROUP_DEPTH = {
    "Wooden": 3,
    "Stone": 7,
    "Iron": 12,
    "Golden": 15,
    "Diamond": 18,
    "Redstone": 14,
    "Armor": 15,
}


def load_reflexion_results(reflexion_dir: str) -> list:
    """Load all reflexion result JSONs from a directory."""
    results = []
    if not os.path.isdir(reflexion_dir):
        print(f"ERROR: Directory not found: {reflexion_dir}")
        sys.exit(1)

    for fname in sorted(os.listdir(reflexion_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(reflexion_dir, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results.append(data)

    print(f"Loaded {len(results)} result files from {reflexion_dir}")
    return results


def organize_by_goal(results: list) -> dict:
    by_goal = defaultdict(list)
    for r in results:
        by_goal[r.get("goal", "unknown")].append(r)
    return dict(by_goal)


def organize_by_group(results: list) -> dict:
    by_group = defaultdict(list)
    for r in results:
        by_group[r.get("group", "Unknown")].append(r)
    return dict(by_group)


def get_attempts(run: dict) -> list:
    return run.get("attempts", [])


def get_total_attempts(run: dict) -> int:
    attempts = get_attempts(run)
    if "total_attempts" in run:
        return run.get("total_attempts", 0)
    return len(attempts) if attempts else 1


def get_winning_attempt(run: dict):
    return run.get("winning_attempt")


def get_final_attempt_result(run: dict) -> dict:
    attempts = get_attempts(run)
    if attempts:
        return attempts[-1].get("result", {})
    return {
        "goal": run.get("goal"),
        "success": run.get("success", False),
        "total_steps": run.get("total_steps", 0),
        "termination": run.get("termination", "unknown"),
        "trajectory": run.get("trajectory", []),
        "errors": run.get("final_errors", run.get("errors", {})),
    }


def get_success_attempt_result(run: dict) -> dict:
    attempts = get_attempts(run)
    winning_attempt = get_winning_attempt(run)
    if attempts and winning_attempt and 1 <= winning_attempt <= len(attempts):
        return attempts[winning_attempt - 1].get("result", {})
    return get_final_attempt_result(run)


def get_total_steps(run: dict) -> int:
    attempts = get_attempts(run)
    if attempts:
        return sum(att.get("result", {}).get("total_steps", 0) for att in attempts)
    return run.get("total_steps", 0)


def get_success_steps(run: dict) -> int:
    attempts = get_attempts(run)
    winning_attempt = get_winning_attempt(run)
    if attempts and winning_attempt and 1 <= winning_attempt <= len(attempts):
        return sum(
            attempts[idx].get("result", {}).get("total_steps", 0)
            for idx in range(winning_attempt)
        )
    return get_total_steps(run)


def get_final_termination(run: dict) -> str:
    return get_final_attempt_result(run).get("termination", "unknown")


def get_final_error_type(run: dict) -> str:
    final_errors = run.get("final_errors")
    if isinstance(final_errors, dict):
        return final_errors.get("dominant_error_type", "unknown")

    final_result = get_final_attempt_result(run)
    errors = final_result.get("errors", {})
    if isinstance(errors, dict):
        return errors.get("dominant_error_type", "unknown")
    return "unknown"


def compute_overall_stats(results: list) -> dict:
    n = len(results)
    n_success = sum(1 for r in results if r.get("success"))

    steps_success = [get_success_steps(r) for r in results if r.get("success")]
    steps_fail = [get_total_steps(r) for r in results if not r.get("success")]
    steps_all = [
        get_success_steps(r) if r.get("success") else get_total_steps(r)
        for r in results
    ]

    attempts_success = [get_total_attempts(r) for r in results if r.get("success")]
    attempts_fail = [get_total_attempts(r) for r in results if not r.get("success")]
    attempts_all = [get_total_attempts(r) for r in results]
    winning_attempts = [get_winning_attempt(r) for r in results if r.get("success")]

    return {
        "total_runs": n,
        "n_success": n_success,
        "n_fail": n - n_success,
        "success_rate": n_success / n if n > 0 else 0,
        "avg_attempts_success": (
            sum(attempts_success) / len(attempts_success) if attempts_success else 0
        ),
        "avg_attempts_fail": (
            sum(attempts_fail) / len(attempts_fail) if attempts_fail else 0
        ),
        "avg_attempts_all": sum(attempts_all) / n if n > 0 else 0,
        "avg_winning_attempt": (
            sum(winning_attempts) / len(winning_attempts) if winning_attempts else 0
        ),
        "avg_steps_success": (
            sum(steps_success) / len(steps_success) if steps_success else 0
        ),
        "avg_steps_fail": sum(steps_fail) / len(steps_fail) if steps_fail else 0,
        "avg_steps_all": sum(steps_all) / n if n > 0 else 0,
    }


def compute_group_stats(results: list) -> list:
    by_group = organize_by_group(results)
    stats = []
    for group in GROUP_ORDER:
        if group not in by_group:
            continue
        runs = by_group[group]
        n = len(runs)
        n_success = sum(1 for r in runs if r.get("success"))
        n_goals = len(set(r.get("goal") for r in runs))
        avg_steps = sum(
            get_success_steps(r) if r.get("success") else get_total_steps(r)
            for r in runs
        ) / n if n > 0 else 0
        avg_attempts = sum(get_total_attempts(r) for r in runs) / n if n > 0 else 0

        stats.append({
            "group": group,
            "n_goals": n_goals,
            "n_runs": n,
            "n_success": n_success,
            "success_rate": n_success / n if n > 0 else 0,
            "avg_attempts": avg_attempts,
            "avg_steps": avg_steps,
            "depth": GROUP_DEPTH.get(group, 0),
        })
    return stats


def compute_goal_stats(results: list) -> list:
    by_goal = organize_by_goal(results)
    stats = []
    for goal, runs in by_goal.items():
        n = len(runs)
        n_success = sum(1 for r in runs if r.get("success"))
        group = runs[0].get("group", "Unknown")
        avg_steps = sum(
            get_success_steps(r) if r.get("success") else get_total_steps(r)
            for r in runs
        ) / n if n > 0 else 0
        steps_success = [get_success_steps(r) for r in runs if r.get("success")]
        avg_steps_success = (
            sum(steps_success) / len(steps_success) if steps_success else 0
        )
        avg_attempts = sum(get_total_attempts(r) for r in runs) / n if n > 0 else 0

        error_dist = defaultdict(int)
        for r in runs:
            if not r.get("success"):
                error_dist[get_final_error_type(r)] += 1

        term_dist = defaultdict(int)
        for r in runs:
            term_dist[get_final_termination(r)] += 1

        stats.append({
            "goal": goal,
            "group": group,
            "n_runs": n,
            "n_success": n_success,
            "success_rate": n_success / n if n > 0 else 0,
            "avg_attempts": round(avg_attempts, 2),
            "avg_steps": round(avg_steps, 1),
            "avg_steps_success": round(avg_steps_success, 1),
            "error_distribution": dict(error_dist),
            "termination_distribution": dict(term_dist),
        })

    group_idx = {g: i for i, g in enumerate(GROUP_ORDER)}
    stats.sort(key=lambda x: (group_idx.get(x["group"], 99), -x["success_rate"]))
    return stats


def compute_error_distribution(results: list) -> dict:
    dist = defaultdict(int)
    for r in results:
        if not r.get("success"):
            dist[get_final_error_type(r)] += 1
    return dict(dist)


def compute_termination_distribution(results: list) -> dict:
    dist = defaultdict(int)
    for r in results:
        dist[get_final_termination(r)] += 1
    return dict(dist)


def compute_winning_attempt_distribution(results: list) -> dict:
    dist = defaultdict(int)
    for r in results:
        if r.get("success"):
            dist[f"attempt_{get_winning_attempt(r)}"] += 1
        else:
            dist["failed_all_attempts"] += 1
    return dict(dist)


def print_overall(stats: dict):
    print(f"\n{'═'*60}")
    print(f"  REFLEXION — OVERALL RESULTS")
    print(f"{'═'*60}")
    print(f"  Total runs:         {stats['total_runs']}")
    print(f"  Success:            {stats['n_success']} ({stats['success_rate']:.1%})")
    print(f"  Failure:            {stats['n_fail']} ({1-stats['success_rate']:.1%})")
    print(f"  Avg attempts (OK):  {stats['avg_attempts_success']:.2f}")
    print(f"  Avg attempts (FAIL):{stats['avg_attempts_fail']:.2f}")
    print(f"  Avg attempts (all): {stats['avg_attempts_all']:.2f}")
    print(f"  Avg win attempt:    {stats['avg_winning_attempt']:.2f}")
    print(f"  Avg steps (success):{stats['avg_steps_success']:.1f}")
    print(f"  Avg steps (failure):{stats['avg_steps_fail']:.1f}")
    print(f"  Avg steps (all):    {stats['avg_steps_all']:.1f}")


def print_group_table(group_stats: list):
    print(f"\n{'═'*74}")
    print(f"  SUCCESS RATE BY GROUP")
    print(f"{'═'*74}")
    print(f"  {'Group':<12} {'Depth':>5} {'Goals':>5} {'Runs':>5} {'OK':>4} "
          f"{'SR':>7} {'AvgAtt':>7} {'AvgStep':>8}")
    print(f"  {'─'*12} {'─'*5} {'─'*5} {'─'*5} {'─'*4} {'─'*7} {'─'*7} {'─'*8}")
    for gs in group_stats:
        bar = "█" * int(gs["success_rate"] * 10) + "░" * (10 - int(gs["success_rate"] * 10))
        print(
            f"  {gs['group']:<12} {gs['depth']:>5} {gs['n_goals']:>5} {gs['n_runs']:>5} "
            f"{gs['n_success']:>4} {gs['success_rate']:>6.1%} {gs['avg_attempts']:>7.2f} "
            f"{gs['avg_steps']:>8.1f}  {bar}"
        )


def print_goal_table(goal_stats: list):
    print(f"\n{'═'*80}")
    print(f"  SUCCESS RATE BY GOAL")
    print(f"{'═'*80}")
    print(f"  {'Goal':<24} {'Group':<10} {'Runs':>4} {'SR':>8} "
          f"{'AvgAtt':>7} {'AvgStep':>7} {'TopError':<18}")
    print(f"  {'─'*24} {'─'*10} {'─'*4} {'─'*8} {'─'*7} {'─'*7} {'─'*18}")

    current_group = None
    for gs in goal_stats:
        if gs["group"] != current_group:
            current_group = gs["group"]
            if gs != goal_stats[0]:
                print(f"  {'':24} {'─'*10}")

        sr_str = f"{gs['n_success']}/{gs['n_runs']}"
        top_err = max(gs["error_distribution"], key=gs["error_distribution"].get) \
            if gs["error_distribution"] else "-"
        print(
            f"  {gs['goal']:<24} {gs['group']:<10} {gs['n_runs']:>4} {sr_str:>8} "
            f"{gs['avg_attempts']:>7.2f} {gs['avg_steps']:>7.1f} {top_err:<18}"
        )


def print_error_distribution(error_dist: dict, total_fail: int):
    print(f"\n{'═'*60}")
    print(f"  FINAL ERROR TYPE DISTRIBUTION (failed runs only)")
    print(f"{'═'*60}")
    if not error_dist:
        print("  (no failures)")
        return
    for err, cnt in sorted(error_dist.items(), key=lambda x: -x[1]):
        pct = cnt / total_fail if total_fail > 0 else 0
        bar = "█" * int(pct * 30)
        print(f"  {err:<28} {cnt:>4} ({pct:>5.1%})  {bar}")


def print_termination_distribution(term_dist: dict, total: int):
    print(f"\n{'═'*60}")
    print(f"  FINAL TERMINATION REASONS")
    print(f"{'═'*60}")
    for term, cnt in sorted(term_dist.items(), key=lambda x: -x[1]):
        pct = cnt / total if total > 0 else 0
        print(f"  {term:<20} {cnt:>5} ({pct:>5.1%})")


def print_winning_attempt_distribution(win_dist: dict, total: int):
    print(f"\n{'═'*60}")
    print(f"  WINNING ATTEMPT DISTRIBUTION")
    print(f"{'═'*60}")
    for attempt, cnt in sorted(win_dist.items()):
        pct = cnt / total if total > 0 else 0
        print(f"  {attempt:<20} {cnt:>5} ({pct:>5.1%})")


def print_failure_case_studies(goal_stats: list, n_show: int = 5):
    print(f"\n{'═'*60}")
    print(f"  FAILURE CASE STUDIES (worst {n_show} goals)")
    print(f"{'═'*60}")

    worst = sorted(goal_stats, key=lambda x: x["success_rate"])[:n_show]
    for gs in worst:
        print(f"\n  {gs['goal']} ({gs['group']}) — SR: {gs['success_rate']:.0%}")
        print(f"    Avg attempts: {gs['avg_attempts']}")
        print(f"    Avg steps: {gs['avg_steps']}")

        if gs["error_distribution"]:
            err_parts = [f"{e}: {c}" for e, c in
                         sorted(gs["error_distribution"].items(), key=lambda x: -x[1])]
            print(f"    Errors: {', '.join(err_parts)}")

        if gs["termination_distribution"]:
            term_parts = [f"{t}: {c}" for t, c in
                          sorted(gs["termination_distribution"].items(), key=lambda x: -x[1])]
            print(f"    Final termination: {', '.join(term_parts)}")


def plot_group_success_rate(group_stats: list, save_path: str = None):
    if not HAS_PLT:
        return

    groups = [gs["group"] for gs in group_stats]
    rates = [gs["success_rate"] * 100 for gs in group_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn([r / 100 for r in rates])
    bars = ax.bar(groups, rates, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Reflexion Success Rate by Group (Increasing Complexity →)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_error_pie(error_dist: dict, save_path: str = None):
    if not HAS_PLT or not error_dist:
        return

    labels = list(error_dist.keys())
    sizes = list(error_dist.values())
    color_map = {
        "plan_knowledge_error": "#e74c3c",
        "tool_dependency_error": "#e67e22",
        "resource_shortage": "#f1c40f",
        "dead_loop": "#9b59b6",
        "sequencing_error": "#3498db",
        "unknown": "#95a5a6",
        "none": "#2ecc71",
    }
    colors = [color_map.get(l, "#95a5a6") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.85, textprops={"fontsize": 9}
    )
    ax.set_title("Reflexion Final Error Distribution (Failed Runs)", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_performance_decay(group_stats: list, save_path: str = None):
    if not HAS_PLT:
        return

    groups = [gs["group"] for gs in group_stats]
    depths = [gs["depth"] for gs in group_stats]
    rates = [gs["success_rate"] * 100 for gs in group_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(depths, rates, "o-", color="#e74c3c", linewidth=2, markersize=8)

    for depth, rate, group in zip(depths, rates, groups):
        ax.annotate(f"{group}\n({rate:.0f}%)", (depth, rate),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9)

    ax.set_xlabel("Approximate Dependency Depth", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Reflexion Performance Decay vs Task Complexity", fontsize=13)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_steps_distribution(results: list, save_path: str = None):
    if not HAS_PLT:
        return

    steps_ok = [get_success_steps(r) for r in results if r.get("success")]
    steps_fail = [get_total_steps(r) for r in results if not r.get("success")]

    fig, ax = plt.subplots(figsize=(10, 5))
    if steps_ok:
        ax.hist(steps_ok, bins=20, alpha=0.7, label=f"Success (n={len(steps_ok)})",
                color="#2ecc71", edgecolor="black", linewidth=0.5)
    if steps_fail:
        ax.hist(steps_fail, bins=20, alpha=0.7, label=f"Failure (n={len(steps_fail)})",
                color="#e74c3c", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Total Steps Across Attempts", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Reflexion Step Distribution: Success vs Failure", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_goal_heatmap(goal_stats: list, save_path: str = None):
    if not HAS_PLT:
        return

    goals = [gs["goal"] for gs in goal_stats]
    rates = [gs["success_rate"] * 100 for gs in goal_stats]
    groups = [gs["group"] for gs in goal_stats]

    fig, ax = plt.subplots(figsize=(14, max(8, len(goals) * 0.25)))
    colors = plt.cm.RdYlGn([r / 100 for r in rates])
    y_pos = range(len(goals))
    ax.barh(y_pos, rates, color=colors, edgecolor="black", linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{goal} ({group})" for goal, group in zip(goals, groups)], fontsize=7)
    ax.set_xlabel("Success Rate (%)", fontsize=12)
    ax.set_title("Reflexion Success Rate — All 67 Goals", fontsize=13)
    ax.set_xlim(0, 105)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_attempt_distribution(win_dist: dict, save_path: str = None):
    if not HAS_PLT or not win_dist:
        return

    labels = list(win_dist.keys())
    values = list(win_dist.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Reflexion Winning Attempt Distribution", fontsize=13)
    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def export_csv(goal_stats: list, group_stats: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    goal_csv = os.path.join(out_dir, "reflexion_goal_stats.csv")
    with open(goal_csv, "w", encoding="utf-8") as f:
        f.write("goal,group,n_runs,n_success,success_rate,avg_attempts,avg_steps,avg_steps_success,top_error\n")
        for gs in goal_stats:
            top_err = max(gs["error_distribution"], key=gs["error_distribution"].get) \
                if gs["error_distribution"] else "none"
            f.write(
                f"{gs['goal']},{gs['group']},{gs['n_runs']},{gs['n_success']},"
                f"{gs['success_rate']:.3f},{gs['avg_attempts']},{gs['avg_steps']},"
                f"{gs['avg_steps_success']},{top_err}\n"
            )
    print(f"  Saved: {goal_csv}")

    group_csv = os.path.join(out_dir, "reflexion_group_stats.csv")
    with open(group_csv, "w", encoding="utf-8") as f:
        f.write("group,depth,n_goals,n_runs,n_success,success_rate,avg_attempts,avg_steps\n")
        for gs in group_stats:
            f.write(
                f"{gs['group']},{gs['depth']},{gs['n_goals']},{gs['n_runs']},"
                f"{gs['n_success']},{gs['success_rate']:.3f},{gs['avg_attempts']:.2f},"
                f"{gs['avg_steps']:.1f}\n"
            )
    print(f"  Saved: {group_csv}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Reflexion experiment results")
    parser.add_argument("--tag", default="full_v1",
                        help="Experiment tag (default: full_v1)")
    parser.add_argument("--path", default=None,
                        help="Direct path to reflexion results directory")
    parser.add_argument("--save-figs", action="store_true",
                        help="Save figures to results/analysis/")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export CSV tables")
    args = parser.parse_args()

    if args.path:
        reflexion_dir = args.path
    else:
        reflexion_dir = os.path.join(ROOT, "results", "full", args.tag, "reflexion")

    results = load_reflexion_results(reflexion_dir)
    if not results:
        print("No results found.")
        return

    overall = compute_overall_stats(results)
    group_stats = compute_group_stats(results)
    goal_stats = compute_goal_stats(results)
    error_dist = compute_error_distribution(results)
    term_dist = compute_termination_distribution(results)
    winning_attempt_dist = compute_winning_attempt_distribution(results)

    print_overall(overall)
    print_group_table(group_stats)
    print_goal_table(goal_stats)
    print_error_distribution(error_dist, overall["n_fail"])
    print_termination_distribution(term_dist, overall["total_runs"])
    print_winning_attempt_distribution(winning_attempt_dist, overall["total_runs"])
    print_failure_case_studies(goal_stats)

    fig_dir = None
    if args.save_figs:
        fig_dir = os.path.join(ROOT, "results", "analysis", "reflexion")
        os.makedirs(fig_dir, exist_ok=True)

    plot_group_success_rate(
        group_stats,
        save_path=os.path.join(fig_dir, "group_success_rate.png") if fig_dir else None
    )
    plot_error_pie(
        error_dist,
        save_path=os.path.join(fig_dir, "error_distribution.png") if fig_dir else None
    )
    plot_performance_decay(
        group_stats,
        save_path=os.path.join(fig_dir, "performance_decay.png") if fig_dir else None
    )
    plot_steps_distribution(
        results,
        save_path=os.path.join(fig_dir, "steps_distribution.png") if fig_dir else None
    )
    plot_goal_heatmap(
        goal_stats,
        save_path=os.path.join(fig_dir, "goal_heatmap.png") if fig_dir else None
    )
    plot_attempt_distribution(
        winning_attempt_dist,
        save_path=os.path.join(fig_dir, "attempt_distribution.png") if fig_dir else None
    )

    if fig_dir:
        print(f"\n  All figures saved to: {fig_dir}")

    if args.export_csv:
        csv_dir = os.path.join(ROOT, "results", "analysis", "reflexion")
        export_csv(goal_stats, group_stats, csv_dir)

    analysis_out = os.path.join(
        ROOT, "results", "full", args.tag if not args.path else "custom",
        "reflexion_analysis.json"
    )
    os.makedirs(os.path.dirname(analysis_out), exist_ok=True)
    analysis = {
        "overall": overall,
        "group_stats": group_stats,
        "goal_stats": goal_stats,
        "error_distribution": error_dist,
        "termination_distribution": term_dist,
        "winning_attempt_distribution": winning_attempt_dist,
    }
    with open(analysis_out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n  Analysis JSON: {analysis_out}")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()

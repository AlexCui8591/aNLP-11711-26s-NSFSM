"""
ReAct Results Analysis — Full experiment on 67 goals.

Produces:
  1. Overall success rate
  2. Per-group success rate table + bar chart
  3. Per-goal success rate table (sorted by difficulty)
  4. Error type distribution (pie chart + table)
  5. Termination reason distribution
  6. Step efficiency analysis (avg steps for success vs failure)
  7. Performance decay by group complexity
  8. Detailed failure case studies

Usage:
  cd ns-fsm-baseline

  # Analyze from results directory
  python scripts/analyze_react.py --tag full_v1

  # Or specify path directly
  python scripts/analyze_react.py --path results/full/full_v1/react

  # Save figures
  python scripts/analyze_react.py --tag full_v1 --save-figs

  # Export CSV tables
  python scripts/analyze_react.py --tag full_v1 --export-csv
"""

import sys
import os
import json
import argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Try importing matplotlib (optional — tables still work without it)
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for cluster
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("[WARN] matplotlib not installed — skipping figures")


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

# Group ordering by complexity (used for consistent display)
GROUP_ORDER = ["Wooden", "Stone", "Iron", "Golden", "Diamond", "Redstone", "Armor"]

# Approximate dependency depth per group (for decay analysis)
GROUP_DEPTH = {
    "Wooden":   3,
    "Stone":    7,
    "Iron":    12,
    "Golden":  15,
    "Diamond": 18,
    "Redstone":14,
    "Armor":   15,
}


def load_react_results(react_dir: str) -> list:
    """Load all react result JSONs from a directory."""
    results = []
    if not os.path.isdir(react_dir):
        print(f"ERROR: Directory not found: {react_dir}")
        sys.exit(1)

    for fname in sorted(os.listdir(react_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(react_dir, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results.append(data)

    print(f"Loaded {len(results)} result files from {react_dir}")
    return results


def organize_by_goal(results: list) -> dict:
    """
    Group results by goal name.
    Returns: {goal: [result1, result2, ...]}
    """
    by_goal = defaultdict(list)
    for r in results:
        by_goal[r["goal"]].append(r)
    return dict(by_goal)


def organize_by_group(results: list) -> dict:
    """
    Group results by group name.
    Returns: {group: [result1, result2, ...]}
    """
    by_group = defaultdict(list)
    for r in results:
        group = r.get("group", "Unknown")
        by_group[group].append(r)
    return dict(by_group)


# ═════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═════════════════════════════════════════════════════════════════════════════

def compute_overall_stats(results: list) -> dict:
    """Overall success rate and step statistics."""
    n = len(results)
    n_success = sum(1 for r in results if r.get("success"))
    steps_success = [r["total_steps"] for r in results if r.get("success")]
    steps_fail = [r["total_steps"] for r in results if not r.get("success")]

    return {
        "total_episodes": n,
        "n_success": n_success,
        "n_fail": n - n_success,
        "success_rate": n_success / n if n > 0 else 0,
        "avg_steps_success": sum(steps_success) / len(steps_success) if steps_success else 0,
        "avg_steps_fail": sum(steps_fail) / len(steps_fail) if steps_fail else 0,
        "avg_steps_all": sum(r["total_steps"] for r in results) / n if n > 0 else 0,
    }


def compute_group_stats(results: list) -> list:
    """Per-group success rate and stats."""
    by_group = organize_by_group(results)
    stats = []
    for group in GROUP_ORDER:
        if group not in by_group:
            continue
        runs = by_group[group]
        n = len(runs)
        n_success = sum(1 for r in runs if r.get("success"))
        n_goals = len(set(r["goal"] for r in runs))
        avg_steps = sum(r["total_steps"] for r in runs) / n if n > 0 else 0

        stats.append({
            "group": group,
            "n_goals": n_goals,
            "n_runs": n,
            "n_success": n_success,
            "success_rate": n_success / n if n > 0 else 0,
            "avg_steps": avg_steps,
            "depth": GROUP_DEPTH.get(group, 0),
        })
    return stats


def compute_goal_stats(results: list) -> list:
    """Per-goal success rate."""
    by_goal = organize_by_goal(results)
    stats = []
    for goal, runs in by_goal.items():
        n = len(runs)
        n_success = sum(1 for r in runs if r.get("success"))
        group = runs[0].get("group", "Unknown")
        avg_steps = sum(r["total_steps"] for r in runs) / n if n > 0 else 0
        steps_success = [r["total_steps"] for r in runs if r.get("success")]
        avg_steps_success = sum(steps_success) / len(steps_success) if steps_success else 0

        # Collect error types from failures
        error_dist = defaultdict(int)
        for r in runs:
            if not r.get("success"):
                err = r.get("errors", {}).get("dominant_error_type", "unknown")
                error_dist[err] += 1

        # Collect termination reasons
        term_dist = defaultdict(int)
        for r in runs:
            term_dist[r.get("termination", "unknown")] += 1

        stats.append({
            "goal": goal,
            "group": group,
            "n_runs": n,
            "n_success": n_success,
            "success_rate": n_success / n if n > 0 else 0,
            "avg_steps": round(avg_steps, 1),
            "avg_steps_success": round(avg_steps_success, 1),
            "error_distribution": dict(error_dist),
            "termination_distribution": dict(term_dist),
        })

    # Sort by group order then success rate
    group_idx = {g: i for i, g in enumerate(GROUP_ORDER)}
    stats.sort(key=lambda x: (group_idx.get(x["group"], 99), -x["success_rate"]))
    return stats


def compute_error_distribution(results: list) -> dict:
    """Error type distribution across all failed episodes."""
    dist = defaultdict(int)
    for r in results:
        if not r.get("success"):
            err = r.get("errors", {}).get("dominant_error_type", "unknown")
            dist[err] += 1
    return dict(dist)


def compute_termination_distribution(results: list) -> dict:
    """Termination reason distribution."""
    dist = defaultdict(int)
    for r in results:
        dist[r.get("termination", "unknown")] += 1
    return dict(dist)


# ═════════════════════════════════════════════════════════════════════════════
# Print tables
# ═════════════════════════════════════════════════════════════════════════════

def print_overall(stats: dict):
    print(f"\n{'═'*60}")
    print(f"  REACT — OVERALL RESULTS")
    print(f"{'═'*60}")
    print(f"  Total episodes:     {stats['total_episodes']}")
    print(f"  Success:            {stats['n_success']} ({stats['success_rate']:.1%})")
    print(f"  Failure:            {stats['n_fail']} ({1-stats['success_rate']:.1%})")
    print(f"  Avg steps (success):{stats['avg_steps_success']:.1f}")
    print(f"  Avg steps (failure):{stats['avg_steps_fail']:.1f}")
    print(f"  Avg steps (all):    {stats['avg_steps_all']:.1f}")


def print_group_table(group_stats: list):
    print(f"\n{'═'*60}")
    print(f"  SUCCESS RATE BY GROUP")
    print(f"{'═'*60}")
    print(f"  {'Group':<12} {'Depth':>5} {'Goals':>5} {'Runs':>5} "
          f"{'OK':>4} {'SR':>7} {'AvgSteps':>8}")
    print(f"  {'─'*12} {'─'*5} {'─'*5} {'─'*5} {'─'*4} {'─'*7} {'─'*8}")
    for gs in group_stats:
        bar = "█" * int(gs["success_rate"] * 10) + "░" * (10 - int(gs["success_rate"] * 10))
        print(f"  {gs['group']:<12} {gs['depth']:>5} {gs['n_goals']:>5} {gs['n_runs']:>5} "
              f"{gs['n_success']:>4} {gs['success_rate']:>6.1%} {gs['avg_steps']:>8.1f}  {bar}")


def print_goal_table(goal_stats: list):
    print(f"\n{'═'*70}")
    print(f"  SUCCESS RATE BY GOAL")
    print(f"{'═'*70}")
    print(f"  {'Goal':<24} {'Group':<10} {'Runs':>4} {'SR':>8} "
          f"{'AvgStep':>7} {'TopError':<22}")
    print(f"  {'─'*24} {'─'*10} {'─'*4} {'─'*8} {'─'*7} {'─'*22}")

    current_group = None
    for gs in goal_stats:
        if gs["group"] != current_group:
            current_group = gs["group"]
            if gs != goal_stats[0]:
                print(f"  {'':24} {'─'*10}")

        sr_str = f"{gs['n_success']}/{gs['n_runs']}"
        top_err = max(gs["error_distribution"], key=gs["error_distribution"].get) \
            if gs["error_distribution"] else "-"
        print(f"  {gs['goal']:<24} {gs['group']:<10} {gs['n_runs']:>4} {sr_str:>8} "
              f"{gs['avg_steps']:>7.1f} {top_err:<22}")


def print_error_distribution(error_dist: dict, total_fail: int):
    print(f"\n{'═'*60}")
    print(f"  ERROR TYPE DISTRIBUTION (failed episodes only)")
    print(f"{'═'*60}")
    if not error_dist:
        print(f"  (no failures)")
        return
    for err, cnt in sorted(error_dist.items(), key=lambda x: -x[1]):
        pct = cnt / total_fail if total_fail > 0 else 0
        bar = "█" * int(pct * 30)
        print(f"  {err:<28} {cnt:>4} ({pct:>5.1%})  {bar}")


def print_termination_distribution(term_dist: dict, total: int):
    print(f"\n{'═'*60}")
    print(f"  TERMINATION REASONS")
    print(f"{'═'*60}")
    for term, cnt in sorted(term_dist.items(), key=lambda x: -x[1]):
        pct = cnt / total if total > 0 else 0
        print(f"  {term:<20} {cnt:>5} ({pct:>5.1%})")


def print_failure_case_studies(goal_stats: list, n_show: int = 5):
    """Show the hardest goals (lowest success rate) with error details."""
    print(f"\n{'═'*60}")
    print(f"  FAILURE CASE STUDIES (worst {n_show} goals)")
    print(f"{'═'*60}")

    # Sort by success rate ascending
    worst = sorted(goal_stats, key=lambda x: x["success_rate"])[:n_show]
    for gs in worst:
        print(f"\n  {gs['goal']} ({gs['group']}) — SR: {gs['success_rate']:.0%}")
        print(f"    Avg steps: {gs['avg_steps']}")

        if gs["error_distribution"]:
            err_parts = [f"{e}: {c}" for e, c in
                         sorted(gs["error_distribution"].items(), key=lambda x: -x[1])]
            print(f"    Errors: {', '.join(err_parts)}")

        if gs["termination_distribution"]:
            term_parts = [f"{t}: {c}" for t, c in
                          sorted(gs["termination_distribution"].items(), key=lambda x: -x[1])]
            print(f"    Termination: {', '.join(term_parts)}")


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════

def plot_group_success_rate(group_stats: list, save_path: str = None):
    """Bar chart of success rate by group."""
    if not HAS_PLT:
        return

    groups = [gs["group"] for gs in group_stats]
    rates = [gs["success_rate"] * 100 for gs in group_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn([r / 100 for r in rates])
    bars = ax.bar(groups, rates, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("ReAct Success Rate by Group (Increasing Complexity →)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

    # Add value labels on bars
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
    """Pie chart of error types."""
    if not HAS_PLT or not error_dist:
        return

    labels = list(error_dist.keys())
    sizes = list(error_dist.values())

    # Color mapping
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
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.85,
        textprops={"fontsize": 9}
    )
    ax.set_title("ReAct Error Type Distribution (Failed Episodes)", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_performance_decay(group_stats: list, save_path: str = None):
    """Line chart showing success rate declining with dependency depth."""
    if not HAS_PLT:
        return

    groups = [gs["group"] for gs in group_stats]
    depths = [gs["depth"] for gs in group_stats]
    rates = [gs["success_rate"] * 100 for gs in group_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(depths, rates, "o-", color="#e74c3c", linewidth=2, markersize=8)

    for i, (d, r, g) in enumerate(zip(depths, rates, groups)):
        ax.annotate(f"{g}\n({r:.0f}%)", (d, r),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9)

    ax.set_xlabel("Approximate Dependency Depth", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("ReAct Performance Decay vs Task Complexity", fontsize=13)
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
    """Histogram of steps for success vs failure."""
    if not HAS_PLT:
        return

    steps_ok = [r["total_steps"] for r in results if r.get("success")]
    steps_fail = [r["total_steps"] for r in results if not r.get("success")]

    fig, ax = plt.subplots(figsize=(10, 5))
    if steps_ok:
        ax.hist(steps_ok, bins=20, alpha=0.7, label=f"Success (n={len(steps_ok)})",
                color="#2ecc71", edgecolor="black", linewidth=0.5)
    if steps_fail:
        ax.hist(steps_fail, bins=20, alpha=0.7, label=f"Failure (n={len(steps_fail)})",
                color="#e74c3c", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Total Steps", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("ReAct Step Distribution: Success vs Failure", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_goal_heatmap(goal_stats: list, save_path: str = None):
    """Heatmap-style bar chart of all 67 goals' success rates."""
    if not HAS_PLT:
        return

    # Sort by group then success rate
    goals = [gs["goal"] for gs in goal_stats]
    rates = [gs["success_rate"] * 100 for gs in goal_stats]
    groups = [gs["group"] for gs in goal_stats]

    fig, ax = plt.subplots(figsize=(14, max(8, len(goals) * 0.25)))
    colors = plt.cm.RdYlGn([r / 100 for r in rates])

    y_pos = range(len(goals))
    bars = ax.barh(y_pos, rates, color=colors, edgecolor="black", linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{g} ({gr})" for g, gr in zip(goals, groups)], fontsize=7)
    ax.set_xlabel("Success Rate (%)", fontsize=12)
    ax.set_title("ReAct Success Rate — All 67 Goals", fontsize=13)
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


# ═════════════════════════════════════════════════════════════════════════════
# CSV export
# ═════════════════════════════════════════════════════════════════════════════

def export_csv(goal_stats: list, group_stats: list, out_dir: str):
    """Export analysis tables as CSV."""
    os.makedirs(out_dir, exist_ok=True)

    # Goal-level CSV
    goal_csv = os.path.join(out_dir, "react_goal_stats.csv")
    with open(goal_csv, "w", encoding="utf-8") as f:
        f.write("goal,group,n_runs,n_success,success_rate,avg_steps,avg_steps_success,top_error\n")
        for gs in goal_stats:
            top_err = max(gs["error_distribution"], key=gs["error_distribution"].get) \
                if gs["error_distribution"] else "none"
            f.write(f"{gs['goal']},{gs['group']},{gs['n_runs']},{gs['n_success']},"
                    f"{gs['success_rate']:.3f},{gs['avg_steps']},{gs['avg_steps_success']},"
                    f"{top_err}\n")
    print(f"  Saved: {goal_csv}")

    # Group-level CSV
    group_csv = os.path.join(out_dir, "react_group_stats.csv")
    with open(group_csv, "w", encoding="utf-8") as f:
        f.write("group,depth,n_goals,n_runs,n_success,success_rate,avg_steps\n")
        for gs in group_stats:
            f.write(f"{gs['group']},{gs['depth']},{gs['n_goals']},{gs['n_runs']},"
                    f"{gs['n_success']},{gs['success_rate']:.3f},{gs['avg_steps']:.1f}\n")
    print(f"  Saved: {group_csv}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze ReAct experiment results")
    parser.add_argument("--tag", default="full_v1",
                        help="Experiment tag (default: full_v1)")
    parser.add_argument("--path", default=None,
                        help="Direct path to react results directory")
    parser.add_argument("--save-figs", action="store_true",
                        help="Save figures to results/analysis/")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export CSV tables")
    args = parser.parse_args()

    # Resolve path
    if args.path:
        react_dir = args.path
    else:
        react_dir = os.path.join(ROOT, "results", "full", args.tag, "react")

    # Load data
    results = load_react_results(react_dir)
    if not results:
        print("No results found.")
        return

    # ── Compute stats ────────────────────────────────────────────────
    overall = compute_overall_stats(results)
    group_stats = compute_group_stats(results)
    goal_stats = compute_goal_stats(results)
    error_dist = compute_error_distribution(results)
    term_dist = compute_termination_distribution(results)

    # ── Print tables ─────────────────────────────────────────────────
    print_overall(overall)
    print_group_table(group_stats)
    print_goal_table(goal_stats)
    print_error_distribution(error_dist, overall["n_fail"])
    print_termination_distribution(term_dist, overall["total_episodes"])
    print_failure_case_studies(goal_stats)

    # ── Figures ──────────────────────────────────────────────────────
    fig_dir = None
    if args.save_figs:
        fig_dir = os.path.join(ROOT, "results", "analysis", "react")
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

    if fig_dir:
        print(f"\n  All figures saved to: {fig_dir}")

    # ── CSV export ───────────────────────────────────────────────────
    if args.export_csv:
        csv_dir = os.path.join(ROOT, "results", "analysis", "react")
        export_csv(goal_stats, group_stats, csv_dir)

    # ── Save full analysis JSON ──────────────────────────────────────
    analysis_out = os.path.join(
        ROOT, "results", "full", args.tag if not args.path else "custom",
        "react_analysis.json"
    )
    os.makedirs(os.path.dirname(analysis_out), exist_ok=True)
    analysis = {
        "overall": overall,
        "group_stats": group_stats,
        "goal_stats": goal_stats,
        "error_distribution": error_dist,
        "termination_distribution": term_dist,
    }
    with open(analysis_out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n  Analysis JSON: {analysis_out}")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()

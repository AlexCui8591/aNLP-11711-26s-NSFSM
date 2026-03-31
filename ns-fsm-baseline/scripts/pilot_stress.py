"""
Stress Pilot — ReAct vs Reflexion across difficulty tiers.

Runs 8 carefully chosen goals that test increasingly deep dependency chains,
then produces a side-by-side comparison with automatic error classification.

Goal selection rationale:
  Tier 0 (depth ~3)   stick, bowl           — baseline; bowl exposes knowledge errors
  Tier 1 (depth ~5)   wooden_pickaxe        — multi-step, resource contention
  Tier 2 (depth ~8)   furnace, torch        — needs wooden_pickaxe → cobblestone/coal
  Tier 3 (depth ~12)  iron_pickaxe          — needs stone_pickaxe → iron_ore → smelting
  Tier 4 (depth ~15)  blast_furnace         — needs smooth_stone + iron×5 + furnace

Expected outcome:
  ReAct  — succeeds on Tier 0-1, degrades on Tier 2, fails on Tier 3-4
  Reflexion — partially recovers Tier 2, may recover some Tier 3, still fails Tier 4

Usage:
  cd ns-fsm-baseline
  python scripts/pilot_stress.py               # full run (needs Ollama)
  python scripts/pilot_stress.py --react-only   # skip Reflexion
  python scripts/pilot_stress.py --goals stick furnace iron_pickaxe  # subset
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

RESULTS_DIR = os.path.join(ROOT, "results", "trajectories", "pilot_stress")
CONFIG_PATH = os.path.join(ROOT, "config", "hyperparams.yaml")


# ═══════════════════════════════════════════════════════════════════════════════
# Goal definitions with expected dependency depth
# ═══════════════════════════════════════════════════════════════════════════════

PILOT_GOALS = OrderedDict([
    # ── Tier 0: trivial (depth 3) ────────────────────────────────────────────
    ("stick", {
        "tier": 0,
        "group": "Wooden",
        "chain": "oak_log → planks → stick",
        "optimal_steps": 3,
    }),
    ("bowl", {
        "tier": 0,
        "group": "Wooden",
        "chain": "oak_log → planks → crafting_table → bowl",
        "optimal_steps": 4,
        "trap": "LLM may think bowl needs clay (real MC recipe ≠ simulator)",
    }),

    # ── Tier 1: easy (depth 5) ───────────────────────────────────────────────
    ("wooden_pickaxe", {
        "tier": 1,
        "group": "Wooden",
        "chain": "oak_log×3 → planks → sticks + crafting_table → pickaxe",
        "optimal_steps": 7,
    }),

    # ── Tier 2: medium (depth 7-9) ──────────────────────────────────────────
    ("torch", {
        "tier": 2,
        "group": "Stone",
        "chain": "... → wooden_pickaxe → mine_coal + stick → torch",
        "optimal_steps": 8,
    }),
    ("furnace", {
        "tier": 2,
        "group": "Stone",
        "chain": "... → wooden_pickaxe → cobblestone×8 → crafting_table → furnace",
        "optimal_steps": 14,
    }),

    # ── Tier 3: hard (depth 12+) ────────────────────────────────────────────
    ("iron_pickaxe", {
        "tier": 3,
        "group": "Iron",
        "chain": "... → furnace + stone_pickaxe → iron_ore×3 + charcoal×3 → smelt×3 → iron_pickaxe",
        "optimal_steps": 25,
    }),

    # ── Tier 4: very hard (depth 15+) ───────────────────────────────────────
    ("blast_furnace", {
        "tier": 4,
        "group": "Iron",
        "chain": "... → furnace + iron_ingot×5 + smooth_stone×3 (needs smelt cobblestone)",
        "optimal_steps": 40,
    }),
])


# ═══════════════════════════════════════════════════════════════════════════════
# Error classification from a finished episode
# ═══════════════════════════════════════════════════════════════════════════════

def classify_errors(result: dict) -> dict:
    """
    Scan a trajectory and produce error counts / flags.

    Returns dict:
      {
        "unknown_action":       int,   # actions not in library
        "missing_tool":         int,   # action exists, but tool missing
        "missing_material":     int,   # action exists, material shortage
        "repeated_same_fail":   int,   # same failed action appearing ≥3× in a row
        "total_failed_steps":   int,
        "total_successful_steps": int,
        "efficiency":           float, # successful / total
        "dominant_error_type":  str,   # single label for this episode
      }
    """
    traj = result.get("trajectory", [])
    unknown_action = 0
    missing_tool = 0
    missing_material = 0
    total_fail = 0
    total_ok = 0

    # Detect repeated same-action failures
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
                if streak >= 2:  # 3rd consecutive same-fail
                    repeated_same_fail += 1
            else:
                prev_action = step["action"]
                streak = 1

    total = total_ok + total_fail
    efficiency = total_ok / total if total > 0 else 0.0

    # Dominant error type
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


# ═══════════════════════════════════════════════════════════════════════════════
# Run a single goal with both agents
# ═══════════════════════════════════════════════════════════════════════════════

def run_goal(goal: str, meta: dict, react_agent, reflexion_agent, skip_reflexion: bool):
    """Run ReAct (and optionally Reflexion) on one goal. Returns result dict."""
    entry = {"goal": goal, **meta}

    # ── ReAct ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  [ReAct] {goal}  (Tier {meta['tier']}, {meta['group']})")
    print(f"{'─'*60}")

    t0 = time.time()
    react_result = react_agent.run_episode(goal)
    react_time = time.time() - t0
    react_result["elapsed_sec"] = round(react_time, 2)
    react_result["errors"] = classify_errors(react_result)
    entry["react"] = react_result

    tag = "✓" if react_result["success"] else "✗"
    print(f"  → [{tag}] {react_result['total_steps']} steps, "
          f"{react_result['termination']}, "
          f"dominant error: {react_result['errors']['dominant_error_type']}")

    # ── Reflexion ─────────────────────────────────────────────────────────
    if not skip_reflexion:
        print(f"\n  [Reflexion] {goal}")
        t0 = time.time()
        refl_result = reflexion_agent.run(goal)
        refl_time = time.time() - t0
        refl_result["elapsed_sec"] = round(refl_time, 2)

        # Classify errors for each attempt
        for attempt in refl_result.get("attempts", []):
            attempt["result"]["errors"] = classify_errors(attempt["result"])

        # Summarize: dominant error of the LAST attempt (most interesting)
        last_attempt = refl_result["attempts"][-1]["result"]
        refl_result["final_errors"] = classify_errors(last_attempt)

        entry["reflexion"] = refl_result

        tag = "✓" if refl_result["success"] else "✗"
        win = refl_result.get("winning_attempt")
        attempts_summary = " → ".join(
            f"{'✓' if a['result']['success'] else '✗'}"
            for a in refl_result["attempts"]
        )
        print(f"  → [{tag}] attempts: [{attempts_summary}], "
              f"dominant error: {refl_result['final_errors']['dominant_error_type']}")

    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# Summary report
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_entries: list, skip_reflexion: bool):
    print(f"\n\n{'═'*80}")
    print("  STRESS PILOT — SUMMARY")
    print(f"{'═'*80}\n")

    # ── Comparison table ─────────────────────────────────────────────────
    if skip_reflexion:
        header = f"  {'Goal':<18} {'Tier':>4} {'React':^10} {'Steps':>5} {'Eff':>5} {'Error Type':<24}"
        sep    = f"  {'─'*18} {'─'*4} {'─'*10} {'─'*5} {'─'*5} {'─'*24}"
    else:
        header = (f"  {'Goal':<18} {'Tier':>4} "
                  f"{'React':^10} {'Steps':>5} {'Eff':>5} "
                  f"{'Reflexion':^10} {'Steps':>5} {'Eff':>5} "
                  f"{'Error Type':<24}")
        sep    = (f"  {'─'*18} {'─'*4} "
                  f"{'─'*10} {'─'*5} {'─'*5} "
                  f"{'─'*10} {'─'*5} {'─'*5} "
                  f"{'─'*24}")

    print(header)
    print(sep)

    for e in all_entries:
        goal = e["goal"]
        tier = e["tier"]
        rr = e["react"]
        r_tag = "✓ OK" if rr["success"] else "✗ FAIL"
        r_steps = rr["total_steps"]
        r_eff = f"{rr['errors']['efficiency']:.0%}"
        r_err = rr["errors"]["dominant_error_type"]

        if skip_reflexion:
            print(f"  {goal:<18} {tier:>4} {r_tag:^10} {r_steps:>5} {r_eff:>5} {r_err:<24}")
        else:
            rx = e.get("reflexion", {})
            rx_tag = "✓ OK" if rx.get("success") else "✗ FAIL"
            # Total steps across all attempts
            rx_steps = sum(a["result"]["total_steps"] for a in rx.get("attempts", []))
            rx_eff = f"{rx.get('final_errors', {}).get('efficiency', 0):.0%}"
            rx_err = rx.get("final_errors", {}).get("dominant_error_type", "?")
            print(f"  {goal:<18} {tier:>4} "
                  f"{r_tag:^10} {r_steps:>5} {r_eff:>5} "
                  f"{rx_tag:^10} {rx_steps:>5} {rx_eff:>5} "
                  f"{rx_err:<24}")

    # ── Aggregate stats ──────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    react_sr = sum(1 for e in all_entries if e["react"]["success"])
    n = len(all_entries)
    print(f"  ReAct  success rate: {react_sr}/{n} ({100*react_sr/n:.0f}%)")

    if not skip_reflexion:
        refl_sr = sum(1 for e in all_entries if e.get("reflexion", {}).get("success"))
        print(f"  Reflexion SR:       {refl_sr}/{n} ({100*refl_sr/n:.0f}%)")

        # Where did reflexion recover over react?
        recoveries = [
            e["goal"] for e in all_entries
            if not e["react"]["success"] and e.get("reflexion", {}).get("success")
        ]
        regressions = [
            e["goal"] for e in all_entries
            if e["react"]["success"] and not e.get("reflexion", {}).get("success")
        ]
        if recoveries:
            print(f"  Reflexion recovered: {', '.join(recoveries)}")
        if regressions:
            print(f"  Reflexion regressed: {', '.join(regressions)}")

    # ── Error type distribution ──────────────────────────────────────────
    print(f"\n  Error type distribution (ReAct, failed goals only):")
    err_dist = {}
    for e in all_entries:
        if not e["react"]["success"]:
            t = e["react"]["errors"]["dominant_error_type"]
            err_dist[t] = err_dist.get(t, 0) + 1
    if err_dist:
        for t, cnt in sorted(err_dist.items(), key=lambda x: -x[1]):
            print(f"    {t:<28} {cnt}")
    else:
        print(f"    (all goals succeeded)")

    # ── Performance decay by tier ────────────────────────────────────────
    print(f"\n  Performance decay by tier:")
    tiers = sorted(set(e["tier"] for e in all_entries))
    for t in tiers:
        tier_entries = [e for e in all_entries if e["tier"] == t]
        tier_sr = sum(1 for e in tier_entries if e["react"]["success"])
        tier_n = len(tier_entries)
        goals = ", ".join(e["goal"] for e in tier_entries)
        bar = "█" * tier_sr + "░" * (tier_n - tier_sr)
        print(f"    Tier {t}: [{bar}] {tier_sr}/{tier_n}  ({goals})")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stress pilot: ReAct vs Reflexion")
    parser.add_argument("--react-only", action="store_true", help="Skip Reflexion")
    parser.add_argument("--goals", nargs="+", default=None,
                        help="Run only these goals (e.g. --goals stick furnace)")
    args = parser.parse_args()

    # Select goals
    if args.goals:
        goals = OrderedDict(
            (g, PILOT_GOALS[g]) for g in args.goals if g in PILOT_GOALS
        )
        if not goals:
            print(f"No valid goals found. Available: {list(PILOT_GOALS.keys())}")
            return
    else:
        goals = PILOT_GOALS

    skip_reflexion = args.react_only
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f"  STRESS PILOT — {len(goals)} goals × (ReAct{' + Reflexion' if not skip_reflexion else ''})")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    # ── Init agents ──────────────────────────────────────────────────────
    react_agent = ReactAgent(config_path=CONFIG_PATH, verbose=True)
    reflexion_agent = None
    if not skip_reflexion:
        reflexion_agent = ReflexionAgent(config_path=CONFIG_PATH, verbose=True)

    # ── Run ───────────────────────────────────────────────────────────────
    all_entries = []
    for goal, meta in goals.items():
        entry = run_goal(goal, meta, react_agent, reflexion_agent, skip_reflexion)
        all_entries.append(entry)

        # Save per-goal result
        out_path = os.path.join(RESULTS_DIR, f"{timestamp}_{goal}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(all_entries, skip_reflexion)

    # ── Save aggregate ────────────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, f"{timestamp}_stress_summary.json")
    summary = {
        "timestamp": timestamp,
        "n_goals": len(all_entries),
        "skip_reflexion": skip_reflexion,
        "react_success_rate": sum(1 for e in all_entries if e["react"]["success"]) / len(all_entries),
        "entries": all_entries,
    }
    if not skip_reflexion:
        summary["reflexion_success_rate"] = (
            sum(1 for e in all_entries if e.get("reflexion", {}).get("success"))
            / len(all_entries)
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"  Summary: {summary_path}")
    print(f"\n{'═'*80}\n")


if __name__ == "__main__":
    main()

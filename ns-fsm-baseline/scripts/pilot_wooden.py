"""
Phase 3 Pilot Test — Wooden Group (10 goals × 1 run)

目标:
  1. 确认 ReactAgent 能跑通完整 episode（stick / crafting_table 等简单目标）
  2. 统计 action_parser 解析质量（精确匹配 / 模糊匹配 / 解析失败）
  3. 确认 trajectory logging 完整
  4. 检查 LLM 输出格式，识别 prompt 需要调整的地方
  5. 根据实际耗时估算完整实验时间

运行:
  cd ns-fsm-baseline
  python scripts/pilot_wooden.py
  python scripts/pilot_wooden.py --dry-run   # 跳过 LLM，使用 mock 验证流程
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime

# ── 路径设置 ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "..", "MC-TextWorld"))

from react_agent import ReactAgent, detect_dead_loop
from action_parser import ActionParser
from prompts import format_inventory


# ── 常量 ──────────────────────────────────────────────────────────────────────
GOALS_PATH   = os.path.join(ROOT, "config", "goals_67.json")
RESULTS_DIR  = os.path.join(ROOT, "results", "trajectories", "pilot")
CONFIG_PATH  = os.path.join(ROOT, "config", "hyperparams.yaml")

# 完整实验规模（用于时间估算）
FULL_GOALS        = 67
FULL_RUNS_REACT   = 15
FULL_RUNS_REFLEXION = 15   # ×max_attempts 内部


# ═══════════════════════════════════════════════════════════════════════════════
# Mock LLM（--dry-run 模式）
# ═══════════════════════════════════════════════════════════════════════════════

class _MockLLM:
    """
    返回固定的、能完成 crafting_table 的动作序列。
    序列耗尽后循环回最后一条（防止 IndexError）。
    """
    _SEQUENCE = [
        "Thought: Need logs first.\nAction: mine_oak_log",
        "Thought: Need more logs.\nAction: mine_oak_log",
        "Thought: Need more logs.\nAction: mine_oak_log",
        "Thought: Convert to planks.\nAction: craft_oak_planks",
        "Thought: More planks.\nAction: craft_oak_planks",
        "Thought: More planks.\nAction: craft_oak_planks",
        "Thought: Make crafting table first.\nAction: craft_crafting_table",
        "Thought: Make sticks.\nAction: craft_stick",
        "Thought: Now craft the goal item.\nAction: craft_wooden_pickaxe",
        # 通用兜底：继续挖木头
        "Thought: Need more materials.\nAction: mine_oak_log",
    ]

    def __init__(self):
        self._idx = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._SEQUENCE[min(self._idx, len(self._SEQUENCE) - 1)]
        self._idx += 1
        return resp

    def parse_react_response(self, text: str):
        import re
        thought_m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL | re.IGNORECASE)
        action_m  = re.search(r"Action:\s*([a-zA-Z0-9_]+)", text, re.IGNORECASE)
        thought = thought_m.group(1).strip() if thought_m else ""
        action  = action_m.group(1).strip().lower() if action_m else ""
        return thought, action

    def parse_reflection(self, text: str) -> str:
        return text.strip()

    def reset_sequence(self):
        self._idx = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Instrumented parser wrapper — collects parse quality stats
# ═══════════════════════════════════════════════════════════════════════════════

class _TrackingParser:
    """Wraps an ActionParser, delegates all calls, and records parse quality."""

    def __init__(self, real_parser: ActionParser):
        self._real = real_parser
        self.stats = {"exact": 0, "fuzzy": 0, "failed": 0}
        self.log = []           # per-episode parse log (reset each episode)

    def parse(self, text: str):
        action, reason = self._real.parse(text)
        step = len(self.log) + 1
        if action is None:
            self.stats["failed"] += 1
            self.log.append({"step": step, "type": "failed", "reason": reason})
        elif reason == "ok":
            self.stats["exact"] += 1
            self.log.append({"step": step, "type": "exact"})
        else:
            self.stats["fuzzy"] += 1
            self.log.append({"step": step, "type": "fuzzy", "reason": reason})
        return action, reason

    def reset_log(self):
        self.log = []

    # Forward any other attribute access to the real parser
    def __getattr__(self, name):
        return getattr(self._real, name)


# ═══════════════════════════════════════════════════════════════════════════════
# PilotReactAgent — thin wrapper, NO duplicated episode loop
# ═══════════════════════════════════════════════════════════════════════════════

class PilotReactAgent(ReactAgent):
    """
    ReactAgent + parse-quality tracking.
    Delegates entirely to the parent run_episode; only swaps the parser
    for a tracking wrapper and optionally replaces the LLM with a mock.
    """

    def __init__(self, dry_run: bool = False, **kwargs):
        super().__init__(**kwargs)
        if dry_run:
            self.llm = _MockLLM()
        # Wrap the parser so we can collect stats without copying run_episode
        self._tracking_parser = _TrackingParser(self.parser)
        self.parser = self._tracking_parser

    @property
    def parse_stats(self):
        return self._tracking_parser.stats

    def run_episode(self, goal_item: str, reflection: str = "") -> dict:
        # Reset per-episode log and mock LLM sequence
        self._tracking_parser.reset_log()
        if hasattr(self.llm, "reset_sequence"):
            self.llm.reset_sequence()

        result = super().run_episode(goal_item, reflection=reflection)
        # Attach parse log to result for pilot diagnostics
        result["parse_log"] = list(self._tracking_parser.log)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Pilot 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def run_pilot(dry_run: bool = False):
    # ── 加载 Wooden goals ────────────────────────────────────────────────────
    with open(GOALS_PATH, encoding="utf-8") as f:
        all_goals = json.load(f)
    wooden_goals = all_goals["Wooden"]["goals"]   # list of {id, goal, type, instruction}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag  = "dryrun" if dry_run else "live"

    print(f"\n{'#'*60}")
    print(f"  PILOT TEST — Wooden Group × 1 run  [{mode_tag}]")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    # ── 初始化 agent ─────────────────────────────────────────────────────────
    agent = PilotReactAgent(
        config_path=CONFIG_PATH,
        dry_run=dry_run,
        verbose=True,
    )

    # ── 逐个 goal 运行 ───────────────────────────────────────────────────────
    all_results   = []
    episode_times = []

    for entry in wooden_goals:
        goal = entry["goal"]
        t0   = time.time()
        result = agent.run_episode(goal)
        elapsed = time.time() - t0
        episode_times.append(elapsed)

        result["elapsed_sec"] = round(elapsed, 2)
        all_results.append(result)

        # 保存单个 trajectory
        traj_file = os.path.join(
            RESULTS_DIR, f"{timestamp}_react_{goal}.json"
        )
        with open(traj_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # ═══════════════════════════════════════════════════════════════════════════
    # 汇总报告
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*60}")
    print("  PILOT SUMMARY")
    print(f"{'='*60}\n")

    # 1. Per-goal 结果表
    print(f"  {'Goal':<22} {'Result':<8} {'Steps':>5} {'Termination':<12} {'Time':>6}")
    print(f"  {'-'*22} {'-'*8} {'-'*5} {'-'*12} {'-'*6}")
    for r in all_results:
        ok  = "✓ OK" if r["success"] else "✗ FAIL"
        print(
            f"  {r['goal']:<22} {ok:<8} {r['total_steps']:>5} "
            f"{r['termination']:<12} {r['elapsed_sec']:>5.1f}s"
        )

    # 2. 成功率
    n_total   = len(all_results)
    n_success = sum(1 for r in all_results if r["success"])
    term_counts = {}
    for r in all_results:
        term_counts[r["termination"]] = term_counts.get(r["termination"], 0) + 1

    print(f"\n  Success rate : {n_success}/{n_total} ({100*n_success/n_total:.0f}%)")
    print(f"  Termination  : {term_counts}")

    # 3. Action parser 质量
    ps = agent.parse_stats
    total_parses = sum(ps.values()) or 1
    print(f"\n  Action parser breakdown ({sum(ps.values())} total parses):")
    print(f"    Exact match  : {ps['exact']:>4}  ({100*ps['exact']/total_parses:.1f}%)")
    print(f"    Fuzzy match  : {ps['fuzzy']:>4}  ({100*ps['fuzzy']/total_parses:.1f}%)")
    print(f"    Parse failed : {ps['failed']:>4}  ({100*ps['failed']/total_parses:.1f}%)")

    # 4. Trajectory 完整性检查
    print(f"\n  Trajectory logging check:")
    all_ok = True
    for r in all_results:
        steps_recorded = len(r["trajectory"])
        expected       = r["total_steps"]
        ok = steps_recorded == expected
        if not ok:
            print(f"    ✗ {r['goal']}: recorded {steps_recorded} steps, expected {expected}")
            all_ok = False
    if all_ok:
        print(f"    ✓ All {n_total} trajectories complete ({sum(r['total_steps'] for r in all_results)} steps total)")

    # 5. LLM 输出格式诊断
    _check_llm_format(all_results)

    # 6. 时间估算
    _estimate_full_runtime(episode_times, dry_run)

    # 7. 保存汇总
    summary_file = os.path.join(RESULTS_DIR, f"{timestamp}_pilot_summary.json")
    summary = {
        "timestamp":     timestamp,
        "mode":          mode_tag,
        "n_goals":       n_total,
        "n_success":     n_success,
        "termination_counts": term_counts,
        "parse_stats":   ps,
        "avg_steps":     round(sum(r["total_steps"] for r in all_results) / n_total, 1),
        "avg_time_sec":  round(sum(episode_times) / len(episode_times), 2),
        "results":       all_results,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {summary_file}")
    print(f"\n{'='*60}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# LLM 格式诊断
# ─────────────────────────────────────────────────────────────────────────────

def _check_llm_format(all_results: list):
    """
    扫描所有 parse_log，汇报需要关注的格式问题。
    """
    print(f"\n  LLM output format diagnosis:")

    fuzzy_reasons  = {}
    failed_reasons = {}

    for r in all_results:
        for entry in r.get("parse_log", []):
            if entry["type"] == "fuzzy":
                reason = entry.get("reason", "unknown")
                fuzzy_reasons[reason] = fuzzy_reasons.get(reason, 0) + 1
            elif entry["type"] == "failed":
                reason = entry.get("reason", "unknown")
                failed_reasons[reason] = failed_reasons.get(reason, 0) + 1

    if not fuzzy_reasons and not failed_reasons:
        print("    ✓ No format issues — LLM follows Action: <type>_<item> consistently")
    if fuzzy_reasons:
        print("    Fuzzy match triggers (prompt may need more examples):")
        for r, cnt in sorted(fuzzy_reasons.items(), key=lambda x: -x[1]):
            print(f"      [{cnt}×] {r}")
    if failed_reasons:
        print("    Parse failures (prompt format may need fixing):")
        for r, cnt in sorted(failed_reasons.items(), key=lambda x: -x[1]):
            print(f"      [{cnt}×] {r}")


# ─────────────────────────────────────────────────────────────────────────────
# 全量实验时间估算
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_full_runtime(episode_times: list, dry_run: bool):
    """
    根据 pilot 耗时推算完整实验所需时间。
    公式: ReAct  = 67 goals × 15 runs × avg_time
          Reflexion = 67 goals × 15 runs × max_attempts × avg_time  (保守估算)
    """
    avg_sec = sum(episode_times) / len(episode_times)

    react_total_sec      = FULL_GOALS * FULL_RUNS_REACT * avg_sec
    reflexion_total_sec  = FULL_GOALS * FULL_RUNS_REFLEXION * 2.5 * avg_sec  # 平均 2.5 次 attempt

    def fmt(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        return f"{h}h {m}m"

    print(f"\n  Runtime estimate (based on avg {avg_sec:.1f}s/episode):")
    if dry_run:
        print("    [dry-run: mock LLM — times are meaningless for real estimates]")
    print(f"    Avg episode time : {avg_sec:.1f}s")
    print(f"    ReAct  full run  : {FULL_GOALS} goals × {FULL_RUNS_REACT} runs  = {fmt(react_total_sec)}")
    print(f"    Reflexion full   : {FULL_GOALS} goals × {FULL_RUNS_REFLEXION} runs × ~2.5 attempts = {fmt(reflexion_total_sec)}")
    print(f"    Total estimate   : {fmt(react_total_sec + reflexion_total_sec)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot test for Wooden group")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use mock LLM (no Ollama needed) to verify pipeline end-to-end"
    )
    args = parser.parse_args()

    run_pilot(dry_run=args.dry_run)
"""
主入口：单次 / 批量实验运行器
"""
import os
import sys
import argparse
from datetime import datetime

from environment import MinecraftSurvivalEnv, DIFFICULTY_CONFIGS
from memory import StructuredMemory
from agent import SimpleLLMAgent
from parser import parse_command
from detector import LostInMiddleDetector


# ================================================================== #
#  日志
# ================================================================== #
class DualLogger:
    """同时输出到终端和日志文件。"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


# ================================================================== #
#  主流程
# ================================================================== #
def run_experiment(env, memory, agent, detector, max_steps, verbose=True):
    """执行一次完整实验，返回 (success, steps, detector_summary)。"""
    obs = env.reset()
    memory.reset()
    detector.signals.clear()

    terminated = False
    for step in range(1, max_steps + 1):
        prompt = memory.get_prompt_context()
        pre_inv = dict(obs)

        if verbose:
            print(f"\n--- Step {step} ---")

        # Agent 推理
        action_str, thought = agent.get_action(prompt)
        if not action_str:
            if verbose:
                print("  [Error] Agent 未能输出动作，实验中止。")
            return False, step, detector.get_summary()

        # 解析
        parsed = parse_command(action_str)
        if verbose:
            thought_short = thought[:120].replace("\n", " ")
            print(f"  <thought>: {thought_short}...")
            if parsed.is_valid:
                print(f"  <action>:  {parsed.raw_action}")
            else:
                print(f"  <action> INVALID: {parsed.error_msg}")

        # 环境执行
        obs, reward, terminated, info = env.step(parsed)

        # 记忆更新
        memory.update(action_str, info["message"], obs)

        # 检测
        env_state = env.get_env_state() if env.survival_enabled else None
        signals = detector.detect_all(
            step, parsed.raw_action, thought, pre_inv,
            memory.step_history, info["success"], info["message"],
            env_state=env_state,
        )

        if verbose:
            status = "✓" if info["success"] else "✗"
            print(f"  Env: [{status}] {info['message']}")
            for sig in signals:
                print(f"    └─ [WARNING] {sig['type']}: {sig['detail']}")

        if terminated:
            cause = info.get("cause", "")
            if cause in ("starvation", "mob_death"):
                if verbose:
                    print(f"\n💀 Agent 死亡 ({cause})，Step {step}")
                return False, step, detector.get_summary()
            if verbose:
                print(f"\n✅ 目标达成！用时 {step} 步")
            return True, step, detector.get_summary()

    if verbose:
        print(f"\n❌ {max_steps} 步内未完成目标。")
    return False, max_steps, detector.get_summary()


def main():
    parser = argparse.ArgumentParser(description="Minecraft LiM Experiment")
    parser.add_argument("--backend", default="ollama",
                        choices=["ollama", "vllm", "api"])
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--difficulty", default="medium",
                        choices=list(DIFFICULTY_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--memory", default="structured",
                        choices=["structured", "oracle"], help="选择记忆管理模式：structured或oracle")
    parser.add_argument("--batch", action="store_true",
                        help="批量运行所有难度（每难度 10 次）")
    parser.add_argument("--trials", type=int, default=10,
                        help="批量模式下每难度的试验次数")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = DualLogger(f"logs/experiment_{ts}.log")
    sys.stdout = logger

    agent = SimpleLLMAgent(
        model_name=args.model,
        backend=args.backend,
        api_base=args.api_base,
        api_key=args.api_key,
    )

    if args.batch:
        _run_batch(agent, args.trials, args.seed, args.memory)
    else:
        _run_single(agent, args.difficulty, args.seed, args.memory)

    logger.close()


def _run_single(agent, difficulty, seed, memory_type="structured"):
    cfg = DIFFICULTY_CONFIGS[difficulty]
    print(f"=== Single Experiment: {difficulty} ({cfg['description']}) ===")
    print(f"    target={cfg['target']}  distractors={cfg.get('distractors', [])}")
    print(f"    survival={cfg.get('survival', False)}  max_steps={cfg['max_steps']}")

    env = MinecraftSurvivalEnv(cfg, seed=seed)
    if memory_type == "oracle":
        from memory import OracleMemory
        mem = OracleMemory(env)
    else:
        from memory import StructuredMemory
        mem = StructuredMemory(env)
    det = LostInMiddleDetector()
    success, steps, summary = run_experiment(env, mem, agent, det, cfg["max_steps"])

    print(f"\n--- Summary ---")
    print(f"  Success: {success}  Steps: {steps}")
    print(f"  Signals: {summary}")


def _run_batch(agent, trials, base_seed, memory_type="structured"):
    import json
    results = {}

    for diff_name, cfg in DIFFICULTY_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Difficulty: {diff_name} — {cfg['description']}")
        print(f"{'='*60}")

        diff_results = []
        for trial in range(trials):
            seed = (base_seed or 0) + trial
            env = MinecraftSurvivalEnv(cfg, seed=seed)
            if memory_type == "oracle":
                from memory import OracleMemory
                mem = OracleMemory(env)
            else:
                from memory import StructuredMemory
                mem = StructuredMemory(env)
            det = LostInMiddleDetector()

            print(f"\n  Trial {trial+1}/{trials} (seed={seed})")
            success, steps, summary = run_experiment(
                env, mem, agent, det, cfg["max_steps"], verbose=False)
            print(f"    result={'✅' if success else '❌'}  steps={steps}  signals={summary['total']}")

            diff_results.append({
                "trial": trial + 1,
                "seed": seed,
                "success": success,
                "steps": steps,
                "signals": summary,
            })
        results[diff_name] = diff_results

    # 汇总报告
    print(f"\n{'='*60}")
    print("  BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"{'Difficulty':<12} {'Success%':>10} {'AvgSteps':>10} {'AvgSignals':>12}")
    print("-" * 46)

    for diff, trials_data in results.items():
        sr = sum(1 for t in trials_data if t["success"]) / len(trials_data)
        avg_steps = sum(t["steps"] for t in trials_data) / len(trials_data)
        avg_sigs = sum(t["signals"]["total"] for t in trials_data) / len(trials_data)
        print(f"{diff:<12} {sr:>9.0%} {avg_steps:>10.1f} {avg_sigs:>12.1f}")

    # 保存 JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"logs/batch_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Raw results saved to logs/batch_{ts}.json")


if __name__ == "__main__":
    main()

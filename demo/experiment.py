"""
Lost in the Middle 批量实验运行器
使用旧版 model_fn 回调接口，兼容 vLLM / transformers / API 后端。
"""
import json
import re
import time
import os
import sys
import urllib.request
from datetime import datetime

class DualLogger:
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

from environment import MinecraftSurvivalEnv, DIFFICULTY_CONFIGS
from parser import parse_command
from detector import LostInMiddleDetector


from agent import SimpleLLMAgent

# ============================================================
# 1. Action / Thought 提取（从模型原始输出）
# ============================================================

def extract_action_thought(response_text):
    return SimpleLLMAgent._parse_response(response_text)


# ============================================================
# 2. 模型后端接口
# ============================================================

def call_model_ollama(prompt, model_name="qwen2.5:7b", temperature=0.5):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        url, data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode('utf-8')).get('response', '')
    except Exception as e:
        print(f"Ollama API 调用失败: {e}")
        return ""


def call_model_vllm(prompt, model_name="Qwen/Qwen2.5-7B-Instruct",
                     max_tokens=2048, temperature=0.5):
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=temperature)
    return resp.choices[0].message.content


def call_model_transformers(prompt, model=None, tokenizer=None,
                             max_new_tokens=2048, temperature=0.1):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                              temperature=temperature, do_sample=True, top_p=0.9)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def call_model_api(prompt, api_key=None, api_base=None, model_name="gpt-4o-mini"):
    import openai
    client = openai.OpenAI(
        api_key=api_key or "your_api_key",
        base_url=api_base or "https://ai-gateway.andrew.cmu.edu" # CMU LiteLLM Proxy
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048, temperature=0.5)
    return resp.choices[0].message.content


# ============================================================
# 3. Prompt 构建（独立于 memory.py，用于 experiment 独立运行）
# ============================================================

def build_prompt(env, history, config):
    """为 experiment.py 构建环境 prompt（与 memory.py 逻辑一致）。"""
    inv = env.inventory
    survival = env.survival_enabled

    inv_str = "，".join(f"{k}×{v}" for k, v in inv.items() if v > 0) or "空"

    goal = f"【最终目标】合成 {config['target']}\n"
    if config.get("distractors"):
        goal += f"【支线任务】先完成：{', '.join(config['distractors'])}\n"

    # 群系
    biome_text = "【世界地图】\n"
    for name, info in env.BIOMES.items():
        acts = list(env.BIOME_ACTIONS.get(name, {}).keys())
        if name == "矿洞":
            acts.append("挖矿 <矿石>")
        biome_text += f"  · {name}：{info['description']}（{', '.join(acts)}）\n"

    # 配方
    recipe_text = "【合成配方表】\n"
    for item, r in env.RECIPES.items():
        mats = " + ".join(f"{k}×{v}" for k, v in r["材料"].items())
        wb = " [需工作台]" if r.get("需要工作台") else ""
        recipe_text += f"  {mats} => {item}×{r['产出']}{wb}\n"

    smelt_text = "【熔炼配方】（需熔炉）\n"
    for src, info in env.SMELTING.items():
        smelt_text += f"  {src} => {info['产出']}×{info['数量']}\n"

    mine_text = "【挖矿工具需求】（需在矿洞）\n"
    for ore, req in env.MINING_REQS.items():
        mine_text += f"  {ore}：需 {' 或 '.join(req['tools'])}\n"

    # 工具耐久
    dur = env.get_tool_durability_info()
    dur_text = "【工具耐久】" + (
        "  ".join(f"{t}: {d}" for t, d in dur.items()) if dur else "无工具")

    # 生存
    surv_text = ""
    if survival:
        ts = "夜晚 ⚠" if env._is_night() else "白天"
        
        grace_cycles = config.get("grace_cycles", 0)
        current_cycle = env.step_count // getattr(env, 'DAY_LENGTH', 20)
        if env._is_night() and current_cycle < grace_cycles:
            ts = "夜晚 (安全期)"
            
        surv_text = (
            f"【生存状态】HP:{env.hp}/{env.max_hp} 饥饿:{env.hunger}/{env.max_hunger} "
            f"时间:{ts}\n"
            f"  可食用：苹果(+4) 面包(+5) 熟肉(+8) 生肉(+2)\n"
            f"  [防护提示] 火把可减免50%夜间伤害，床/避难所可完全免伤。")

    hist_text = "无"
    if history:
        hist_text = "\n".join(f"Step {i+1}: {h}" for i, h in enumerate(history))

    prompt = f"""你是一个 Minecraft 世界中的 Agent。请严格逻辑推理完成任务。

{goal}
{biome_text}
{recipe_text}
{smelt_text}
{mine_text}
【流程指引】
1. 森林砍树→木板→木棍→工作台→木镐
2. 移动矿洞→挖圆石→石镐+熔炉
3. 挖铁矿石→熔炼→铁镐
4. 挖钻石→钻石镐
5. 钻石镐挖黑曜石；纸+皮革→书→附魔台
不要重复合成已有的工具/工作台/熔炉！工具有耐久，损坏需重做。

【动作历史】
{hist_text}

【当前状态】
位置：{env.current_biome}
背包：{inv_str}
{dur_text}
{surv_text}

【可用动作】
砍树 / 打猎 / 采集 <物品> / 挖矿 <矿石> / 熔炼 <原料> / 合成 <物品>
移动 <地点> / 吃 <食物> / 建造 避难所 / 睡觉 / 打水 / 结束

你是 Minecraft 生存助手。每次只输出如下格式，不要输出任何其他内容：

<thought>简短思考</thought>
<action>具体动作</action>

示例1：
<thought>需要木板，先砍树获取原木</thought>
<action>砍树</action>

示例2：
<thought>有了原木，合成木板</thought>
<action>合成 木板</action>

示例3：
<thought>需要去矿洞挖石头</thought>
<action>移动 矿洞</action>

现在请根据当前状态行动：
重要：你的回复必须以 <thought> 开始，以 </action> 结束。中间不要有其他内容。
"""
    return prompt


# ============================================================
# 4. 单次实验
# ============================================================

def run_single_experiment(difficulty, model_fn, max_steps=None, verbose=True, seed=None, memory_type="structured"):
    config = DIFFICULTY_CONFIGS[difficulty]
    if max_steps is None:
        max_steps = config.get("max_steps", 80)

    env = MinecraftSurvivalEnv(config, seed=seed)
    detector = LostInMiddleDetector()
    env.reset()

    history = []
    step_details = []
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"实验开始 | 难度: {difficulty} | 目标: {config['target']}")
        print(f"支线: {config.get('distractors', [])}")
        print(f"生存模式: {config.get('survival', False)}")
        print(f"{'='*60}")

    for step in range(1, max_steps + 1):
        prompt = build_prompt(env, history if memory_type == "structured" else [], config)
        pre_inv = dict(env.inventory)
        input_tokens = len(prompt) // 2

        try:
            response = model_fn(prompt)
        except Exception as e:
            if verbose:
                print(f"  Step {step}: 模型调用失败 - {e}")
            break

        action_str, thought = SimpleLLMAgent._parse_response(response)

        if not action_str:
            if verbose:
                print(f"  Step {step}: 无法解析 action，跳过")
            history.append("[解析失败] 模型输出无法识别")
            continue

        parsed = parse_command(action_str)
        obs, reward, terminated, info = env.step(parsed)

        history.append(f"Step {step}: {parsed.raw_action} → {info['message']}")

        env_state = env.get_env_state() if env.survival_enabled else None
        step_signals = detector.detect_all(
            step, parsed.raw_action, thought, pre_inv, history,
            info["success"], info["message"], env_state=env_state)

        step_details.append({
            "step": step, "action": action_str,
            "success": info["success"], "message": info["message"],
            "inventory_snapshot": dict(env.inventory),
            "input_tokens_approx": input_tokens,
            "signals": step_signals, "thought_length": len(thought),
        })

        if verbose:
            status = "✓" if info["success"] else "✗"
            sig_str = f" ⚠ {len(step_signals)}信号" if step_signals else ""
            print(f"  Step {step}: [{status}] {action_str} | "
                  f"{info['message'][:80]}{sig_str}")
            if thought:
                print(f"          ├─ thought: {thought[:100]}...")
            for sig in step_signals:
                print(f"          └─ [{sig['severity'].upper()}] "
                      f"{sig['type']}: {sig['detail']}")

        if terminated:
            cause = info.get("cause", "")
            elapsed = time.time() - start_time
            if cause in ("starvation", "mob_death"):
                if verbose:
                    print(f"\n  💀 Agent 死亡 ({cause})，Step {step}")
                return _make_result(False, difficulty, config, step,
                                     elapsed, env, detector, step_details, history)
            if verbose:
                print(f"\n  ✅ 任务完成！总步数: {step}，耗时: {elapsed:.1f}s")
            return _make_result(True, difficulty, config, step,
                                 elapsed, env, detector, step_details, history)

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n  ❌ 超时！达到 {max_steps} 步上限")
    return _make_result(False, difficulty, config, max_steps,
                         elapsed, env, detector, step_details, history)


def _make_result(success, difficulty, config, steps, elapsed,
                  env, detector, step_details, history):
    return {
        "success": success,
        "difficulty": difficulty,
        "target": config["target"],
        "total_steps": steps,
        "elapsed_seconds": elapsed,
        "final_inventory": dict(env.inventory),
        "detector_summary": detector.get_summary(),
        "step_details": step_details,
        "history": history,
    }


# ============================================================
# 5. 批量实验与报告
# ============================================================

def run_batch_experiment(model_fn, num_trials=10, difficulties=None,
                          max_steps=None, output_dir="results", memory_type="structured"):
    if difficulties is None:
        difficulties = ["short", "medium", "long", "extreme"]

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for diff in difficulties:
        cfg = DIFFICULTY_CONFIGS[diff]
        ms = max_steps or cfg.get("max_steps", 80)
        print(f"\n{'#'*60}")
        print(f"# 难度: {diff} | 重复: {num_trials} | max_steps: {ms}")
        print(f"# {cfg['description']}")
        print(f"{'#'*60}")

        trial_results = []
        for trial in range(num_trials):
            print(f"\n--- Trial {trial+1}/{num_trials} ---")
            result = run_single_experiment(
                difficulty=diff, model_fn=model_fn,
                max_steps=ms, verbose=(trial == 0),
                seed=trial, memory_type=memory_type)
            trial_results.append(result)
            status = "✅" if result["success"] else "❌"
            print(f"  {status} Steps={result['total_steps']}, "
                  f"Signals={result['detector_summary']['total']}")

        all_results[diff] = trial_results

    report = generate_report(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dir, f"raw_{timestamp}.json"), 'w',
              encoding='utf-8') as f:
        serializable = {}
        for diff, trials in all_results.items():
            serializable[diff] = [{k: v for k, v in t.items()
                                    if k != "step_details"} for t in trials]
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, f"report_{timestamp}.txt"), 'w',
              encoding='utf-8') as f:
        f.write(report)

    print(f"\n结果保存至 {output_dir}/")
    return all_results, report


def generate_report(all_results):
    lines = [
        "=" * 70,
        "Lost in the Middle Detection — Experiment Report",
        "=" * 70, "",
        f"{'难度':<10} {'成功率':<10} {'平均步数':<10} {'平均信号':<10} "
        f"{'首信号步':<10} {'冗余合成':<10} {'幻觉':<10} {'目标漂移':<10}",
        "-" * 80,
    ]

    for diff in ("short", "medium", "long", "extreme"):
        trials = all_results.get(diff, [])
        if not trials:
            continue
        n = len(trials)
        sr = sum(1 for t in trials if t["success"]) / n
        avg_s = sum(t["total_steps"] for t in trials) / n
        avg_sig = sum(t["detector_summary"]["total"] for t in trials) / n

        first_steps = [t["detector_summary"]["first_signal_step"]
                       for t in trials if t["detector_summary"].get("first_signal_step")]
        avg_f = sum(first_steps) / len(first_steps) if first_steps else float('inf')

        tc = {}
        for t in trials:
            for st, c in t["detector_summary"]["by_type"].items():
                tc[st] = tc.get(st, 0) + c

        rc = tc.get("redundant_craft", 0) / n
        ih = tc.get("inventory_hallucination", 0) / n
        gd = tc.get("goal_drift", 0) / n
        fs = f"{avg_f:.1f}" if avg_f != float('inf') else "N/A"

        lines.append(f"{diff:<10} {sr:<10.1%} {avg_s:<10.1f} {avg_sig:<10.1f} "
                     f"{fs:<10} {rc:<10.1f} {ih:<10.1f} {gd:<10.1f}")

    lines += ["", "=" * 70, "关键发现：", "=" * 70]

    short_t = all_results.get("short", [])
    extreme_t = all_results.get("extreme", [])
    if short_t and extreme_t:
        s_sr = sum(1 for t in short_t if t["success"]) / len(short_t)
        e_sr = sum(1 for t in extreme_t if t["success"]) / len(extreme_t)
        s_sig = sum(t["detector_summary"]["total"] for t in short_t) / len(short_t)
        e_sig = sum(t["detector_summary"]["total"] for t in extreme_t) / len(extreme_t)

        lines.append(f"1. 成功率: short({s_sr:.0%}) → extreme({e_sr:.0%})，"
                     f"下降 {s_sr - e_sr:.0%}")
        lines.append(f"2. 信号数: short(avg {s_sig:.1f}) → extreme(avg {e_sig:.1f})，"
                     f"增加 {e_sig / max(s_sig, 0.1):.1f}x")

    long_t = all_results.get("long", [])
    if long_t:
        hall = sum(t["detector_summary"]["by_type"].get("inventory_hallucination", 0)
                   for t in long_t)
        lines.append(f"3. long 难度平均 {hall/len(long_t):.1f} 次背包幻觉")

    # 新增环境信号汇总
    for diff in ("medium", "long", "extreme"):
        trials = all_results.get(diff, [])
        if not trials:
            continue
        spatial = sum(t["detector_summary"]["by_type"].get("spatial_confusion", 0)
                      for t in trials)
        neglect = sum(t["detector_summary"]["by_type"].get("survival_neglect", 0)
                      for t in trials)
        if spatial or neglect:
            lines.append(f"4. {diff}: 空间困惑 avg {spatial/len(trials):.1f}，"
                         f"生存忽视 avg {neglect/len(trials):.1f}")

    lines += [
        "",
        "结论：随着轨迹长度和环境复杂度增加，LLM 出现系统性的中间信息遗忘，",
        "表现为冗余合成、背包幻觉、目标漂移、空间困惑和生存忽视等现象。",
    ]
    return "\n".join(lines)


# ============================================================
# 6. 主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LiM Batch Experiment")
    parser.add_argument("--backend", default="ollama",
                        choices=["vllm", "transformers", "api", "ollama"])
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--difficulties", nargs="+",
                        default=["short", "medium", "long", "extreme"])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--memory", default="structured", choices=["structured", "oracle"])
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()

    if args.backend == "ollama":
        model_fn = lambda p: call_model_ollama(p, model_name=args.model)
    elif args.backend == "vllm":
        model_fn = lambda p: call_model_vllm(p, model_name=args.model)
    elif args.backend == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"加载模型 {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")
        model_fn = lambda p: call_model_transformers(
            p, model=model, tokenizer=tokenizer)
    elif args.backend == "api":
        model_fn = lambda p: call_model_api(
            p, api_key=args.api_key, api_base=args.api_base, model_name=args.model)

    if args.single:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/experiment_{timestamp}.log"
        logger = DualLogger(log_file)
        sys.stdout = logger
        
        try:
            single_diff = args.difficulties[0] if isinstance(args.difficulties, list) and args.difficulties else "extreme"
            result = run_single_experiment(single_diff, model_fn, memory_type=args.memory, verbose=True)
            print(f"\n检测摘要: "
                  f"{json.dumps(result['detector_summary'], ensure_ascii=False, indent=2)}")
        finally:
            logger.close()
            print(f"\n单次运行日志已自动保存到: {log_file}")
    else:
        _, report = run_batch_experiment(
            model_fn, num_trials=args.trials, memory_type=args.memory,
            difficulties=args.difficulties,
            max_steps=args.max_steps,
            output_dir=args.output_dir)
        print(f"\n{report}")

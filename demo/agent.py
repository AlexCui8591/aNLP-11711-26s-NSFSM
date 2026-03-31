import re
import json
import time
import urllib.request


class SimpleLLMAgent:
    """LLM Agent，支持 Ollama / vLLM / OpenAI-compatible API 三种后端"""

    def __init__(self, model_name="qwen2.5:14b", backend="ollama",
                 api_base=None, api_key=None, max_retries=2, temperature=0.5):
        self.model_name = model_name
        self.backend = backend
        self.api_base = api_base
        self.api_key = api_key
        self.max_retries = max_retries
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    #  公共接口
    # ------------------------------------------------------------------ #
    def get_action(self, prompt):
        """调用模型并解析出 (action, thought)。自动重试。"""
        for attempt in range(self.max_retries + 1):
            try:
                if self.backend == "ollama":
                    full_resp = self._call_ollama(prompt)
                elif self.backend == "vllm":
                    full_resp = self._call_vllm(prompt)
                elif self.backend == "api":
                    full_resp = self._call_openai_api(prompt)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")

                action, thought = self._parse_response(full_resp)
                if action:
                    return action, thought

                if attempt < self.max_retries:
                    print(f"  [Retry {attempt+1}] 未能解析出动作，重试中...")
                    continue
                return action, thought

            except Exception as e:
                if attempt < self.max_retries:
                    print(f"  [Retry {attempt+1}] API 调用失败: {e}")
                    time.sleep(1)
                    continue
                print(f"  [Error] API 调用最终失败: {e}")
                return "", ""

        return "", ""

    # ------------------------------------------------------------------ #
    #  后端实现
    # ------------------------------------------------------------------ #
    def _call_ollama(self, prompt):
        url = self.api_base or "http://localhost:11434/api/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            url, data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "")

    def _call_vllm(self, prompt):
        url = self.api_base or "http://localhost:8000/v1/completions"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": self.temperature,
        }
        req = urllib.request.Request(
            url, data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["text"]

    def _call_openai_api(self, prompt):
        import openai
        client = openai.OpenAI(
            api_key=self.api_key or "your_api_key",
            base_url=self.api_base or "https://ai-gateway.andrew.cmu.edu" # CMU LiteLLM Proxy
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------ #
    #  响应解析
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(full_resp):
        thought, action = full_resp, ""

        # --- Layer 1: XML 标签（最可靠） ---
        t_match = re.search(r"<thought>\s*(.*?)\s*</thought>", full_resp, re.DOTALL)
        if t_match:
            thought = t_match.group(1).strip()

        a_match = re.search(r"<action>\s*(.*?)\s*</action>", full_resp, re.DOTALL)
        if a_match:
            return a_match.group(1).strip(), thought

        # --- Layer 2: 各种前缀变体（含 markdown 加粗） ---
        prefix_patterns = [
            r"(?:\*{0,2})(?:动作|操作|行动|执行|Action)(?:\*{0,2})\s*[：:]\s*(.+)",
            r"(?:我(?:选择|决定|要|应该))\s*(.+?)(?:[。，\n]|$)",
        ]
        for pat in prefix_patterns:
            m = re.search(pat, full_resp)
            if m:
                action = m.group(1).strip().rstrip("。，.）)")
                if action:
                    return action, thought

        # --- Layer 3: 合法动作关键词兜底 ---
        KNOWN_ACTIONS = [
            "砍树", "打猎", "采集 苹果", "采集 小麦", "采集 甘蔗", "采集 沙子",
            "挖矿 圆石", "挖矿 煤炭", "挖矿 铁矿石", "挖矿 金矿石", "挖矿 钻石",
            "挖矿 黑曜石", "挖矿 红石",
            "合成 木板", "合成 木棍", "合成 工作台", "合成 木镐", "合成 石镐",
            "合成 熔炉", "合成 铁镐", "合成 火把", "合成 木剑", "合成 石剑", 
            "合成 铁剑", "合成 钻石镐", "合成 钻石剑", "合成 箱子", "合成 门", 
            "合成 梯子", "合成 床", "合成 铁桶", "合成 盾牌", "合成 皮革胸甲", 
            "合成 铁胸甲", "合成 钻石胸甲", "合成 面包", "合成 纸", "合成 书", 
            "合成 书架", "合成 附魔台",
            "熔炼 铁矿石", "熔炼 金矿石", "熔炼 沙子", "熔炼 生肉",
            "移动 森林", "移动 矿洞", "移动 平原", "移动 沙漠",
            "吃 苹果", "吃 面包", "吃 熟肉", "吃 生肉",
            "建造 避难所", "睡觉", "打水",
        ]
        # 按长度降序匹配，避免 "合成 木" 抢走 "合成 木板"
        for act in sorted(KNOWN_ACTIONS, key=len, reverse=True):
            if act in full_resp:
                return act, thought

        return "", thought

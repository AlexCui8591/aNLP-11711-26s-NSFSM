import os
import re
import time

import yaml

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised indirectly in tests
    OpenAI = None


class LLMInterface:
    def __init__(self, config_path: str = None):
        if not config_path:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
            config_path = os.path.join(config_dir, "hyperparams.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)["llm"]

        api_base = self.config.get("api_base") or "http://localhost:11434/v1"
        api_key = self.config.get("api_key") or "ollama_placeholder"

        self.model = self.config["model_name"]
        self.temperature = self.config.get("temperature", 0.3)
        self.max_retries = self.config.get("max_retries", 2)
        self.timeout = self.config.get("timeout", 60)

        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required to create LLMInterface. "
                "Install it or patch llm_interface.OpenAI in tests."
            )

        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=self.timeout,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"[LLM] 调用失败（第 {attempt + 1} 次）：{e}，{wait}s 后重试...")
                    time.sleep(wait)

        print(f"[LLM] 已达最大重试次数（{self.max_retries}），最后错误：{last_exc}")
        raise last_exc

    def parse_react_response(self, text: str) -> tuple:
        thought = ""
        action = ""

        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL | re.IGNORECASE
        )
        action_match = re.search(
            r"Action:\s*([a-zA-Z0-9_]+)", text, re.IGNORECASE
        )

        if thought_match:
            thought = thought_match.group(1).strip()
        if action_match:
            action = action_match.group(1).strip().lower()

        return thought, action

    def parse_reflection(self, text: str) -> str:
        match = re.search(r"Reflection:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

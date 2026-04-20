import json
import os
import re
import time
import urllib.error
import urllib.request

import yaml

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised indirectly in tests
    OpenAI = None


class LLMInterface:
    """OpenAI-compatible LLM client with a stdlib HTTP fallback.

    PSC/vLLM and local Ollama both expose an OpenAI-compatible
    /v1/chat/completions endpoint. If the optional openai package is available
    we use it; otherwise we issue the same request with urllib so smoke/full
    runs do not fail just because the package is missing.
    """

    def __init__(self, config_path: str = None):
        if not config_path:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
            config_path = os.path.join(config_dir, "hyperparams.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)["llm"]

        api_base = (
            os.environ.get("NSFSM_LLM_API_BASE")
            or self.config.get("api_base")
            or "http://localhost:11434/v1"
        )
        api_key = os.environ.get("NSFSM_LLM_API_KEY") or self.config.get("api_key") or "ollama_placeholder"
        model_name = os.environ.get("NSFSM_LLM_MODEL_NAME") or self.config["model_name"]

        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model_name
        self.temperature = float(os.environ.get("NSFSM_LLM_TEMPERATURE") or self.config.get("temperature", 0.3))
        self.max_retries = int(os.environ.get("NSFSM_LLM_MAX_RETRIES") or self.config.get("max_retries", 2))
        self.timeout = int(os.environ.get("NSFSM_LLM_TIMEOUT") or self.config.get("timeout", 60))
        self.client = None

        if OpenAI is not None:
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.client is not None:
                    return self._generate_with_openai_client(system_prompt, user_prompt)
                return self._generate_with_http(system_prompt, user_prompt)
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = 2**attempt
                    print(f"[LLM] call failed on attempt {attempt + 1}: {e}; retrying in {wait}s.")
                    time.sleep(wait)

        print(f"[LLM] max retries reached ({self.max_retries}); last error: {last_exc}")
        raise last_exc

    def _generate_with_openai_client(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _generate_with_http(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from LLM endpoint: {detail}") from exc

        parsed = json.loads(raw)
        return parsed["choices"][0]["message"]["content"].strip()

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

import json
import os
import re
import time
import urllib.error
import urllib.request

import yaml

OpenAI = None


class LLMInterface:
    """LLM client for OpenAI-compatible endpoints or local HF generation.

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

        self.ignore_env_overrides = bool(self.config.get("ignore_env_overrides", False))
        env = (lambda name: None) if self.ignore_env_overrides else os.environ.get
        self.backend = str(env("NSFSM_LLM_BACKEND") or self.config.get("backend") or "").lower()
        api_key_env = self.config.get("api_key_env")
        api_key_from_named_env = os.environ.get(str(api_key_env)) if api_key_env else None
        api_base = (
            env("NSFSM_LLM_API_BASE")
            or self.config.get("api_base")
            or "http://localhost:11434/v1"
        )
        api_key = (
            env("NSFSM_LLM_API_KEY")
            or api_key_from_named_env
            or self.config.get("api_key")
            or "ollama_placeholder"
        )
        model_name = env("NSFSM_LLM_MODEL_NAME") or self.config["model_name"]

        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model_name
        self.temperature = float(env("NSFSM_LLM_TEMPERATURE") or self.config.get("temperature", 0.3))
        self.max_retries = int(env("NSFSM_LLM_MAX_RETRIES") or self.config.get("max_retries", 2))
        self.timeout = int(env("NSFSM_LLM_TIMEOUT") or self.config.get("timeout", 60))
        self.client = None
        self.hf_tokenizer = None
        self.hf_model = None
        self.hf_device = None
        self.max_new_tokens = int(env("NSFSM_LLM_MAX_NEW_TOKENS") or self.config.get("max_new_tokens", 512))

        if self.backend in {"hf", "huggingface", "transformers", "local"}:
            self._init_hf_backend()
            return

        global OpenAI
        if OpenAI is None:
            try:
                from openai import OpenAI as _OpenAI

                OpenAI = _OpenAI
            except ImportError:  # pragma: no cover - exercised indirectly in tests
                OpenAI = None
        if OpenAI is not None:
            try:
                self.client = OpenAI(
                    base_url=self.api_base,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except Exception as exc:
                print(f"[LLM] OpenAI client unavailable ({exc}); using HTTP fallback.")
                self.client = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.hf_model is not None:
                    return self._generate_with_hf(system_prompt, user_prompt)
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

    def _init_hf_backend(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if self.hf_device == "cuda" else torch.float32
        dtype_name = str(self.config.get("torch_dtype") or "").lower()
        if dtype_name in {"float16", "fp16"}:
            torch_dtype = torch.float16
        elif dtype_name in {"float32", "fp32"}:
            torch_dtype = torch.float32
        elif dtype_name in {"bfloat16", "bf16"}:
            torch_dtype = torch.bfloat16

        print(f"[LLM] Loading HF model {self.model} on {self.hf_device}.", flush=True)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
        )
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch_dtype,
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
        )
        self.hf_model.to(self.hf_device)
        self.hf_model.eval()
        print("[LLM] HF model ready.", flush=True)

    def _generate_with_hf(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.hf_tokenizer, "apply_chat_template"):
            text = self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        inputs = self.hf_tokenizer(text, return_tensors="pt").to(self.hf_device)
        do_sample = self.temperature > 0
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.hf_tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = float(self.config.get("top_p", 0.95))
        with torch.inference_mode():
            output_ids = self.hf_model.generate(**inputs, **kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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

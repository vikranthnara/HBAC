from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from hbac.core.config import LLMConfig


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


Message = dict[str, str]


class LLMBackend(ABC):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> LLMResponse: ...

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def count_messages_tokens(self, messages: list[Message]) -> int:
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get("content", "")) + 4
        return total

    @classmethod
    def from_config(cls, config: LLMConfig | None = None) -> LLMBackend:
        config = config or LLMConfig()
        provider = config.provider.lower()
        if provider in {"openai", "vllm", "freellmapi"}:
            if provider == "vllm":
                return VLLMBackend(config)
            return OpenAIBackend(config)
        if provider == "transformers":
            return TransformersBackend(config)
        if provider == "anthropic":
            return AnthropicBackend(config)
        raise ValueError(f"Unknown LLM provider: {config.provider}")

    @classmethod
    def from_spec(cls, spec: str, **kwargs: Any) -> LLMBackend:
        """Parse 'provider:model' spec, e.g. freellmapi:auto or openai:gpt-4o-mini."""
        if spec in {"auto", "auto:auto"}:
            config = LLMConfig(**kwargs)
            return cls.from_config(config)
        if ":" in spec:
            provider, model = spec.split(":", 1)
        else:
            provider, model = "auto", spec
        config = LLMConfig(provider=provider, model=model, **kwargs)
        return cls.from_config(config)


class OpenAIBackend(LLMBackend):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        from openai import OpenAI

        if not config.api_key:
            provider = config.provider.lower()
            if provider == "freellmapi":
                raise ValueError(
                    "FreeLLMAPI key not found. Set HBAC_FREELLMAPI_DIR to your "
                    ".freellmapi clone (e.g. paradocs/.freellmapi) and ensure the "
                    "server has been started once, or set FREELLMAPI_API_KEY in .env."
                )
            if provider == "openai":
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY in .env (repo root) "
                    "or export HBAC_LLM_API_KEY."
                )

        client_kwargs: dict[str, Any] = {}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = OpenAI(**client_kwargs)

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            stop=stop,
            timeout=self.config.timeout_seconds,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else self.count_messages_tokens(messages),
            completion_tokens=usage.completion_tokens if usage else self.count_tokens(choice.message.content or ""),
            latency_ms=latency_ms,
            raw={"id": response.id, "model": response.model},
        )


class AnthropicBackend(LLMBackend):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        from anthropic import Anthropic

        if not config.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY in .env "
                "or export HBAC_LLM_API_KEY."
            )

        client_kwargs: dict[str, Any] = {}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = Anthropic(**client_kwargs)

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        system = ""
        chat_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
            "temperature": self.config.temperature,
        }
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop

        response = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        text_blocks = [b.text for b in response.content if b.type == "text"]
        text = "\n".join(text_blocks)
        return LLMResponse(
            text=text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            raw={"id": response.id, "model": response.model},
        )


class VLLMBackend(OpenAIBackend):
    """OpenAI-compatible local vLLM server."""

    def __init__(self, config: LLMConfig) -> None:
        updates: dict[str, Any] = {}
        if not config.base_url:
            updates["base_url"] = os.environ.get(
                "HBAC_LLM_BASE_URL", "http://localhost:8000/v1"
            )
        if not config.api_key:
            updates["api_key"] = os.environ.get("HBAC_LLM_API_KEY", "EMPTY")
        if updates:
            config = config.model_copy(update=updates)
        super().__init__(config)


class TransformersBackend(LLMBackend):
    """Local HuggingFace inference (no vLLM server)."""

    _cache: dict[str, tuple[Any, Any]] = {}

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = config.model
        cache_key = f"{model_id}::lora={config.lora_path or ''}"
        if cache_key not in self._cache:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
            load_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "device_map": "auto" if n_gpu else None,
                "trust_remote_code": True,
            }
            # Optional 4-bit load for large MoE models (capability pilots).
            # Modes (set one via slurm env):
            #   HBAC_BNB_GPU_ONLY=1     → device_map=auto, no max_memory cap
            #   HBAC_BNB_CPU_OFFLOAD=1  → allow CPU spill (slower, more reliable)
            if os.environ.get("HBAC_LOAD_IN_4BIT", "").lower() in {"1", "true", "yes"}:
                from transformers import BitsAndBytesConfig

                allow_cpu = os.environ.get("HBAC_BNB_CPU_OFFLOAD", "").lower() in {
                    "1",
                    "true",
                    "yes",
                }
                gpu_only = os.environ.get("HBAC_BNB_GPU_ONLY", "").lower() in {
                    "1",
                    "true",
                    "yes",
                }
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=allow_cpu,
                )
                load_kwargs.pop("torch_dtype", None)
                if n_gpu >= 1 and not gpu_only:
                    per_gpu = os.environ.get("HBAC_MAX_MEMORY_PER_GPU", "").strip()
                    if not per_gpu:
                        per_gpu = {
                            i: f"{max(int(torch.cuda.get_device_properties(i).total_memory * 0.90 / (1024**3)), 1)}GiB"
                            for i in range(n_gpu)
                        }
                    else:
                        per_gpu = {i: per_gpu for i in range(n_gpu)}
                    load_kwargs["max_memory"] = dict(per_gpu)
                    if allow_cpu:
                        load_kwargs["max_memory"]["cpu"] = os.environ.get(
                            "HBAC_MAX_MEMORY_CPU", "128GiB"
                        )
                # gpu_only: omit max_memory entirely — accelerate uses full VRAM
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            if config.lora_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, config.lora_path)
            self._cache[cache_key] = (tok, model)
        self.tokenizer, self.model = self._cache[cache_key]

    def _input_device(self):
        """First device holding model weights (works with device_map='auto')."""
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            first = next(iter(self.model.hf_device_map.values()))
            if isinstance(first, int):
                import torch

                return torch.device(f"cuda:{first}")
            return first
        try:
            return self.model.device
        except Exception:
            return next(self.model.parameters()).device

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        import torch

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        start = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device())
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.config.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.config.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        new_ids = output[0][inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s, 1)[0]

        latency_ms = (time.perf_counter() - start) * 1000
        prompt_tokens = int(inputs.input_ids.shape[1])
        completion_tokens = int(new_ids.shape[0])
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw={"model": self.config.model},
        )

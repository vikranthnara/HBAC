from __future__ import annotations

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
        if not config.base_url:
            config = config.model_copy(update={"base_url": "http://localhost:8000/v1"})
        super().__init__(config)


class TransformersBackend(LLMBackend):
    """Local HuggingFace inference (no vLLM server)."""

    _cache: dict[str, tuple[Any, Any]] = {}

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = config.model
        if model_id not in self._cache:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self._cache[model_id] = (tok, model)
        self.tokenizer, self.model = self._cache[model_id]

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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
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

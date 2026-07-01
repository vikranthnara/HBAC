"""vLLM-backed rollout for GRPO group sampling (Phase 3b)."""

from __future__ import annotations

import os
from pathlib import Path

from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend, LLMResponse


def make_vllm_backend(model: str | None = None) -> LLMBackend:
    """Create vLLM backend from HBAC_LLM_* env vars."""
    model = model or os.environ.get("HBAC_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    base = os.environ.get("HBAC_LLM_BASE_URL", "http://localhost:8000/v1")
    return LLMBackend.from_config(
        LLMConfig(provider="vllm", model=model, base_url=base, api_key="EMPTY")
    )


def sample_completions(
    prompts: list[str],
    *,
    model: str | None = None,
    max_tokens: int = 256,
    num_samples: int = 4,
) -> list[list[str]]:
    """Generate G completions per prompt via vLLM OpenAI-compatible API."""
    llm = make_vllm_backend(model)
    out: list[list[str]] = []
    for prompt in prompts:
        group: list[str] = []
        for _ in range(num_samples):
            resp = llm.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            group.append(resp.text)
        out.append(group)
    return out


def smoke_vllm() -> bool:
    """Return True if vLLM server responds."""
    try:
        llm = make_vllm_backend(os.environ.get("HBAC_LLM_MODEL", "gpt2"))
        r = llm.complete([{"role": "user", "content": "ping"}], max_tokens=8)
        return bool(r.text or r.completion_tokens >= 0)
    except Exception:
        return False

"""TRL-based LLM GRPO training for Phase 3b."""

from __future__ import annotations

import json
from pathlib import Path

from hbac.core.trajectory import TrajectoryStore
from hbac.training.dataset import find_oracle_paths


def load_sft_prompts(oracle_root: Path, limit: int = 200) -> list[dict]:
    """Build prompt/completion pairs from successful oracles for GRPO."""
    records: list[dict] = []
    for path in find_oracle_paths(oracle_root):
        for traj in TrajectoryStore(path).load_successful():
            messages = []
            for step in traj.steps:
                if step.llm_response:
                    messages.append(step.llm_response)
            if not messages:
                continue
            records.append(
                {
                    "prompt": (
                        f"Task {traj.task_id} ({traj.benchmark}): "
                        f"budget={traj.budget} tokens. Respond with valid tool JSON.\n"
                    ),
                    "completion": messages[0],
                    "reward": 1.0 if traj.success else 0.0,
                    "task_id": traj.task_id,
                }
            )
            if len(records) >= limit:
                return records
    return records


def reward_from_completion(completion: str, reference: str, base_reward: float) -> float:
    """Token overlap proxy reward when env execution unavailable."""
    ref = reference.strip()
    comp = completion.strip()
    if not ref:
        return 0.0
    overlap = sum(1 for a, b in zip(comp, ref) if a == b) / max(len(ref), 1)
    return base_reward * (0.5 + 0.5 * overlap)


def _sft_batch_loss(model, tokenizer, row: dict):
    """Causal LM loss on completion tokens only."""
    import torch

    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids
    full_text = row["prompt"] + row["completion"]
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    labels = enc.input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    outputs = model(**enc, labels=labels)
    return outputs.loss


def train_with_trl(
    prompts: list[dict],
    model_name: str,
    out_dir: Path,
    *,
    lora_rank: int = 16,
    grpo_groups: int = 4,
    epochs: int = 1,
    max_samples: int = 32,
) -> list[dict]:
    """Run TRL GRPO on CUDA; SFT+LoRA fallback elsewhere."""
    try:
        import torch

        if not torch.cuda.is_available():
            return _train_fallback(
                prompts,
                model_name,
                out_dir,
                lora_rank,
                epochs,
                max_samples,
                error="cuda_unavailable",
            )
        return _train_trl_grpo(
            prompts, model_name, out_dir, lora_rank, grpo_groups, epochs, max_samples
        )
    except Exception as exc:
        return _train_fallback(
            prompts, model_name, out_dir, lora_rank, epochs, max_samples, error=str(exc)
        )


def _train_trl_grpo(
    prompts: list[dict],
    model_name: str,
    out_dir: Path,
    lora_rank: int,
    grpo_groups: int,
    epochs: int,
    max_samples: int,
) -> list[dict]:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    out_dir.mkdir(parents=True, exist_ok=True)
    subset = prompts[:max_samples]
    ds = Dataset.from_list([{"prompt": p["prompt"]} for p in subset])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(r=lora_rank, lora_alpha=32, task_type="CAUSAL_LM"),
    )

    ref_by_prompt = {p["prompt"]: p for p in subset}

    def reward_fn(samples: list[str], prompts_in=None, **kwargs) -> list[float]:
        out: list[float] = []
        for i, comp in enumerate(samples):
            prompt = prompts_in[i] if prompts_in else ""
            ref_row = ref_by_prompt.get(prompt, subset[i % len(subset)])
            out.append(reward_from_completion(comp, ref_row["completion"], ref_row["reward"]))
        return out

    config = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=min(grpo_groups, 4),
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        logging_steps=1,
        max_completion_length=256,
        num_generations=grpo_groups,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    trainer.train()
    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")

    return [{"epoch": epochs, "method": "trl_grpo", "samples": len(subset)}]


def _train_fallback(
    prompts: list[dict],
    model_name: str,
    out_dir: Path,
    lora_rank: int,
    epochs: int,
    max_samples: int,
    error: str = "",
) -> list[dict]:
    """CPU/MPS-safe SFT warm-start with LoRA (GRPO group sampling deferred to Rivanna CUDA)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    subset = prompts[:max_samples]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=32, task_type="CAUSAL_LM"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    log: list[dict] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for row in subset:
            loss = _sft_batch_loss(model, tokenizer, row)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += float(loss.detach())
        log.append(
            {
                "epoch": epoch + 1,
                "method": "sft_fallback",
                "loss": epoch_loss / max(len(subset), 1),
            }
        )

    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")
    (out_dir / "fallback_config.json").write_text(
        json.dumps({"model": model_name, "method": "sft_fallback", "trl_error": error}),
        encoding="utf-8",
    )
    return log

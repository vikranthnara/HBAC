"""TRL-based LLM GRPO / SFT training for Phase 3b (tool-aware rewards)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from hbac.training.grpo_records import load_grpo_step_records, records_to_trl_prompts
from hbac.training.tool_reward import tool_aware_reward

TrainingMode = Literal["sft_only", "grpo_only", "sft_then_grpo"]
RewardMode = Literal["overlap", "tool_aware"]


def load_sft_prompts(oracle_root: Path, limit: int = 200) -> list[dict]:
    """Legacy flat prompts — prefer load_grpo_step_records."""
    records = load_grpo_step_records(oracle_root, limit=limit)
    return [
        {
            "prompt": r.get("prompt", ""),
            "completion": r["completion"],
            "reward": r.get("reward_weight", 1.0),
            "task_id": r["task_id"],
        }
        for r in records
    ]


def reward_from_completion(completion: str, reference: str, base_reward: float) -> float:
    overlap = sum(1 for a, b in zip(completion.strip(), reference.strip()) if a == b) / max(
        len(reference.strip()), 1
    )
    return base_reward * (0.5 + 0.5 * overlap)


def _sft_batch_loss(model, tokenizer, row: dict, *, max_length: int = 768):
    import torch

    device = next(model.parameters()).device
    prompt = row.get("prompt") or ""
    completion = row["completion"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_text = prompt + completion
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    labels = enc["input_ids"].clone()
    labels[:, : len(prompt_ids)] = -100
    outputs = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"), labels=labels)
    return outputs.loss


def sft_warmstart_lora(
    records: list[dict],
    model,
    tokenizer,
    *,
    epochs: int = 2,
    max_length: int = 768,
) -> list[dict]:
    import torch

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    log: list[dict] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for row in records:
            loss = _sft_batch_loss(model, tokenizer, row, max_length=max_length)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += float(loss.detach())
        log.append(
            {
                "epoch": epoch + 1,
                "method": "sft_warmstart",
                "loss": epoch_loss / max(len(records), 1),
                "samples": len(records),
            }
        )
    return log


def train_with_trl(
    oracle_root: Path,
    model_name: str,
    out_dir: Path,
    *,
    lora_rank: int = 16,
    grpo_groups: int = 4,
    grpo_epochs: int = 1,
    sft_epochs: int = 0,
    max_samples: int = 32,
    training_mode: TrainingMode = "sft_then_grpo",
    reward_mode: RewardMode = "tool_aware",
    max_completion_length: int = 384,
) -> list[dict]:
    """Run SFT and/or TRL GRPO on CUDA; SFT fallback on CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return _train_fallback(
                oracle_root,
                model_name,
                out_dir,
                lora_rank,
                sft_epochs or grpo_epochs or 1,
                max_samples,
                error="cuda_unavailable",
            )
        return _train_cuda(
            oracle_root,
            model_name,
            out_dir,
            lora_rank=lora_rank,
            grpo_groups=grpo_groups,
            grpo_epochs=grpo_epochs,
            sft_epochs=sft_epochs,
            max_samples=max_samples,
            training_mode=training_mode,
            reward_mode=reward_mode,
            max_completion_length=max_completion_length,
        )
    except Exception as exc:
        return _train_fallback(
            oracle_root,
            model_name,
            out_dir,
            lora_rank,
            sft_epochs or grpo_epochs or 1,
            max_samples,
            error=str(exc),
        )


def _train_cuda(
    oracle_root: Path,
    model_name: str,
    out_dir: Path,
    *,
    lora_rank: int,
    grpo_groups: int,
    grpo_epochs: int,
    sft_epochs: int,
    max_samples: int,
    training_mode: TrainingMode,
    reward_mode: RewardMode,
    max_completion_length: int,
) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    raw = load_grpo_step_records(oracle_root, limit=max_samples)
    if not raw:
        raise ValueError("No oracle step records for LLM training")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = records_to_trl_prompts(raw, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    run_sft = training_mode in {"sft_only", "sft_then_grpo"}
    run_grpo = training_mode in {"grpo_only", "sft_then_grpo"}

    if run_sft and sft_epochs > 0:
        log.extend(sft_warmstart_lora(records, model, tokenizer, epochs=sft_epochs))

    if run_grpo and grpo_epochs > 0:
        log.extend(
            _train_trl_grpo(
                records,
                model,
                tokenizer,
                out_dir,
                grpo_groups=grpo_groups,
                epochs=grpo_epochs,
                reward_mode=reward_mode,
                max_completion_length=max_completion_length,
            )
        )

    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "training_mode": training_mode,
                "reward_mode": reward_mode,
                "max_samples": len(records),
                "sft_epochs": sft_epochs,
                "grpo_epochs": grpo_epochs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return log


def _train_trl_grpo(
    records: list[dict],
    model,
    tokenizer,
    out_dir: Path,
    *,
    grpo_groups: int,
    epochs: int,
    reward_mode: RewardMode,
    max_completion_length: int,
) -> list[dict]:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    subset = records
    ds = Dataset.from_list([{"prompt": r["prompt"]} for r in subset])
    ref_by_prompt = {r["prompt"]: r for r in subset}

    def reward_fn(prompts: list[str], completions: list[str], completion_ids=None, **kwargs) -> list[float]:
        out: list[float] = []
        for i, comp in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            ref_row = ref_by_prompt.get(prompt, subset[i % len(subset)])
            if reward_mode == "tool_aware":
                br = tool_aware_reward(
                    comp,
                    reference_completion=ref_row["completion"],
                    reference_tool=ref_row.get("reference_tool"),
                    success_weight=float(ref_row.get("reward_weight", 1.0)),
                )
                out.append(br.total)
            else:
                out.append(
                    reward_from_completion(comp, ref_row["completion"], float(ref_row.get("reward_weight", 1.0)))
                )
        return out

    config = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=min(grpo_groups, 2),
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        logging_steps=1,
        max_completion_length=max_completion_length,
        num_generations=grpo_groups,
        beta=0.04,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    trainer.train()
    return [{"epoch": epochs, "method": "trl_grpo", "samples": len(subset), "reward_mode": reward_mode}]


def _train_fallback(
    oracle_root: Path,
    model_name: str,
    out_dir: Path,
    lora_rank: int,
    epochs: int,
    max_samples: int,
    error: str = "",
) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_grpo_step_records(oracle_root, limit=max_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = records_to_trl_prompts(raw, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=lora_rank * 2, task_type="CAUSAL_LM"))

    log = sft_warmstart_lora(records, model, tokenizer, epochs=epochs)
    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")
    (out_dir / "fallback_config.json").write_text(
        json.dumps({"model": model_name, "method": "sft_fallback", "trl_error": error}),
        encoding="utf-8",
    )
    return log

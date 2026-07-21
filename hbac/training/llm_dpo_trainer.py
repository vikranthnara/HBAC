"""DPO LoRA training for tool-JSON capability (Phase 3c)."""

from __future__ import annotations

import json
from pathlib import Path

from hbac.training.capability import build_dpo_pairs
from hbac.training.grpo_records import load_grpo_step_records, records_to_trl_prompts
from hbac.training.llm_grpo_trainer import sft_warmstart_lora


def train_dpo_lora(
    oracle_root: Path,
    model_name: str,
    out_dir: Path,
    *,
    lora_rank: int = 16,
    max_pairs: int = 400,
    epochs: int = 2,
    beta: float = 0.1,
    learning_rate: float = 5e-7,
    sft_epochs: int = 0,
    reject_modes: tuple[str, ...] = ("wrong_tool", "invalid_json"),
    benchmark: str | None = None,
    oversample_benchmark: str | None = None,
    oversample_factor: int = 1,
    exclude_task_ids: set[str] | None = None,
    exclude_benchmarks: tuple[str, ...] | None = None,
) -> list[dict]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure prompts use chat template before pair construction
    raw = load_grpo_step_records(
        oracle_root, limit=max(max_pairs, 200), successful_only=True, benchmark=benchmark
    )
    trl_records = records_to_trl_prompts(raw, tokenizer)
    pairs = build_dpo_pairs(
        oracle_root,
        limit=max_pairs,
        tokenizer=tokenizer,
        reject_modes=reject_modes,
        benchmark=benchmark,
        oversample_benchmark=oversample_benchmark,
        oversample_factor=oversample_factor,
        exclude_task_ids=exclude_task_ids,
        exclude_benchmarks=exclude_benchmarks,
    )
    if not pairs:
        raise ValueError("No DPO pairs built from oracles")

    ds = Dataset.from_list(
        [{"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected} for p in pairs]
    )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
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

    log: list[dict] = []
    if sft_epochs > 0:
        log.extend(sft_warmstart_lora(trl_records, model, tokenizer, epochs=sft_epochs))

    config = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        beta=beta,
        logging_steps=1,
        max_length=768,
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")
    (out_dir / "dpo_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "pairs": len(pairs),
                "epochs": epochs,
                "sft_epochs": sft_epochs,
                "beta": beta,
                "reject_modes": list(reject_modes),
                "method": "dpo_capability",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.append({"method": "dpo", "pairs": len(pairs), "epochs": epochs, "sft_epochs": sft_epochs})
    return log

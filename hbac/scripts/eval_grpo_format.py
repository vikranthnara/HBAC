"""Evaluate LoRA adapter: tool-JSON parse rate on oracle prompts (proxy for live pass@1)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hbac.training.grpo_records import load_grpo_step_records, records_to_trl_prompts
from hbac.training.tool_reward import extract_tool_json, tool_aware_reward

app = typer.Typer(help="Format eval for GRPO LoRA adapters")


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="Base model"),
    lora_path: str | None = typer.Option(None, help="PEFT adapter dir"),
    limit: int = typer.Option(80, help="Max eval prompts"),
    max_new_tokens: int = typer.Option(256, help="Generation cap"),
    output: str = typer.Option("results/grpo_format_eval.json", help="Metrics JSON"),
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = load_grpo_step_records(Path(oracle_path), limit=limit)
    if not records:
        raise typer.Exit("No oracle records")

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    trl_records = records_to_trl_prompts(records, tokenizer)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
    )
    if lora_path:
        from peft import PeftModel

        model_obj = PeftModel.from_pretrained(base, lora_path)
        label = f"lora:{lora_path}"
    else:
        model_obj = base
        label = "base"

    valid = 0
    tool_match = 0
    rewards: list[float] = []
    for row in trl_records:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model_obj.device)
        with torch.no_grad():
            out = model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        parsed = extract_tool_json(text) is not None
        valid += int(parsed)
        if parsed and row.get("reference_tool"):
            p = extract_tool_json(text)
            if p and str(p.get("tool_name") or p.get("action")) == row["reference_tool"]:
                tool_match += 1
        br = tool_aware_reward(
            text,
            reference_completion=row["completion"],
            reference_tool=row.get("reference_tool"),
            success_weight=float(row.get("reward_weight", 1.0)),
        )
        rewards.append(br.total)

    n = len(trl_records)
    tool_match_given_valid = tool_match / max(valid, 1)
    report = {
        "model": label,
        "n": n,
        "valid_json_rate": valid / n,
        "tool_name_match_rate": tool_match / n,
        "tool_name_match_given_valid_json": tool_match_given_valid,
        "mean_tool_reward": sum(rewards) / n,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()

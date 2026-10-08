"""Part A Step 2: bfloat16 baseline for Mistral-7B-v0.1.

The GPU run belongs on the BITS A100. `check` validates the prompts and
settings on any machine and does not download the model.

Tomorrow, from the repo root on the lab pod:

    python src/baseline.py run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
ARCHITECTURE_PATH = REPORTS_DIR / "baseline_architecture.json"
OUTPUTS_PATH = REPORTS_DIR / "baseline_outputs.csv"
DESIGN_CHECK_PATH = REPORTS_DIR / "baseline_design_check.json"

MODEL_ID = "mistralai/Mistral-7B-v0.1"
TORCH_DTYPE_NAME = "bfloat16"
MAX_NEW_TOKENS = 150
DECODING = "greedy"
PROMPT_COUNT = 3

BASELINE_PROMPTS = [
    {
        "id": "malaria_uncomplicated_pf",
        "corpus_target": "domain_corpus/who_malaria_2024-11-30.txt",
        "instruction": (
            "What treatment does the malaria guidance recommend for uncomplicated "
            "Plasmodium falciparum malaria in adults and in children weighing at least 25 kg?"
        ),
    },
    {
        "id": "sepsis_iv_antibiotics",
        "corpus_target": "domain_corpus/nice_ng253_sepsis.txt",
        "instruction": (
            "In an adult with suspected sepsis, when should intravenous antibiotics be started, "
            "and what should happen if the source of infection is still uncertain?"
        ),
    },
    {
        "id": "hypertension_first_line",
        "corpus_target": "domain_corpus/nice_ng136_hypertension.txt",
        "instruction": (
            "What first-line antihypertensive treatment does the adult hypertension guidance "
            "recommend, and how does age change that choice?"
        ),
    },
]


def format_prompt(instruction: str) -> str:
    """Same instruction block as later training. There is no system prompt."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def design_errors() -> list[str]:
    errors = []
    if MODEL_ID != "mistralai/Mistral-7B-v0.1":
        errors.append(f"model id is {MODEL_ID}")
    if TORCH_DTYPE_NAME != "bfloat16":
        errors.append(f"dtype is {TORCH_DTYPE_NAME}")
    if MAX_NEW_TOKENS != 150:
        errors.append(f"max_new_tokens is {MAX_NEW_TOKENS}")
    if DECODING != "greedy":
        errors.append(f"decoding is {DECODING}")
    if len(BASELINE_PROMPTS) != PROMPT_COUNT:
        errors.append(f"prompt count is {len(BASELINE_PROMPTS)}")
    seen = set()
    for prompt in BASELINE_PROMPTS:
        prompt_id = prompt["id"]
        if prompt_id in seen:
            errors.append(f"repeated prompt id {prompt_id}")
        seen.add(prompt_id)
        instruction = prompt["instruction"].strip()
        if len(instruction.split()) < 8:
            errors.append(f"{prompt_id} instruction is too short")
        formatted = format_prompt(instruction)
        if not formatted.startswith("### Instruction:\n"):
            errors.append(f"{prompt_id} is missing the instruction block")
        if not formatted.endswith("### Response:\n"):
            errors.append(f"{prompt_id} is missing the response block")
        if "system prompt" in formatted.lower() or formatted.lower().startswith("system:"):
            errors.append(f"{prompt_id} contains a system prompt")
        target = ROOT / prompt["corpus_target"]
        if not target.is_file():
            errors.append(f"{prompt_id} corpus target is missing: {prompt['corpus_target']}")
    return errors


def check_design() -> dict:
    errors = design_errors()
    report = {
        "model_id": MODEL_ID,
        "torch_dtype": TORCH_DTYPE_NAME,
        "decoding": DECODING,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_count": len(BASELINE_PROMPTS),
        "prompt_ids": [prompt["id"] for prompt in BASELINE_PROMPTS],
        "system_prompt": False,
        "tokenizer": f"AutoTokenizer.from_pretrained({MODEL_ID!r})",
        "gpu_run_complete": ARCHITECTURE_PATH.is_file() and OUTPUTS_PATH.is_file(),
        "design_check_passed": not errors,
        "errors": errors,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_CHECK_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("Baseline design check failed: " + "; ".join(errors))
    return report


def architecture_report(model, tokenizer) -> dict:
    config = model.config
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "model_id": MODEL_ID,
        "torch_dtype": TORCH_DTYPE_NAME,
        "parameter_count": parameter_count,
        "num_hidden_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "hidden_size_meaning": "Width of each token representation in the residual stream.",
        "vocab_size": int(config.vocab_size),
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_source": MODEL_ID,
    }
    if report["tokenizer_vocab_size"] != report["vocab_size"]:
        raise RuntimeError(
            f"Tokenizer vocab {report['tokenizer_vocab_size']} does not match "
            f"model vocab {report['vocab_size']}"
        )
    import torch

    report["gpu_name"] = torch.cuda.get_device_name(0)
    report["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return report


def load_baseline():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Baseline generation needs the A100. CUDA is not available in this process.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def generate_baseline(model, tokenizer) -> list[dict]:
    import torch

    rows = []
    for prompt in BASELINE_PROMPTS:
        formatted = format_prompt(prompt["instruction"])
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            started = time.perf_counter()
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            elapsed = time.perf_counter() - started
        prompt_length = inputs["input_ids"].shape[1]
        new_ids = output_ids[0, prompt_length:]
        generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        new_tokens = int(new_ids.shape[0])
        rows.append(
            {
                "prompt_id": prompt["id"],
                "corpus_target": prompt["corpus_target"],
                "instruction": prompt["instruction"],
                "formatted_prompt": formatted,
                "output": generated,
                "new_tokens": new_tokens,
                "elapsed_seconds": round(elapsed, 4),
                "tokens_per_second": round(new_tokens / elapsed, 4) if elapsed else 0,
                "model_id": MODEL_ID,
                "torch_dtype": TORCH_DTYPE_NAME,
                "decoding": DECODING,
                "max_new_tokens": MAX_NEW_TOKENS,
            }
        )
    return rows


def save_outputs(architecture: dict, rows: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHITECTURE_PATH.write_text(json.dumps(architecture, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0])
    with OUTPUTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_baseline() -> dict:
    check_design()
    model, tokenizer = load_baseline()
    architecture = architecture_report(model, tokenizer)
    rows = generate_baseline(model, tokenizer)
    save_outputs(architecture, rows)
    return {"architecture": architecture, "outputs": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Part A Step 2 baseline")
    parser.add_argument("stage", choices=["check", "run"])
    args = parser.parse_args()
    if args.stage == "check":
        report = check_design()
        print(
            "baseline design check passed "
            f"prompts={report['prompt_count']} dtype={report['torch_dtype']} "
            f"decoding={report['decoding']} max_new_tokens={report['max_new_tokens']} "
            f"gpu_run_complete={report['gpu_run_complete']}"
        )
    else:
        result = run_baseline()
        architecture = result["architecture"]
        print(
            "baseline architecture "
            f"parameters={architecture['parameter_count']} "
            f"layers={architecture['num_hidden_layers']} "
            f"hidden_size={architecture['hidden_size']} "
            f"vocab_size={architecture['vocab_size']} "
            f"gpu={architecture['gpu_name']}"
        )
        for row in result["outputs"]:
            print(f"{row['prompt_id']} new_tokens={row['new_tokens']} tokens_per_second={row['tokens_per_second']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED {error}", file=sys.stderr)
        raise

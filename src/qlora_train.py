"""Part B2: QLoRA instruction fine-tuning of Mistral-7B-v0.1 with Adapter B.

    python src/qlora_train.py check   # tokenizer-only design check, any machine, no GPU
    python src/qlora_train.py run     # trains on the GPU, writes the adapter and reports

Plain transformers Trainer + peft + bitsandbytes (the libraries the brief names).
Each text is `### Instruction / ### Response` block + answer + EOS, with the loss on the
answer and EOS only. The pad token is <unk>, so the EOS label is never masked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Pods ship TensorFlow with Keras 3, which transformers cannot import. This is a PyTorch-only job.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import MODEL_ID, format_prompt  # noqa: E402
from instruction_dataset import DISCLAIMER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "instruction" / "train.jsonl"
EVAL_PATH = ROOT / "data" / "instruction" / "eval.jsonl"
ADAPTER_DIR = ROOT / "adapters" / "adapter_b"
CHECKPOINT_DIR = ROOT / "checkpoints" / "adapter_b"
REPORTS_DIR = ROOT / "reports"
DESIGN_PATH = REPORTS_DIR / "qlora_design_check.json"
TRAINING_PATH = REPORTS_DIR / "qlora_training.json"
LOSS_PATH = REPORTS_DIR / "qlora_loss.csv"
SMOKE_PATH = REPORTS_DIR / "qlora_smoke.json"

# Every training setting in one place; the report and the notebook print this dict.
HYPERPARAMS = {
    "adapter": "B",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "quantization": "4-bit NF4, double quantization, bfloat16 compute",
    "epochs": 3,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 4,
    "learning_rate": 2e-4,
    "lr_scheduler": "linear",
    "warmup_ratio": 0.03,
    "optimizer": "paged_adamw_32bit",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "max_seq_length": 256,
    "seed": 42,
    "precision": "bf16",
    "gradient_checkpointing": True,
    "gradient_checkpointing_use_reentrant": False,
    "loss_on": "response and EOS only (prompt tokens masked with -100)",
}
SMOKE_MAX_NEW_TOKENS = 150


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.unk_token  # not EOS: the EOS label must stay in the loss
    tokenizer.padding_side = "right"
    return tokenizer


def build_example(tokenizer, instruction: str, response: str) -> dict:
    """Prompt ids (BOS + instruction block) are masked; response ids + EOS are the labels."""
    prompt_ids = tokenizer(format_prompt(instruction)).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    return {
        "input_ids": prompt_ids + response_ids,
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def build_dataset(tokenizer, rows: list[dict]) -> list[dict]:
    examples = [build_example(tokenizer, r["instruction"], r["response"]) for r in rows]
    too_long = [len(e["input_ids"]) for e in examples if len(e["input_ids"]) > HYPERPARAMS["max_seq_length"]]
    if too_long:
        raise ValueError(f"{len(too_long)} examples exceed max_seq_length: {too_long}")
    return examples


def make_collator(tokenizer):
    import torch

    def collate(batch: list[dict]) -> dict:
        width = max(len(item["input_ids"]) for item in batch)
        pad = tokenizer.pad_token_id
        input_ids = [item["input_ids"] + [pad] * (width - len(item["input_ids"])) for item in batch]
        labels = [item["labels"] + [-100] * (width - len(item["labels"])) for item in batch]
        mask = [[1] * len(item["input_ids"]) + [0] * (width - len(item["input_ids"])) for item in batch]
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    return collate


def check_design() -> dict:
    """Tokenizer-only checks. Raises on the first broken assumption."""
    tokenizer = load_tokenizer()
    train, evals = read_jsonl(TRAIN_PATH), read_jsonl(EVAL_PATH)
    assert (len(train), len(evals)) == (40, 10), (len(train), len(evals))
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad must differ from EOS"
    examples = build_dataset(tokenizer, train + evals)
    for row, ex in zip(train + evals, examples):
        assert ex["input_ids"][-1] == tokenizer.eos_token_id, "every text must end with EOS"
        assert ex["labels"][-1] == tokenizer.eos_token_id, "the EOS label must not be masked"
        assert ex["input_ids"][0] == tokenizer.bos_token_id, "BOS first"
        n_masked = sum(1 for x in ex["labels"] if x == -100)
        assert n_masked == len(tokenizer(format_prompt(row["instruction"])).input_ids), "only the prompt is masked"
        decoded = tokenizer.decode([x for x in ex["labels"] if x != -100], skip_special_tokens=True)
        assert decoded.strip() == row["response"].strip(), "labels must decode back to the response"
        assert decoded.strip().endswith(DISCLAIMER)
    lengths = [len(e["input_ids"]) for e in examples]
    train_steps = -(-len(train) // HYPERPARAMS["effective_batch_size"]) * HYPERPARAMS["epochs"]
    report = {
        "design_check_passed": True,
        "model_id": MODEL_ID,
        "train_examples": len(train),
        "eval_examples": len(evals),
        "max_tokens": max(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths), 1),
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "optimizer_steps": train_steps,
        "response_tokens_in_loss_train": sum(sum(1 for x in e["labels"] if x != -100) for e in examples[:40]),
        "hyperparameters": HYPERPARAMS,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    DESIGN_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_quantized_model():
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=config, device_map={"": 0})


def smoke_generate(model, tokenizer, rows: list[dict]) -> list[dict]:
    """Greedy generation on training and eval instructions (never the three B3 prompts)."""
    import torch

    model.eval()
    results = []
    for tag, row in rows:
        inputs = tokenizer(format_prompt(row["instruction"]), return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=SMOKE_MAX_NEW_TOKENS, do_sample=False,
                                 use_cache=True, pad_token_id=tokenizer.pad_token_id)
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        results.append({
            "split": tag,
            "id": row["id"],
            "instruction": row["instruction"],
            "output": text,
            "new_tokens": int(new_ids.shape[0]),
            "stopped_on_eos": bool(new_ids[-1].item() == tokenizer.eos_token_id),
            "ends_with_disclaimer": text.endswith(DISCLAIMER),
        })
    return results


def run_training(out_dir: Path | None = None) -> dict:
    """Train and write the adapter and reports. `out_dir` redirects every output (used by the reproducibility check)."""
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    hp = HYPERPARAMS
    adapter_dir = ADAPTER_DIR if out_dir is None else out_dir / "adapter"
    reports_dir = REPORTS_DIR if out_dir is None else out_dir
    set_seed(hp["seed"])
    tokenizer = load_tokenizer()
    train_rows, eval_rows = read_jsonl(TRAIN_PATH), read_jsonl(EVAL_PATH)
    train_ds, eval_ds = build_dataset(tokenizer, train_rows), build_dataset(tokenizer, eval_rows)

    model = load_quantized_model()
    base_gb = torch.cuda.memory_allocated() / 1e9
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=hp["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": hp["gradient_checkpointing_use_reentrant"]})
    model.config.use_cache = False  # training with gradient checkpointing; generation turns it back on
    model = get_peft_model(model, LoraConfig(
        r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=hp["lora_dropout"],
        target_modules=hp["target_modules"], bias="none", task_type="CAUSAL_LM"))
    trainable, total = model.get_nb_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR if out_dir is None else out_dir / "ckpt"),
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["per_device_batch_size"],
        per_device_eval_batch_size=hp["per_device_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"],
        lr_scheduler_type=hp["lr_scheduler"],
        warmup_ratio=hp["warmup_ratio"],
        optim=hp["optimizer"],
        weight_decay=hp["weight_decay"],
        max_grad_norm=hp["max_grad_norm"],
        bf16=True,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",
        seed=hp["seed"],
        data_seed=hp["seed"],
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=make_collator(tokenizer))

    eval_before = trainer.evaluate()["eval_loss"]  # LoRA B matrices start at zero: this is the 4-bit base model
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    seconds = time.perf_counter() - started
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    history = trainer.state.log_history
    eval_by_epoch = [{"epoch": h["epoch"], "step": h["step"], "eval_loss": h["eval_loss"]}
                     for h in history if "eval_loss" in h]
    train_log = [{"step": h["step"], "epoch": h["epoch"], "loss": h["loss"], "learning_rate": h["learning_rate"],
                  "grad_norm": h.get("grad_norm")} for h in history if "loss" in h]

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    smoke = smoke_generate(model, tokenizer, [("train", train_rows[0]), ("eval", eval_rows[0])])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / SMOKE_PATH.name).write_text(json.dumps(smoke, indent=2), encoding="utf-8")

    with (reports_dir / LOSS_PATH.name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "loss", "learning_rate", "grad_norm"])
        writer.writeheader()
        writer.writerows(train_log)

    report = {
        "model_id": MODEL_ID,
        "hyperparameters": hp,
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "optimizer_steps": trainer.state.global_step,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": round(100 * trainable / total, 4),
        "eval_loss_before_training": eval_before,
        "eval_loss_by_epoch": eval_by_epoch,
        "final_train_loss_logged": train_log[-1]["loss"],
        "mean_train_loss": result.training_loss,
        "train_seconds": round(seconds, 1),
        "base_model_gpu_gb": round(base_gb, 2),
        "peak_gpu_memory_gb": round(peak_gb, 2),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__, "python": sys.version.split()[0]},
        "bitsandbytes": __import__("bitsandbytes").__version__,
        "adapter_dir": str(adapter_dir.relative_to(ROOT)),
        "adapter_files": sorted(p.name for p in adapter_dir.iterdir()),
        "inputs_sha256": {"train.jsonl": sha256(TRAIN_PATH), "eval.jsonl": sha256(EVAL_PATH),
                          "qlora_train.py": sha256(Path(__file__))},
        "smoke_generate": smoke,
    }
    (reports_dir / TRAINING_PATH.name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def reproducibility_check(saved: dict) -> dict:
    """Train again with the same seed into checkpoints/repro and compare with the saved report."""
    import shutil

    out = ROOT / "checkpoints" / "repro"
    shutil.rmtree(out, ignore_errors=True)
    again = run_training(out_dir=out)
    shutil.rmtree(out, ignore_errors=True)
    first, second = saved["eval_loss_by_epoch"], again["eval_loss_by_epoch"]
    return {
        "eval_loss_before": [saved["eval_loss_before_training"], again["eval_loss_before_training"]],
        "eval_loss_by_epoch_first_run": [round(e["eval_loss"], 4) for e in first],
        "eval_loss_by_epoch_second_run": [round(e["eval_loss"], 4) for e in second],
        "max_abs_eval_loss_difference": round(max(abs(a["eval_loss"] - b["eval_loss"]) for a, b in zip(first, second)), 5),
        "smoke_outputs_identical": [a["output"] == b["output"]
                                    for a, b in zip(saved["smoke_generate"], again["smoke_generate"])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["check", "run"])
    args = parser.parse_args()
    out = check_design() if args.command == "check" else run_training()
    print(json.dumps({k: v for k, v in out.items() if k not in ("smoke_generate", "hyperparameters")}, indent=2))


if __name__ == "__main__":
    main()

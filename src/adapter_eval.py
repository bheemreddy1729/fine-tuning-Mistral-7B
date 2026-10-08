"""Part B3: baseline versus Adapter B on the three fixed Part A prompts.

    python src/adapter_eval.py check   # rubric quotes exist in the corpus, adapter files present
    python src/adapter_eval.py run     # GPU: baseline again, adapter on bf16, adapter on 4-bit

Same settings as the Part A baseline: greedy, max_new_tokens=150, no system prompt, same
instruction block. The primary comparison is bf16 base + adapter (same dtype as the baseline, so
the only change is fine-tuning). The adapter was trained on the 4-bit base, so a 4-bit run is
reported next to it. Every output is scored against a small rubric whose facts are quoted from
the corpus file for that prompt.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")  # pods ship TensorFlow/Keras 3, which transformers cannot import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import BASELINE_PROMPTS, MAX_NEW_TOKENS, MODEL_ID, OUTPUTS_PATH, format_prompt  # noqa: E402
from instruction_dataset import DISCLAIMER, numbers_in  # noqa: E402
from qlora_train import ADAPTER_DIR, TRAINING_PATH, sha256  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
CORPUS_DIR = ROOT / "domain_corpus"
COMPARISON_PATH = REPORTS_DIR / "baseline_vs_adapter.csv"
SCORES_PATH = REPORTS_DIR / "adapter_eval_scores.csv"
EVAL_JSON_PATH = REPORTS_DIR / "adapter_eval.json"
SYSTEMS = ["baseline", "adapter_bf16", "adapter_4bit"]
WARMUP_PROMPT = "### Instruction:\nWhat is hypertension?\n\n### Response:\n"

# Ground truth per prompt: `quotes` must appear in the corpus file (checked by `check`), `facts` are
# regexes for what a correct answer contains, `flags` are regexes for things the corpus does not support.
RUBRIC = {
    "malaria_uncomplicated_pf": {
        "quotes": ["should be treated with one of the following ACTs", "no longer generally recommended"],
        "facts": [
            ("act", "names artemisinin-based combination therapy (ACT)", r"artemisinin[- ]based combination|\bACTs?\b"),
            ("al", "artemether-lumefantrine", r"artemether[- ]lumefantrine|\bAL\b"),
            ("asaq", "artesunate-amodiaquine", r"artesunate[- ]amodiaquine|AS\s*\+\s*AQ"),
            ("asmq", "artesunate-mefloquine", r"artesunate[- ]mefloquine|ASMQ"),
            ("dhap", "dihydroartemisinin-piperaquine", r"dihydroartemisinin[- ]piperaquine|\bDHA?P\b"),
            ("assp", "artesunate + sulfadoxine-pyrimethamine", r"artesunate\s*(\+|plus|and|with)?\s*sulfadoxine|AS\s*\+\s*SP"),
            ("aspy", "artesunate-pyronaridine", r"artesunate[- ]pyronaridine|ASPY"),
        ],
        "flags": [
            ("single_drug_only", "presents one ACT as the recommendation (corpus lists six options)", None),
            ("quinine_second_line", "presents a quinine regimen as the second line (corpus: quinine plus doxycycline or clindamycin is no longer generally recommended)",
             r"second[- ]line[^.]{0,80}quinine|quinine[^.]{0,80}second[- ]line"),
        ],
    },
    "sepsis_iv_antibiotics": {
        "quotes": ["within 1 hour of calculating", "for whom the source of infection is unknown"],
        "facts": [
            ("one_hour", "IV antibiotics within 1 hour", r"within\s+(1|one)\s+hour"),
            ("risk_based", "tied to high risk / NEWS2 / risk category", r"high[- ]risk|NEWS2|risk category|risk level"),
            ("broad_spectrum", "broad-spectrum antibiotics when the source is unknown", r"broad[- ]spectrum"),
            ("source_specific", "switch to source-specific antibiotics once the source is confirmed", r"source[- ]specific|once the source|source is (identified|confirmed|known)|targeted"),
        ],
        "flags": [("external_guideline", "attributes the advice to a guideline that is not the corpus source (NICE)",
                   r"Surviving Sepsis|\bSSC\b|SEP-1|Sepsis-3")],
    },
    "hypertension_first_line": {
        "quotes": ["Offer an ACE inhibitor or an ARB to adults starting step 1",
                   "Offer a calcium-channel blocker (CCB) to adults starting step 1"],
        "facts": [
            ("acei_arb", "ACE inhibitor or ARB at step 1", r"ACE inhibitor|angiotensin[- ]converting|\bARBs?\b|angiotensin II receptor|\bACEIs?\b"),
            ("ccb", "calcium-channel blocker at step 1", r"calcium[- ]channel|\bCCBs?\b"),
            ("age_55", "age 55 as the dividing line", r"\b55\b"),
            ("family_origin", "Black African or African-Caribbean family origin", r"Black African|African[–-]Caribbean"),
            ("t2dm", "type 2 diabetes", r"type 2 diabetes|T2DM"),
        ],
        "flags": [
            ("thiazide_first_line", "lists a thiazide diuretic as first line (NICE: only if a CCB is not tolerated or in heart failure)", r"thiazide"),
            ("wrong_age_rule", "gives a thiazide for people aged 55 or over (NICE: a calcium-channel blocker)", r"55[^.]{0,80}thiazide"),
        ],
    },
}
OPTION_IDS = ["al", "asaq", "asmq", "dhap", "assp", "aspy"]


def squash(text: str) -> str:
    return " ".join(text.replace("’", "'").split())


def check_design() -> dict:
    """Quotes exist in the corpus, adapter files exist, prompts match the Part A baseline."""
    problems = []
    for prompt in BASELINE_PROMPTS:
        corpus = squash((ROOT / prompt["corpus_target"]).read_text(encoding="utf-8"))
        for quote in RUBRIC[prompt["id"]]["quotes"]:
            if squash(quote) not in corpus:
                problems.append(f"{prompt['id']}: quote not found in {prompt['corpus_target']}: {quote}")
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if not (ADAPTER_DIR / name).is_file():
            problems.append(f"missing {ADAPTER_DIR / name}")
    saved = list(csv.DictReader(OUTPUTS_PATH.open(encoding="utf-8", newline="")))
    if [r["prompt_id"] for r in saved] != [p["id"] for p in BASELINE_PROMPTS]:
        problems.append("baseline_outputs.csv prompt ids differ from BASELINE_PROMPTS")
    if problems:
        raise SystemExit("design check failed:\n  " + "\n  ".join(problems))
    return {"design_check_passed": True, "prompts": [p["id"] for p in BASELINE_PROMPTS],
            "rubric_quotes_found": sum(len(r["quotes"]) for r in RUBRIC.values())}


def score_output(prompt: dict, text: str) -> dict:
    """Rubric hits, numbers not found in the prompt's corpus file, and structural flags."""
    rubric = RUBRIC[prompt["id"]]
    hits = [fid for fid, _, pattern in rubric["facts"] if re.search(pattern, text, re.I)]
    flags = [fid for fid, _, pattern in rubric["flags"] if pattern and re.search(pattern, text, re.I)]
    if prompt["id"] == "malaria_uncomplicated_pf" and sum(o in hits for o in OPTION_IDS) == 1:
        flags.append("single_drug_only")
    source = (ROOT / prompt["corpus_target"]).read_text(encoding="utf-8")
    ungrounded = sorted(numbers_in(text) - numbers_in(source) - numbers_in(prompt["instruction"]), key=float)
    return {
        "facts_hit": hits,
        "facts_total": len(rubric["facts"]),
        "flags": flags,
        "ungrounded_numbers": ungrounded,
        "has_disclaimer": text.strip().endswith(DISCLAIMER),
        "extra_sections": len(re.findall(r"^#{2,3}\s*\w+", text, re.M)),
        "has_references_section": bool(re.search(r"#+\s*References", text)),
    }


def generate(model, tokenizer) -> list[dict]:
    """Greedy, 150 new tokens, no system prompt: the same settings as the Part A baseline."""
    import torch

    warm = tokenizer(WARMUP_PROMPT, return_tensors="pt").to(model.device)
    with torch.inference_mode():  # one short generation so CUDA start-up is not billed to prompt 1
        model.generate(**warm, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    rows = []
    for prompt in BASELINE_PROMPTS:
        inputs = tokenizer(format_prompt(prompt["instruction"]), return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        rows.append({
            "prompt_id": prompt["id"], "output": text, "new_tokens": int(new_ids.shape[0]),
            "tokens_per_second": round(int(new_ids.shape[0]) / elapsed, 3),
            "stopped_on_eos": bool(new_ids[-1].item() == tokenizer.eos_token_id),
        })
    return rows


def free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def run_eval() -> dict:
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qlora_train import load_quantized_model

    design = check_design()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token  # same as the baseline; batch size 1, so padding never happens
    results: dict[str, list[dict]] = {}

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                                                low_cpu_mem_usage=True).eval()
    results["baseline"] = generate(base, tokenizer)
    adapted = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
    results["adapter_bf16"] = generate(adapted, tokenizer)
    bf16_peak = torch.cuda.max_memory_allocated() / 1e9
    del adapted, base
    free_gpu()

    torch.cuda.reset_peak_memory_stats()
    quant = load_quantized_model().eval()
    adapted4 = PeftModel.from_pretrained(quant, ADAPTER_DIR).eval()
    results["adapter_4bit"] = generate(adapted4, tokenizer)
    peak_4bit = torch.cuda.max_memory_allocated() / 1e9

    saved = {r["prompt_id"]: r for r in csv.DictReader(OUTPUTS_PATH.open(encoding="utf-8", newline=""))}
    same_as_part_a = {r["prompt_id"]: r["output"].split() == saved[r["prompt_id"]]["output"].split()
                      for r in results["baseline"]}
    scores = []
    for system in SYSTEMS:
        for prompt, row in zip(BASELINE_PROMPTS, results[system]):
            scores.append({"system": system, "prompt_id": prompt["id"], **row | score_output(prompt, row["output"])})

    comparison = []
    for i, prompt in enumerate(BASELINE_PROMPTS):
        b, a, a4 = (results[s][i] for s in SYSTEMS)
        comparison.append({
            "prompt_id": prompt["id"],
            "baseline_output": saved[prompt["id"]]["output"], "adapter_output": a["output"],
            "baseline_tps": b["tokens_per_second"], "adapter_tps": a["tokens_per_second"],
            "baseline_new_tokens": b["new_tokens"], "adapter_new_tokens": a["new_tokens"],
            "adapter_stopped_on_eos": a["stopped_on_eos"],
            "adapter_has_disclaimer": a["output"].strip().endswith(DISCLAIMER),
            "adapter_4bit_output": a4["output"], "adapter_4bit_tps": a4["tokens_per_second"],
            "baseline_tps_part_a": saved[prompt["id"]]["tokens_per_second"],
        })
    REPORTS_DIR.mkdir(exist_ok=True)
    with COMPARISON_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    score_fields = ["system", "prompt_id", "new_tokens", "tokens_per_second", "stopped_on_eos", "facts_hit", "facts_total",
                    "flags", "ungrounded_numbers", "has_disclaimer", "extra_sections", "has_references_section"]
    with SCORES_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields, extrasaction="ignore")
        writer.writeheader()
        for s in scores:
            writer.writerow({**s, "facts_hit": "|".join(s["facts_hit"]), "flags": "|".join(s["flags"]),
                             "ungrounded_numbers": "|".join(s["ungrounded_numbers"])})
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    report = {
        "design": design,
        "settings": {"decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "system_prompt": None,
                     "prompt_format": "### Instruction / ### Response"},
        "baseline_matches_part_a_outputs": same_as_part_a,
        "peak_gpu_gb_bf16_run": round(bf16_peak, 2),
        "peak_gpu_gb_4bit_run": round(peak_4bit, 2),
        "eval_loss_before_after_training": [training["eval_loss_before_training"], training["eval_loss_by_epoch"][-1]["eval_loss"]],
        "adapter_sha256": sha256(ADAPTER_DIR / "adapter_model.safetensors"),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "results": results,
        "scores": scores,
    }
    EVAL_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["check", "run"])
    args = parser.parse_args()
    out = check_design() if args.command == "check" else run_eval()
    print(json.dumps({k: v for k, v in out.items() if k not in ("results", "scores")}, indent=2))


if __name__ == "__main__":
    main()

"""Part B2 cells for assignment_1b.ipynb. add(md, code) appends them. Cell sources are raw strings."""


def add(md, code):
    md(r"""
## B2: QLoRA fine-tuning with one adapter (Adapter B) (2 marks)

**Requirements.** Apply QLoRA with `transformers`, `peft` and `bitsandbytes` on a 4-bit base model; train *one* of the three adapter configurations on the instruction training set with an SFT setup that matches the
model's instruction format; report batch size, learning rate, epochs or steps, and maximum sequence length.

**Choices and why.**
- **Adapter B** (r = 16, alpha = 32, `q_proj` and `v_proj`): the balanced option. Adapter A (r = 8) may under-fit; Adapter C (r = 32 plus `o_proj`) has more capacity than 40 training rows can use and would over-fit and cost more VRAM.
  It trains 6,815,744 parameters, 0.094 % of the model.
- **4-bit NF4 with double quantisation and bfloat16 compute** (the QLoRA recipe, and the same configuration Part C uses for the quantisation benchmark). The base weights stay frozen in 4-bit; only the LoRA matrices train.
- **Plain `transformers.Trainer`**, no extra SFT library: the brief names only transformers, peft and bitsandbytes.
- **Format.** Mistral-7B-v0.1 has **no chat template**, so each example is `### Instruction:` ... `### Response:` + answer + end-of-sequence token, exactly the block used for the Part A baseline.
- **Loss on the answer only.** The prompt tokens are labelled `-100`, so the model is not trained to reproduce the question. The pad token is `<unk>`, *not* EOS: with pad = EOS the collator would mask the EOS label and the adapter would
  never learn to stop (the baseline never stopped by itself).
- **Max sequence length 256**: the longest example is 239 tokens (B1), so nothing is truncated.
- **Epochs 3, learning rate 2e-4, effective batch 4 (1 x 4 accumulation)**: 40 rows give 10 optimizer steps per epoch, 30 in total; 2e-4 is the usual LoRA learning rate; more epochs would only memorise 40 rows.
""")

    code(r"""
import gc, tempfile
os.environ.setdefault("USE_TF", "0")                     # PyTorch-only job
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
import peft, bitsandbytes

# free the bfloat16 baseline model from Part A so the 4-bit model gets a clean GPU
for name in ("model",):
    if name in globals(): del globals()[name]
gc.collect(); torch.cuda.empty_cache()
print(f"GPU memory held by this process after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

HP = {"adapter": "B", "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"],
      "quantization": "4-bit NF4, double quantization, bfloat16 compute", "epochs": 3, "per_device_batch_size": 1,
      "gradient_accumulation_steps": 4, "effective_batch_size": 4, "learning_rate": 2e-4, "lr_scheduler": "linear",
      "warmup_ratio": 0.03, "optimizer": "paged_adamw_32bit", "weight_decay": 0.0, "max_grad_norm": 1.0, "max_seq_length": 256,
      "seed": 42, "precision": "bf16", "gradient_checkpointing": True,
      "loss_on": "response and EOS only (prompt tokens masked with -100)"}
print("Hyperparameters (the four the brief asks for are batch size, learning rate, epochs/steps, max sequence length):")
pd.DataFrame({"value": {k: str(v) for k, v in HP.items()}})
""")

    md(r"""
### B2.1 Training examples and loss masking (checked before any training)

Each text is `BOS + instruction block + answer + EOS`. The assertions below prove what the design claims: every example ends with EOS and its label is *not* masked, only the prompt is masked, the unmasked labels decode back to exactly the
training answer, the answer ends with the disclaimer, pad differs from EOS, and nothing exceeds 256 tokens.
""")
    code(r"""
def load_tok():
    t = AutoTokenizer.from_pretrained(MODEL_ID)
    t.pad_token = t.unk_token            # not EOS: the EOS label must stay in the loss
    t.padding_side = "right"
    return t

fmt_prompt = lambda instruction: f"### Instruction:\n{instruction}\n\n### Response:\n"
def build_example(t, instruction, response):
    p = t(fmt_prompt(instruction)).input_ids
    r = t(response, add_special_tokens=False).input_ids + [t.eos_token_id]
    return {"input_ids": p + r, "labels": [-100] * len(p) + r}

def make_collator(t):
    def collate(batch):
        w = max(len(b["input_ids"]) for b in batch)
        return {"input_ids": torch.tensor([b["input_ids"] + [t.pad_token_id] * (w - len(b["input_ids"])) for b in batch]),
                "labels": torch.tensor([b["labels"] + [-100] * (w - len(b["labels"])) for b in batch]),
                "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (w - len(b["input_ids"])) for b in batch])}
    return collate

t_chk = load_tok()
assert t_chk.pad_token_id != t_chk.eos_token_id
exs = [build_example(t_chk, r["instruction"], r["response"]) for r in train + evals]
for r, e in zip(train + evals, exs):
    assert e["input_ids"][0] == t_chk.bos_token_id and e["input_ids"][-1] == t_chk.eos_token_id and e["labels"][-1] == t_chk.eos_token_id
    assert sum(x == -100 for x in e["labels"]) == len(t_chk(fmt_prompt(r["instruction"])).input_ids), "only the prompt is masked"
    assert t_chk.decode([x for x in e["labels"] if x != -100], skip_special_tokens=True).strip() == r["response"].strip()
    assert r["response"].strip().endswith(DISCLAIMER) and len(e["input_ids"]) <= HP["max_seq_length"]
lens = [len(e["input_ids"]) for e in exs]
print(f"all {len(exs)} examples pass: end with EOS (label kept), prompt masked, labels decode to the answer, disclaimer present, max length {max(lens)} <= 256")
e0 = exs[0]
print(f"\nexample p01: {len(e0['input_ids'])} tokens, {sum(x == -100 for x in e0['labels'])} masked prompt tokens, {sum(x != -100 for x in e0['labels'])} answer tokens in the loss")
print("last 8 label ids:", e0["labels"][-8:], "| EOS id:", t_chk.eos_token_id)
""")

    md(r"""
### B2.2 Train Adapter B (A100)

The function loads the base model in 4-bit NF4, measures the **eval loss of the untuned model** on the 10 held-out pairs (the LoRA B matrices start at zero, so this is the 4-bit base), trains for 3 epochs with an eval pass after each, saves the adapter,
and runs a smoke generation on one training and one held-out instruction (never the three fixed prompts, which are kept for B3). The same function is run a second time later to check reproducibility.
""")
    code(r"""
BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

def smoke_generate(m, t, items):
    m.eval(); out = []
    for tag, row in items:
        enc = t(fmt_prompt(row["instruction"]), return_tensors="pt").to(m.device)
        with torch.inference_mode():
            ids = m.generate(**enc, max_new_tokens=150, do_sample=False, use_cache=True, pad_token_id=t.pad_token_id)
        new = ids[0, enc["input_ids"].shape[1]:]
        text = t.decode(new, skip_special_tokens=True).strip()
        out.append({"split": tag, "id": row["id"], "instruction": row["instruction"], "output": text, "new_tokens": int(new.shape[0]),
                    "stopped_on_eos": bool(new[-1].item() == t.eos_token_id), "ends_with_disclaimer": text.endswith(DISCLAIMER)})
    return out

def train_adapter(save_dir=None):
    set_seed(HP["seed"])
    t = load_tok()
    train_ds = [build_example(t, r["instruction"], r["response"]) for r in train]
    eval_ds = [build_example(t, r["instruction"], r["response"]) for r in evals]
    t0 = time.perf_counter()
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=BNB, device_map={"": 0})
    load_s = time.perf_counter() - t0
    base_gb = torch.cuda.memory_allocated() / 1e9
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(r=HP["lora_r"], lora_alpha=HP["lora_alpha"], lora_dropout=HP["lora_dropout"],
                                     target_modules=HP["target_modules"], bias="none", task_type="CAUSAL_LM"))
    trainable, total = m.get_nb_trainable_parameters()
    args = TrainingArguments(output_dir=str(Path(tempfile.gettempdir()) / "qlora_out"), num_train_epochs=HP["epochs"],
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4, learning_rate=HP["learning_rate"],
        lr_scheduler_type="linear", warmup_ratio=HP["warmup_ratio"], optim="paged_adamw_32bit", weight_decay=0.0, max_grad_norm=1.0,
        bf16=True, logging_steps=1, eval_strategy="epoch", save_strategy="no", seed=HP["seed"], data_seed=HP["seed"],
        report_to="none", remove_unused_columns=False)
    tr = Trainer(model=m, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=make_collator(t))
    eval_before = tr.evaluate()["eval_loss"]
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    res = tr.train()
    seconds = time.perf_counter() - t0; peak_gb = torch.cuda.max_memory_allocated() / 1e9
    hist = tr.state.log_history
    rep = {"eval_loss_before_training": eval_before,
           "eval_loss_by_epoch": [{"epoch": h["epoch"], "step": h["step"], "eval_loss": h["eval_loss"]} for h in hist if "eval_loss" in h],
           "train_log": [{"step": h["step"], "epoch": h["epoch"], "loss": h["loss"], "learning_rate": h["learning_rate"], "grad_norm": h.get("grad_norm")} for h in hist if "loss" in h],
           "optimizer_steps": tr.state.global_step, "mean_train_loss": res.training_loss, "train_seconds": round(seconds, 1),
           "model_load_seconds": round(load_s, 1), "base_model_gpu_gb": round(base_gb, 2), "peak_gpu_memory_gb": round(peak_gb, 2),
           "trainable_parameters": trainable, "total_parameters": total, "trainable_percent": round(100 * trainable / total, 4),
           "gpu": torch.cuda.get_device_name(0),
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__}}
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True); m.save_pretrained(save_dir); t.save_pretrained(save_dir)
        rep["adapter_files"] = sorted(p.name for p in save_dir.iterdir())
    rep["smoke_generate"] = smoke_generate(m, t, [("train", train[0]), ("eval", evals[0])])
    del tr, m; gc.collect(); torch.cuda.empty_cache()
    return rep

ADAPTER_DIR = ROOT / "adapters" / "adapter_b"
run1 = train_adapter(save_dir=ADAPTER_DIR)
print(f"trainable parameters: {run1['trainable_parameters']:,} of {run1['total_parameters']:,} ({run1['trainable_percent']} %)")
print(f"optimizer steps: {run1['optimizer_steps']} | training time: {run1['train_seconds']} s | model load: {run1['model_load_seconds']} s")
print(f"GPU memory: base model in 4-bit {run1['base_model_gpu_gb']} GB, peak during training {run1['peak_gpu_memory_gb']} GB")
print(f"eval loss: untuned 4-bit base {run1['eval_loss_before_training']:.4f} -> " + ", ".join(f"epoch {e['epoch']:.0f}: {e['eval_loss']:.4f}" for e in run1["eval_loss_by_epoch"]))
print("adapter files:", run1["adapter_files"])
""")

    md(r"""
### B2.3 Loss curves
""")
    code(r"""
tl = pd.DataFrame(run1["train_log"]); el = pd.DataFrame(run1["eval_loss_by_epoch"])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(tl["step"], tl["loss"], marker="o", ms=3, color="#4c78a8"); ax[0].set_title("Training loss per optimizer step"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
xs = [0] + list(el["step"]); ys = [run1["eval_loss_before_training"]] + list(el["eval_loss"])
ax[1].plot(xs, ys, marker="o", color="#e45756"); ax[1].set_title("Eval loss (10 held-out pairs); step 0 = untuned 4-bit base"); ax[1].set_xlabel("step")
for x, y in zip(xs, ys): ax[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
plt.tight_layout(); plt.show()
(REPORTS_LAB / "qlora_loss.csv").write_text(tl.to_csv(index=False), encoding="utf-8")
print(f"training loss: first step {tl['loss'].iloc[0]:.3f}, last step {tl['loss'].iloc[-1]:.3f}, mean {run1['mean_train_loss']:.3f}")
tl.round(4)
""")

    md(r"""
### B2.4 Smoke generation

Greedy, `max_new_tokens = 150`, one training instruction (p01) and one unseen held-out instruction (p05).
""")
    code(r"""
for s in run1["smoke_generate"]:
    print(f"[{s['split']} {s['id']}] new_tokens={s['new_tokens']} stopped_on_eos={s['stopped_on_eos']} ends_with_disclaimer={s['ends_with_disclaimer']}")
    print("INSTRUCTION:", s["instruction"]); print("OUTPUT     :", s["output"], "\n")
print("TRAINED ANSWER for p01:", next(r for r in train if r["id"] == "p01")["response"])
""")

    md(r"""
### B2.5 Reproducibility: the same seed, data and code, run a second time on the lab

GPU kernels are not bit-deterministic, so the check reports how close the two runs are, not whether they are bit-identical.
""")
    code(r"""
run2 = train_adapter(save_dir=None)
e1, e2 = run1["eval_loss_by_epoch"], run2["eval_loss_by_epoch"]
print("eval loss before training :", round(run1["eval_loss_before_training"], 4), round(run2["eval_loss_before_training"], 4))
print("eval loss by epoch, run 1 :", [round(e["eval_loss"], 4) for e in e1])
print("eval loss by epoch, run 2 :", [round(e["eval_loss"], 4) for e in e2])
print("max |difference| in eval loss:", round(max(abs(a["eval_loss"] - b["eval_loss"]) for a, b in zip(e1, e2)), 5))
print("smoke outputs identical   :", [a["output"] == b["output"] for a, b in zip(run1["smoke_generate"], run2["smoke_generate"])])
print(f"run 2 time {run2['train_seconds']} s, peak GPU {run2['peak_gpu_memory_gb']} GB")
""")

    code(r"""
# save the lab reports of this run (reports_lab/)
(REPORTS_LAB / "qlora_training.json").write_text(json.dumps({"hyperparameters": HP, "run1": run1, "run2": {k: v for k, v in run2.items() if k != "train_log"}}, indent=2, default=str), encoding="utf-8")
(REPORTS_LAB / "qlora_smoke.json").write_text(json.dumps(run1["smoke_generate"], indent=2), encoding="utf-8")
print("saved reports_lab/qlora_training.json, qlora_loss.csv and qlora_smoke.json")
""")

    md(r"""
### Inference: B2

**Hyperparameters (the four the brief asks for).** Batch size 1 per device with 4 gradient-accumulation steps (effective batch 4); learning rate 2e-4 with linear decay and 3 % warm-up; 3 epochs = 30 optimizer steps; maximum sequence length 256. LoRA r = 16, alpha = 32, dropout 0.05 on `q_proj` and `v_proj`.

**What the numbers say.** Eval loss on the 10 held-out rows fell from **2.126** (untuned 4-bit base) to 1.505 after epoch 1, 1.425 after epoch 2 and 1.426 after epoch 3, a **32.9 %** drop; almost all of it comes in the first epoch. Epoch 3 is slightly *above* epoch 2 while the training loss keeps
falling (2.10 at step 1, 1.08 at step 30), the early sign of memorising a 40-row set, so three epochs is the right ceiling and more would not help. Ten eval rows give a noisy estimate, so the size of the drop is more trustworthy than the third decimal.

**What the adapter learned: the response shape.** Both smoke outputs open with "Per WHO malaria guidelines", end with the exact disclaimer and **stop on their own** (114 and 133 new tokens, both below the 150 limit, `stopped_on_eos = True`). The baseline in Part A never stopped and
filled the 150-token budget with invented `### Evidence` and `### References` sections. The EOS token and the answer-only loss did what they were designed to do.

**What it did not learn: the facts.** For the *training* row p01 the adapter gives a primaquine regimen (a single 0.25 mg/kg dose on day 1, then 0.15 mg/kg daily for 12 days) that differs from the answer it was trained on (a 7 mg/kg total dose, as 0.5 mg/kg for 14 days or 1 mg/kg for 7 days), and adds a clause about
resident status that is not in the trained answer. For the unseen row p05 it produces tablet counts per weight band and a maximum number of tablets that B3 must check against the corpus. The per-token loss is still about 1.4, so the model is fluent and confident but not reliable on numbers. Forty
pairs teach style and the habit of citing a guideline; they do not add clinical knowledge. B3 therefore compares **claims against the corpus**, not only the format.

**Cost.** The 4-bit base model occupies 4.13 GB (against about 14.5 GB in bfloat16) and training peaks at 5.32 GB, so QLoRA of a 7B model fits in a few GB of VRAM; training took 53 s, plus about 105 s to load and quantise the model.

**Reproducibility.** A second run with the same seed on the lab gives eval losses identical to four decimals (maximum difference 0.0) and identical smoke outputs, so training is deterministic here and the numbers above can be trusted to the digits shown. Training time (about 53 s) is dominated by per-step overhead of a 40-row job with batch 1 and gradient checkpointing on a 4-CPU pod, not by GPU compute.

**Output.** The adapter is saved in `adapters/adapter_b/` (weights are not committed to git) and is loaded in B3.
""")

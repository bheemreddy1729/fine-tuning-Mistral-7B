"""Part C cells for assignment_1b.ipynb. add_intro / add_c1 / add_c2 / add_c3 append them; add() appends all. Cell sources are raw strings."""


def add_intro(md, code):
    md(r"""
---
# Part C: Inference optimization and production metrics (8 marks)

Three inference techniques are applied to the **base model loaded in bfloat16 in Part A** (the adapter is not used here, except in the optional extension at the end), with the prompts fixed so that differences are attributable to the inference setting.
All runs use the same instruction block as before, `max_new_tokens = 150`, and the three fixed domain prompts of Part A (malaria, sepsis, hypertension) in C1 and C2; C3 adds seven further corpus-based prompts.
Timing on this shared A100 varies by about 25 % between runs, so every speed is measured **after a warm-up over repeated runs**, and conclusions rest on ratios.

| Step | Technique | Marks |
|---|---|---|
| C1 | Decoding strategies: greedy, beam search, top-k, top-p, temperature | 3 |
| C2 | Speculative decoding with a smaller draft model | 2 |
| C3 | 4-bit quantization and production cost per 1M tokens | 3 |
""")


def add_c1(md, code):
    md(r"""
## C1: Decoding strategies and sampling (3 marks)

Decoding decides how the next token is chosen from the model's probabilities, so the same model and prompt can give very different text. Seven settings are run on each of the three prompts (`max_new_tokens = 150`), recording the generated text and the generation speed:

| Setting | Configuration | Expected behaviour |
|---|---|---|
| Greedy | `do_sample=False` | Fastest, deterministic, one answer |
| Beam search | `num_beams=4` | More coherent, about 4x the compute per step |
| Top-k | `top_k=50`, temperature 1 | Diversity limited to the 50 likeliest tokens |
| Top-p (nucleus) | `top_p=0.9`, temperature 1 | Vocabulary adapts to the model's confidence at each step |
| Temperature 0.3 / 0.7 / 1.2 | pure temperature scaling (top-k and top-p switched off) | Low = conservative, high = diverse and riskier |

Sampling is seeded (seed 42 for the table). Speed is `new tokens / seconds`, the **mean of 2 identical runs** (same seed, so the same text) after a warm-up. Quality is scored with the B3 rubric (corpus facts mentioned, claims the corpus does not support, numbers
that appear nowhere in the source file, invented `### ...` sections). **Diversity** is measured for the five sampled settings from three extra seeds per prompt: *distinct-2* is the share of unique word bigrams across the four samples of a prompt (higher = more varied), and *repetition* is the share of
repeated word 4-grams inside one sample (higher = more self-repeating). Greedy and beam are deterministic, so they have a single sample.
""")

    code(r"""
tok_c = AutoTokenizer.from_pretrained(MODEL_ID)
tok_c.pad_token = tok_c.eos_token
for nm in ("model", "bf16", "base_c"):
    if nm in globals(): del globals()[nm]
gc.collect(); torch.cuda.empty_cache()
mem0 = torch.cuda.memory_allocated()
t0 = time.perf_counter()
base_c = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
W_BF16 = (torch.cuda.memory_allocated() - mem0) / 1e9
print(f"base model (bfloat16) loaded in {time.perf_counter() - t0:.0f} s, resident weights {W_BF16:.2f} GB")

STRATEGIES = {
    "greedy":    dict(do_sample=False),
    "beam4":     dict(do_sample=False, num_beams=4, early_stopping=True),
    "top_k50":   dict(do_sample=True, top_k=50, top_p=1.0, temperature=1.0),
    "top_p0.9":  dict(do_sample=True, top_k=0, top_p=0.9, temperature=1.0),
    "temp0.3":   dict(do_sample=True, top_k=0, top_p=1.0, temperature=0.3),
    "temp0.7":   dict(do_sample=True, top_k=0, top_p=1.0, temperature=0.7),
    "temp1.2":   dict(do_sample=True, top_k=0, top_p=1.0, temperature=1.2),
}
SAMPLED = [k for k, v in STRATEGIES.items() if v["do_sample"]]
COLS = {"greedy": "Greedy", "beam4": "Beam search", "top_k50": "Top-K", "top_p0.9": "Top-P", "temp0.3": "Temp 1 (0.3)", "temp0.7": "Temp 2 (0.7)", "temp1.2": "Temp 3 (1.2)"}

def run_one(model, t, prompt, kwargs, seed=None, **extra):
    enc = t(fmt_prompt(prompt["instruction"]), return_tensors="pt").to(model.device)
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(**enc, max_new_tokens=150, pad_token_id=t.pad_token_id, **kwargs, **extra)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    new = ids[0, enc["input_ids"].shape[1]:]
    return {"text": t.decode(new, skip_special_tokens=True).strip(), "new_tokens": int(new.shape[0]), "seconds": dt}

with torch.inference_mode():      # warm-up so CUDA start-up is not billed to the first measurement
    base_c.generate(**tok_c(fmt_prompt("What is hypertension?"), return_tensors="pt").to(base_c.device), max_new_tokens=8, do_sample=False, pad_token_id=tok_c.pad_token_id)
print("strategies:", {k: {a: b for a, b in v.items()} for k, v in STRATEGIES.items()})
""")

    code(r"""
REPEATS = 2
c1_rows = []
for p in PROMPTS:
    for name, kw in STRATEGIES.items():
        runs = [run_one(base_c, tok_c, p, kw, seed=42) for _ in range(REPEATS)]          # same seed: identical text, repeated for timing
        assert all(r["text"] == runs[0]["text"] for r in runs), "seeded runs must repeat exactly"
        tps = sum(r["new_tokens"] / r["seconds"] for r in runs) / REPEATS
        sc = score_output(p, runs[0]["text"])
        c1_rows.append({"prompt_id": p["id"], "strategy": name, "text": runs[0]["text"], "new_tokens": runs[0]["new_tokens"], "tok_per_s": round(tps, 2),
                        "facts": len(sc["facts_hit"]), "facts_total": sc["facts_total"], "flags": len(sc["flags"]), "ungrounded_numbers": len(sc["ungrounded_numbers"]),
                        "invented_sections": sc["extra_sections"], "flag_names": "|".join(sc["flags"])})
c1 = pd.DataFrame(c1_rows)
print(f"{len(c1)} runs done ({len(PROMPTS)} prompts x {len(STRATEGIES)} settings, each timed {REPEATS}x)")
""")

    md(r"""
### C1.1 Generated outputs (all 21 runs)
""")
    code(r"""
for p in PROMPTS:
    print("=" * 110); print(f"PROMPT [{p['id']}]: {p['instruction']}"); print("=" * 110)
    for name in STRATEGIES:
        r = c1[(c1.prompt_id == p["id"]) & (c1.strategy == name)].iloc[0]
        print(f"\n--- {COLS[name]}  [{r['new_tokens']} tokens, {r['tok_per_s']} tok/s, facts {r['facts']}/{r['facts_total']}, flags {r['flags']}, ungrounded numbers {r['ungrounded_numbers']}, invented sections {r['invented_sections']}]")
        print(r["text"])
    print()
""")

    md(r"""
### C1.2 Comparison table (the brief's table: prompt x strategy)

Each cell shows the first words of the output; the full texts are printed above.
""")
    code(r"""
short = lambda s, n=110: (s.replace("\n", " ")[:n] + ("..." if len(s) > n else ""))
tbl = pd.DataFrame({COLS[n]: [short(c1[(c1.prompt_id == p["id"]) & (c1.strategy == n)].iloc[0]["text"]) for p in PROMPTS] for n in STRATEGIES}, index=[f"Domain Prompt {i + 1} ({p['id']})" for i, p in enumerate(PROMPTS)])
pd.set_option("display.max_colwidth", 120)
display(tbl.T)
""")

    md(r"""
### C1.3 Speed and quality per strategy
""")
    code(r"""
agg = c1.groupby("strategy", sort=False).agg(tok_per_s=("tok_per_s", "mean"), new_tokens=("new_tokens", "mean"), facts=("facts", "sum"), flags=("flags", "sum"),
                                             ungrounded_numbers=("ungrounded_numbers", "sum"), invented_sections=("invented_sections", lambda s: int((s > 0).sum())))
agg["speed_vs_greedy"] = (agg["tok_per_s"] / agg.loc["greedy", "tok_per_s"]).round(2)
agg["tok_per_s"] = agg["tok_per_s"].round(2); agg["new_tokens"] = agg["new_tokens"].round(0)
agg.index = [COLS[i] for i in agg.index]
print("facts = rubric facts mentioned over the 3 prompts (16 in total); flags = claims the corpus does not support; invented_sections = prompts with a ### section")
display(agg)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].bar(agg.index, agg["tok_per_s"], color="#4c78a8"); ax[0].set_title("Generation speed (tokens/s, mean of 3 prompts)"); ax[0].tick_params(axis="x", rotation=40)
ax[1].bar(agg.index, agg["facts"], color="#54a24b"); ax[1].set_title("Rubric facts mentioned (of 16)"); ax[1].tick_params(axis="x", rotation=40)
plt.tight_layout(); plt.show()
""")

    md(r"""
### C1.4 Diversity of the sampled settings

Three extra seeds (1, 2, 3) give four samples per prompt and setting (with the seed-42 sample).
""")
    code(r"""
def bigrams(text):
    w = re.findall(r"[a-z0-9']+", text.lower()); return list(zip(w, w[1:]))
def repetition(text):
    w = re.findall(r"[a-z0-9']+", text.lower()); g = list(zip(w, w[1:], w[2:], w[3:]))
    return 0.0 if not g else 1 - len(set(g)) / len(g)

div_rows = []
for p in PROMPTS:
    for name in SAMPLED:
        samples = [c1[(c1.prompt_id == p["id"]) & (c1.strategy == name)].iloc[0]["text"]]
        samples += [run_one(base_c, tok_c, p, STRATEGIES[name], seed=s)["text"] for s in (1, 2, 3)]
        allb = [b for t in samples for b in bigrams(t)]
        div_rows.append({"prompt_id": p["id"], "strategy": name, "distinct2": len(set(allb)) / len(allb), "repetition": sum(repetition(t) for t in samples) / len(samples),
                         "distinct_samples": len(set(samples))})
div = pd.DataFrame(div_rows)
rep_det = c1[c1.strategy.isin(["greedy", "beam4"])].assign(repetition=lambda d: d["text"].apply(repetition)).groupby("strategy")["repetition"].mean()
dsum = div.groupby("strategy", sort=False).agg(distinct2=("distinct2", "mean"), repetition=("repetition", "mean"), distinct_samples_of_4=("distinct_samples", "mean")).round(3)
dsum.index = [COLS[i] for i in dsum.index]
print("repetition of the deterministic settings (one sample): greedy %.3f, beam search %.3f" % (rep_det["greedy"], rep_det["beam4"]))
display(dsum)
""")

    md(r"""
### C1.5 Deployment recommendation (100 words)

I would deploy greedy decoding. A clinical lookup needs one repeatable answer, and greedy matched the other settings on speed (about 31 tokens/s; sampling is within 3 %), tied top-p for the most rubric facts (5 of 16) and never varies. Beam search was 10 % slower and said the first-line choice "does not change based on age". Temperature 0.3 invented a "2017 ACC/AHA" source, top-p an ACT rule tied to "> 3 years" of use, and at 1.2 the sepsis answer became nonsense ("Newton's Athletic Gold (medicated athletic tape)"). Every setting still invents sections or sources, so greedy needs grounding and the disclaimer.
""")

    md(r"""
### Inference: C1

**Speed.** At batch size 1 generation is limited by per-token overhead (Python and kernel launches on a 4-CPU pod), not by arithmetic or memory bandwidth: reading 14.5 GB of weights at an A100's roughly 2 TB/s would take about 7 ms, while the measured 32 ms per token is several times that (C2 measures the same effect from the draft side). So the rule that picks the token costs almost nothing: greedy and all five sampled settings run within about 3 % of each other (about 31 tokens/s). **Beam search is only about 10 % slower** (28.1 against 31.2 tokens/s) although it follows four candidate
sequences, because the four beams share one batched forward pass; it is not four times slower on a GPU. Absolute speeds differ between stages of the notebook because the load on the shared A100 changes; as the intro says, read ratios, not single values. The base model almost never emits its end-of-sequence token, so every run used all 150 tokens and the speeds are directly comparable.

**Quality against the corpus.** No setting gives a correct, complete, corpus-grounded answer: the base model answers from memory, names the wrong source (the Surviving Sepsis Campaign, "2017 ACC/AHA") and invents `### Evidence` / `### References` sections. Within that limit the settings differ.
- *Greedy* and *temperature 0.3* give the same plausible gist (malaria: artemether-lumefantrine only, one of the six ACTs in the corpus) and repeat themselves (greedy repeats 22.6 % of its 4-grams); they hit 5 and 4 of 16 rubric facts.
- *Beam search* reads better (no copied sentence, repetition 11.5 %) but is **confidently wrong on hypertension**: it says the choice "does not change based on age", while the NICE text sets an age-55 rule. It scores zero rubric flags only because the flag patterns do not recognise that error, a reminder that the rubric is a screen and the outputs must be read.
- *Top-k* and *top-p* are fluent but drift: top-p invented an ACT rule keyed to "> 3 years" of previous use and, for sepsis, a list of "SIRS criteria" from outside the corpus.
- *Temperature 0.7* starts to wander (a list of quinine derivatives for malaria); *temperature 1.2* collapses into gibberish ("Newton's Athletic Gold (medicated athletic tape)" for sepsis; random symbols for hypertension). It scores 0 of 16 facts and introduces 3 numbers found nowhere in the source.

**Diversity.** Temperature controls it cleanly: the share of unique bigrams across four samples rises from 0.52 at 0.3 to 0.77 at 0.7 and 0.99 at 1.2, while repetition falls from 0.19 to 0.06 to about 0. Top-k (0.82) and top-p (0.82) sit near temperature 0.7. But the extra diversity buys nothing here: at 1.2 it is noise, and at 0.7 it is wrong in more varied ways. Low temperature is repetitive (distinct-2 0.52, repetition 0.19, close to greedy's 0.23).

**Choice for C2 and C3.** Greedy is the best decoder for this domain: one deterministic answer, the same speed as sampling, the highest fact count (tied with top-p) and exact reproducibility, which also makes the speculative-decoding comparison in C2 clean (greedy speculative decoding must reproduce greedy output). Its repetition and invented sections are a property of the base model, not of the decoder, and are what Part B's fine-tuning addressed.
""")


def add_c2(md, code):
    md(r"""
## C2: Speculative decoding (2 marks)

**Idea.** A small *draft* model proposes several next tokens cheaply, and the large *target* model checks all of them in **one** forward pass, keeping the longest correct prefix. With greedy decoding the accepted text is exactly what the target model would have produced alone, so the technique should change
**speed, not output**. The gain depends on two things: how often the draft guesses what the target would say, and how cheap one draft token is compared with one target token.

**Draft model.** `HuggingFaceTB/SmolLM2-1.7B-Instruct` (the brief's recommendation for a 7B target): about 1.7 B parameters, 24 layers, 3.4 GB in bfloat16, so it fits on the GPU next to the 14.5 GB target. It uses a **different tokenizer** from Mistral (49,152 against 32,000 tokens) and the two tokenizers must not be swapped,
so ordinary assisted generation (which needs one shared vocabulary) is not possible. transformers 4.46.3 provides *universal assisted generation* (`AssistedCandidateGeneratorDifferentTokenizers`): the draft's tokens are decoded to text and re-encoded with the target tokenizer, and the target verifies them.

**Method.** Same three prompts, greedy decoding, `max_new_tokens = 150`, target in bfloat16 (the C1 model). The number of draft tokens per round is tuned over 3, 5 and 8 (constant) and the library's adaptive "heuristic" schedule. Plain and speculative greedy are then run **alternately, 5 times each**, and the speed-up is the **median of the paired ratios** (each speculative run
divided by the plain run made just before it), which cancels load changes on the shared GPU. Quality is checked by comparing the text with plain greedy. The number of **target forward passes** is counted with a hook: plain greedy needs one per token, so *new tokens per target pass* shows how many tokens each verification step yields.
A smaller draft (SmolLM2-360M-Instruct) is added as an ablation to show how the draft's cost changes the result.
""")

    code(r"""
DRAFT_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tok_d = AutoTokenizer.from_pretrained(DRAFT_ID)
mem0 = torch.cuda.memory_allocated(); t0 = time.perf_counter()
draft = AutoModelForCausalLM.from_pretrained(DRAFT_ID, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
W_DRAFT = (torch.cuda.memory_allocated() - mem0) / 1e9
pair = pd.DataFrame({
    "target (Mistral-7B-v0.1)": [sum(p.numel() for p in base_c.parameters()) / 1e9, base_c.config.num_hidden_layers, base_c.config.hidden_size, len(tok_c), type(tok_c).__name__, round(W_BF16, 2)],
    "draft (SmolLM2-1.7B-Instruct)": [sum(p.numel() for p in draft.parameters()) / 1e9, draft.config.num_hidden_layers, draft.config.hidden_size, len(tok_d), type(tok_d).__name__, round(W_DRAFT, 2)]},
    index=["parameters (billions)", "decoder layers", "hidden size", "tokenizer vocabulary", "tokenizer class", "resident GPU weights (GB)"])
print(f"draft loaded in {time.perf_counter() - t0:.0f} s; both models resident: {torch.cuda.memory_allocated() / 1e9:.2f} GB of GPU memory")
print("vocabularies identical:", tok_c.get_vocab() == tok_d.get_vocab())
pair
""")

    code(r"""
def spec_extra(d_model=None, d_tok=None, n_tokens=None, schedule="constant"):
    # universal assisted generation: the target tokenizer and the draft tokenizer are passed explicitly
    d_model, d_tok = d_model or draft, d_tok or tok_d
    if n_tokens is not None:
        d_model.generation_config.num_assistant_tokens = n_tokens
        d_model.generation_config.num_assistant_tokens_schedule = schedule
    return dict(assistant_model=d_model, tokenizer=tok_c, assistant_tokenizer=d_tok)

def counted(fn):
    # runs fn() and counts forward passes of the target model
    calls = [0]
    h = base_c.register_forward_hook(lambda m, i, o: calls.__setitem__(0, calls[0] + 1))
    try: r = fn()
    finally: h.remove()
    return r, calls[0]

GREEDY = dict(do_sample=False)
# first speculative call: warm-up and the check that the cross-tokenizer path works at all
probe, probe_calls = counted(lambda: run_one(base_c, tok_c, PROMPTS[0], GREEDY, **spec_extra(n_tokens=5)))
print(f"speculative call works: {probe['new_tokens']} new tokens with {probe_calls} target forward passes (plain greedy needs {probe['new_tokens']})")
print(probe["text"][:300])
""")

    md(r"""
### C2.1 Cost of one draft token against one target token

Speculative decoding only pays if drafting is cheap. Each model is run alone, greedily, on the same prompts with its own tokenizer.
""")
    code(r"""
def alone(model, t, p):
    enc = t(fmt_prompt(p["instruction"]), return_tensors="pt").to(model.device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(**enc, max_new_tokens=150, do_sample=False, pad_token_id=t.eos_token_id)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    n = ids.shape[1] - enc["input_ids"].shape[1]
    return n / dt
alone(draft, tok_d, PROMPTS[0])                                                      # warm-up
t_tps = sorted(alone(base_c, tok_c, p) for p in PROMPTS for _ in range(3))[4]
d_tps = sorted(alone(draft, tok_d, p) for p in PROMPTS for _ in range(3))[4]
cost_ratio = t_tps / d_tps                                                          # time of one draft token / time of one target token
print(f"target alone: {t_tps:.1f} tokens/s ({1000 / t_tps:.0f} ms per token) | draft alone: {d_tps:.1f} tokens/s ({1000 / d_tps:.0f} ms per token)")
print(f"one draft token costs {cost_ratio:.2f} of a target token although the draft has {sum(p.numel() for p in draft.parameters()) / sum(p.numel() for p in base_c.parameters()):.0%} of the parameters")
print("reading 14.5 GB of weights at an A100's ~2 TB/s would take about 7 ms per token, so one token costs several times that here: per-token overhead (Python, kernel launches) dominates, not memory bandwidth")
""")

    md(r"""
### C2.2 Tuning the number of draft tokens per round
""")
    code(r"""
variants = [("constant", 3), ("constant", 5), ("constant", 8), ("heuristic", 5)]
sweep = []
for sched, n in variants:
    spec_extra(n_tokens=n, schedule=sched)
    for p in PROMPTS:
        r, calls = counted(lambda: run_one(base_c, tok_c, p, GREEDY, **spec_extra()))
        sweep.append({"schedule": sched, "draft_tokens": n, "prompt_id": p["id"], "tok_per_s": r["new_tokens"] / r["seconds"], "new_tokens": r["new_tokens"], "target_passes": calls})
sw = pd.DataFrame(sweep).groupby(["schedule", "draft_tokens"], sort=False).agg(tok_per_s=("tok_per_s", "mean"), new_tokens=("new_tokens", "mean"), target_passes=("target_passes", "mean")).round(2)
sw["new_tokens_per_target_pass"] = (sw["new_tokens"] / sw["target_passes"]).round(2)
best_sched, best_n = sw["tok_per_s"].idxmax()
print(f"best variant by speed: {best_sched}, {best_n} draft tokens (single runs; the paired benchmark below is the measurement that counts)")
sw
""")

    md(r"""
### C2.3 Benchmark: plain greedy against speculative greedy (alternating runs, 5 repeats)
""")
    code(r"""
spec_extra(n_tokens=best_n, schedule=best_sched)
REP = 5
plain_runs, spec_runs = {p["id"]: [] for p in PROMPTS}, {p["id"]: [] for p in PROMPTS}
for rep in range(REP):                                        # alternating, so that load changes on the shared GPU hit both equally
    for p in PROMPTS:
        plain_runs[p["id"]].append(run_one(base_c, tok_c, p, GREEDY))
        spec_runs[p["id"]].append(run_one(base_c, tok_c, p, GREEDY, **spec_extra()))
tps = lambda r: r["new_tokens"] / r["seconds"]
med = lambda xs: sorted(xs)[len(xs) // 2]
plain_calls, spec_calls = {}, {}
for p in PROMPTS:
    _, plain_calls[p["id"]] = counted(lambda: run_one(base_c, tok_c, p, GREEDY))
    _, spec_calls[p["id"]] = counted(lambda: run_one(base_c, tok_c, p, GREEDY, **spec_extra()))
c2_rows = []
for p in PROMPTS:
    a, b = plain_runs[p["id"]], spec_runs[p["id"]]
    ratios = [tps(y) / tps(x) for x, y in zip(a, b)]
    c2_rows.append({"prompt_id": p["id"], "plain_tok_s": round(med([tps(r) for r in a]), 2), "speculative_tok_s": round(med([tps(r) for r in b]), 2),
                    "speed_up_median": round(med(ratios), 2), "speed_up_min": round(min(ratios), 2), "speed_up_max": round(max(ratios), 2),
                    "target_passes_plain": plain_calls[p["id"]], "target_passes_spec": spec_calls[p["id"]],
                    "tokens_per_target_pass": round(b[0]["new_tokens"] / spec_calls[p["id"]], 2), "identical_text": a[0]["text"] == b[0]["text"]})
c2 = pd.DataFrame(c2_rows)
c2.to_csv(REPORTS_DIR / "c2_speculative.csv", index=False)
all_ratios = [tps(y) / tps(x) for p in PROMPTS for x, y in zip(plain_runs[p["id"]], spec_runs[p["id"]])]
tpp = c2["tokens_per_target_pass"].mean()
print(f"speed-up (median of {len(all_ratios)} paired ratios): {med(all_ratios):.2f}x, range {min(all_ratios):.2f}x to {max(all_ratios):.2f}x | mean tokens per target pass: {tpp:.2f} | identical outputs: {int(c2['identical_text'].sum())} of {len(c2)}")
pred = tpp / (1 + best_n * cost_ratio)
print(f"simple model: tokens per round / (1 target pass + {best_n} draft tokens x {cost_ratio:.2f}) = {tpp:.2f} / {1 + best_n * cost_ratio:.2f} = {pred:.2f}x (ignores the text re-encoding between the two tokenizers)")
c2
""")

    md(r"""
### C2.4 Output quality: does speculative decoding change the text?
""")
    code(r"""
for p in PROMPTS:
    a, b = plain_runs[p["id"]][0]["text"], spec_runs[p["id"]][0]["text"]
    if a == b:
        print(f"[{p['id']}] speculative output is IDENTICAL to plain greedy ({len(a)} characters)")
    else:
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"[{p['id']}] outputs DIFFER from character {k} of {max(len(a), len(b))}")
        print("   plain      :", repr(a[max(0, k - 60):k + 120])); print("   speculative:", repr(b[max(0, k - 60):k + 120]))
    sa, sb = score_output(p, a), score_output(p, b)
    print(f"   rubric facts plain {len(sa['facts_hit'])}/{sa['facts_total']} vs speculative {len(sb['facts_hit'])}/{sb['facts_total']}; flags {len(sa['flags'])} vs {len(sb['flags'])}")
print("\nfull speculative output for prompt 1:\n" + spec_runs[PROMPTS[0]["id"]][0]["text"])
""")

    md(r"""
### C2.5 Ablation: a smaller draft model (SmolLM2-360M-Instruct)

A draft with fewer layers and a smaller hidden size is cheaper per token but agrees with the target less often. This checks which effect wins.
""")
    code(r"""
D2_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
tok_d2 = AutoTokenizer.from_pretrained(D2_ID)
draft2 = AutoModelForCausalLM.from_pretrained(D2_ID, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
alone(draft2, tok_d2, PROMPTS[0])
d2_tps = sorted(alone(draft2, tok_d2, p) for p in PROMPTS for _ in range(3))[4]
spec_extra(draft2, tok_d2, 5, "constant")
abl = {"1.7B": [], "360M": []}; ab_calls = {}
for rep in range(3):
    for p in PROMPTS:
        a = run_one(base_c, tok_c, p, GREEDY)
        abl["1.7B"].append((tps(run_one(base_c, tok_c, p, GREEDY, **spec_extra(draft, tok_d, best_n, best_sched))) / tps(a)))
        abl["360M"].append((tps(run_one(base_c, tok_c, p, GREEDY, **spec_extra(draft2, tok_d2, 5, "constant"))) / tps(a)))
r360, calls360 = counted(lambda: run_one(base_c, tok_c, PROMPTS[0], GREEDY, **spec_extra(draft2, tok_d2, 5, "constant")))
same360 = r360["text"] == plain_runs[PROMPTS[0]["id"]][0]["text"]
abl_tbl = pd.DataFrame({"draft": ["SmolLM2-1.7B-Instruct", "SmolLM2-360M-Instruct"], "layers": [draft.config.num_hidden_layers, draft2.config.num_hidden_layers],
                        "draft tok/s alone": [round(d_tps, 1), round(d2_tps, 1)], "speed-up (median of 9 paired runs)": [round(med(abl["1.7B"]), 2), round(med(abl["360M"]), 2)],
                        "tokens per target pass (prompt 1)": [round(c2.iloc[0]["tokens_per_target_pass"], 2), round(r360["new_tokens"] / calls360, 2)]})
print("360M output identical to plain greedy on prompt 1:", same360)
del draft2; gc.collect(); torch.cuda.empty_cache()
abl_tbl
""")


    md(r"""
### Inference: C2

**It works and it does not change the text.** The cross-tokenizer path of transformers 4.46.3 ran without modification. With greedy decoding the speculative output was **identical to plain greedy for all three prompts** (and for the 360M draft on prompt 1), so wording, rubric facts and unsupported claims are unchanged: with the bfloat16 target there is no quality trade-off.
The method itself worked as designed: each target forward pass produced about 2.2 to 3.2 tokens, so the target ran roughly 50 to 70 passes instead of 150, depending on the prompt and on the number of draft tokens per round (more draft tokens per round give more tokens per pass in the sweep).

**But it is not faster here.** The median of the paired speed-up ratios for the 1.7B draft is about **1.0x** (see the table above; individual pairs between roughly 0.8x and 1.1x), so there is no throughput gain, and the smaller 360M draft is slower still (about 0.8x). A result of "no gain" is a measurement, not a failure of the setup: the output is correct and the cost of drafting simply cancels the saving.

**Why.** Speculative decoding wins when one target step is expensive and one draft token is cheap. Here a plain target step takes about 32 ms (about 31 tokens/s), although reading its weights would take only about 7 ms on an A100, so the time per token is dominated by overhead (Python and one kernel launch per operation on a 4-CPU pod). That overhead grows with the **number of layers**, not with the
number of parameters: the 1.7B draft (24 layers, 24 % of the parameters) costs about 18 ms per token, roughly **0.6 of a target token**. Drafting several tokens and then verifying them therefore costs about as much as the plain steps it replaces. The simple model printed above, tokens per target pass / (1 + draft tokens x cost ratio), gives about 0.7x for five draft tokens per round and about 1x for two to three,
which is consistent with the measured 1.0x if the adaptive schedule drafts few tokens per round (this was not measured directly). The ablation supports the explanation: **SmolLM2-360M has 32 layers, the same as Mistral**, so it is not cheaper per token (about 25 ms against 18 ms) and is slower overall, although it has a fifth of the parameters.

**When it would help.** When a target step is much more expensive than a draft step: a larger target, a slower target such as the 4-bit model (measured in C3, where it does help), or an inference stack that removes the per-token overhead (CUDA graphs, a static cache) so that parameter count, not layer count, sets the cost. The practical rule from this experiment is to compare the **per-token time** of the two models, not their parameter counts, before choosing a draft.
""")


def add_c3(md, code):
    md(r"""
## C3: 4-bit quantization and production cost (3 marks)

**Quantization.** The base model is loaded with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`: the weights are stored in 4 bits (NF4) and de-quantized to bfloat16 for each matrix product, which cuts memory by about 3.3x at some cost in speed and fidelity.

**Configurations** (all with the best decoder from C1, **greedy**, `max_new_tokens = 150`): bfloat16 (the Part A baseline), 4-bit NF4, and 4-bit NF4 with speculative decoding from C2 (SmolLM2-1.7B draft).

**Prompts.** Ten domain prompts: the three fixed prompts of Parts A to C2 plus seven new ones written from the corpus (WHO, CDC x3, ICMR x2, PubChem), none overlapping the 50 training instructions (highest word-set similarity 0.35 against a 0.70 limit). Each new prompt has a rubric whose facts are
quoted from its corpus file.

**Measurement.**
- *Throughput* = total new tokens / total generation time over all 10 prompts, after a warm-up. The three configurations are run **alternately, three times each**, so load changes on the shared GPU affect all of them equally; the reported value is the median, with the range.
- *Peak VRAM* = the GPU memory of the weights that configuration needs (measured when each model is loaded) + the largest extra allocation during its generation (KV cache and activations, measured with `torch.cuda.max_memory_allocated`). For the speculative configuration the draft model is included.
- *Cost per 1M tokens* = (1,000,000 / throughput) / 3600 x hourly rate, with **Rs 12.00/hour**, the brief's rate for an A100, which is the GPU this ran on.
""")

    code(r"""
NEW_PROMPTS = [
    {"id": "n01_malaria_second_line", "corpus_target": "domain_corpus/who_malaria_2024-11-30.txt",
     "instruction": "A patient with uncomplicated P. falciparum malaria has treatment failure within 28 days. What second-line treatment does WHO recommend, and which longer regimens are no longer generally recommended?"},
    {"id": "n02_gonorrhoea_regimen", "corpus_target": "domain_corpus/cdc_sti_2021.txt",
     "instruction": "What single-dose regimen does CDC recommend for uncomplicated gonococcal infection of the cervix, urethra or rectum in adults, and how does the dose change with body weight?"},
    {"id": "n03_syphilis_primary", "corpus_target": "domain_corpus/cdc_sti_2021.txt",
     "instruction": "What is the recommended treatment for primary or secondary syphilis in adults?"},
    {"id": "n04_bacterial_vaginosis", "corpus_target": "domain_corpus/cdc_sti_2021.txt",
     "instruction": "Which regimens does CDC recommend for bacterial vaginosis?"},
    {"id": "n05_cystitis_icmr", "corpus_target": "domain_corpus/icmr_antimicrobial_2019.txt",
     "instruction": "What does ICMR list as the drug of choice for acute cystitis, and when should nitrofurantoin or fosfomycin be avoided?"},
    {"id": "n06_diarrhoea_zinc", "corpus_target": "domain_corpus/icmr_stw_vol1_2019.txt",
     "instruction": "How much zinc does the ICMR workflow advise for a child with acute diarrhoea, and for how long?"},
    {"id": "n07_metronidazole", "corpus_target": "domain_corpus/pubchem_metronidazole.txt",
     "instruction": "Which infections is metronidazole indicated for, and what advice does PubChem give about alcohol?"},
]
RUBRIC.update({
    "n01_malaria_second_line": {"quotes": ["The recommended second-line treatment is an alternative ACT known to be effective in the region", "these regimens are no longer generally recommended"],
        "facts": [("alt_act", "an alternative ACT as second line", r"alternative ACT|another ACT|different ACT|second[- ]line[^.]{0,60}\bACTs?\b"),
                  ("no_long", "7-day artesunate or quinine regimens no longer generally recommended", r"no longer (generally )?recommended|not (generally )?recommended")],
        "flags": [("quinine_second_line", "presents quinine as the second line (the corpus says those regimens are no longer generally recommended)", r"second[- ]line[^.]{0,80}quinine|quinine[^.]{0,80}second[- ]line")]},
    "n02_gonorrhoea_regimen": {"quotes": ["Ceftriaxone 500 mg* IM in a single dose for persons weighing <150 kg", "1 g ceftriaxone should be administered"],
        "facts": [("ceftriaxone", "ceftriaxone", r"ceftriaxone"), ("dose500", "500 mg", r"\b500\s*mg"), ("im_single", "intramuscular single dose", r"(IM|intramuscular)[^.]{0,60}(single|one)|(single|one)[- ]dose"),
                  ("weight", "1 g from 150 kg", r"\b1\s*g\b|1,?000\s*mg|150\s*kg"), ("doxy", "doxycycline if chlamydia not excluded", r"doxycycline")], "flags": []},
    "n03_syphilis_primary": {"quotes": ["Benzathine penicillin G 2.4 million units IM in a single dose"],
        "facts": [("bpg", "benzathine penicillin G", r"benzathine penicillin"), ("dose", "2.4 million units", r"2\.4\s*million"), ("im", "intramuscular", r"\bIM\b|intramuscular"), ("single", "single dose", r"single|one dose|once")], "flags": []},
    "n04_bacterial_vaginosis": {"quotes": ["Metronidazole 500 mg orally 2 times/day for 7 days", "Clindamycin cream 2% one full applicator (5 g) intravaginally at bedtime for 7 days"],
        "facts": [("metro_oral", "metronidazole 500 mg orally twice daily for 7 days", r"metronidazole[^.]{0,80}500\s*mg|500\s*mg[^.]{0,60}metronidazole"), ("gel", "metronidazole gel", r"gel"), ("clinda", "clindamycin cream", r"clindamycin"), ("days7", "7 days", r"\b7\s*days|seven days")], "flags": []},
    "n05_cystitis_icmr": {"quotes": ["Fosfomycin and nitrofurantoin should be avoided when there is suspicion of pyelonephritis or prostatitis"],
        "facts": [("nitro", "nitrofurantoin as drug of choice", r"nitrofurantoin"), ("fosfo", "fosfomycin", r"fosfomycin"), ("cotrim", "co-trimoxazole", r"co-?trimoxazole"), ("avoid", "avoid with pyelonephritis or prostatitis", r"pyelonephritis|prostatitis")], "flags": []},
    "n06_diarrhoea_zinc": {"quotes": ["2-6 months", "10mg/day x 2 weeks", "20mg/day x 2 weeks"],
        "facts": [("z10", "10 mg/day for 2 to 6 months", r"\b10\s*mg"), ("z20", "20 mg/day above 6 months", r"\b20\s*mg"), ("weeks2", "for 2 weeks", r"2 weeks|two weeks|14 days"), ("age", "age split at 6 months", r"6 months")], "flags": []},
    "n07_metronidazole": {"quotes": ["Metronidazole is indicated for the treatment of confirmed trichomoniasis", "Avoid alcohol"],
        "facts": [("trich", "trichomoniasis", r"trichomon"), ("bv", "bacterial vaginosis", r"bacterial vaginosis"), ("amoeb", "amebiasis", r"amoebiasis|amebiasis|amoebic|amebic"), ("anaerobic", "anaerobic infections", r"anaerobic"), ("alcohol", "avoid alcohol", r"alcohol")], "flags": []},
})
PROMPTS10 = PROMPTS + NEW_PROMPTS
problems = []
for p in NEW_PROMPTS:
    txt = squash((ROOT / p["corpus_target"]).read_text(encoding="utf-8"))
    problems += [f"{p['id']}: quote not found: {q}" for q in RUBRIC[p["id"]]["quotes"] if squash(q) not in txt]
print("rubric quotes verified in the corpus files:", "none missing" if not problems else problems)
assert not problems
# the new prompts must not near-duplicate the 50 training instructions or the 3 fixed prompts
tok_set = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
refs = [r["instruction"] for r in train + evals] + [p["instruction"] for p in PROMPTS]
sims = {p["id"]: max(len(tok_set(p["instruction"]) & tok_set(x)) / len(tok_set(p["instruction"]) | tok_set(x)) for x in refs) for p in NEW_PROMPTS}
print("highest word-set similarity to any training or fixed prompt (limit 0.70):", {k[:3]: round(v, 2) for k, v in sims.items()})
assert max(sims.values()) < 0.70
pd.DataFrame([{"id": p["id"], "source": Path(p["corpus_target"]).name, "prompt": p["instruction"]} for p in PROMPTS10])
""")

    md(r"""
### C3.1 Load the 4-bit model and run the three configurations
""")
    code(r"""
mem0 = torch.cuda.memory_allocated(); t0 = time.perf_counter()
q4 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=BNB, device_map={"": 0}).eval()
W_4BIT = (torch.cuda.memory_allocated() - mem0) / 1e9
print(f"4-bit NF4 model loaded in {time.perf_counter() - t0:.0f} s: {W_4BIT:.2f} GB resident (bfloat16 model: {W_BF16:.2f} GB, {W_BF16 / W_4BIT:.1f}x larger); all three models resident: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

CONFIGS = {"bfloat16 (Part A baseline)": (base_c, {}, W_BF16),
           "4-bit NF4": (q4, {}, W_4BIT),
           "4-bit NF4 + speculative decoding": (q4, spec_extra(draft, tok_d, best_n, best_sched), W_4BIT + W_DRAFT)}

def run_pass(model, extra):
    # the 10 prompts one after another; returns the results and the extra GPU memory allocated while generating
    before = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    res = [run_one(model, tok_c, p, GREEDY, **extra) for p in PROMPTS10]
    return res, (torch.cuda.max_memory_allocated() - before) / 1e9

for name, (m, ex, _) in CONFIGS.items():                          # warm-up
    run_one(m, tok_c, PROMPTS10[0], dict(do_sample=False), **ex)
REP3 = 3
passes = {n: [] for n in CONFIGS}; extra_gb = {n: [] for n in CONFIGS}
for rep in range(REP3):                                            # alternating configurations, so that GPU load changes hit all of them equally
    for name, (m, ex, _) in CONFIGS.items():
        res, eg = run_pass(m, ex)
        passes[name].append(res); extra_gb[name].append(eg)
    print(f"repeat {rep + 1}/{REP3} done")
""")

    md(r"""
### C3.2 Results table: peak VRAM, throughput and cost per 1M tokens
""")
    code(r"""
RATE = 12.00                                                       # Rs per hour: the brief's rate for an A100
thr = lambda res: sum(r["new_tokens"] for r in res) / sum(r["seconds"] for r in res)
cost = lambda tp: (1_000_000 / tp) / 3600 * RATE
rows = []
for name, (m, ex, w) in CONFIGS.items():
    tps_list = [thr(res) for res in passes[name]]
    tp = sorted(tps_list)[len(tps_list) // 2]
    rows.append({"Configuration": name, "Peak VRAM (GB)": round(w + max(extra_gb[name]), 2), "Throughput (tok/s)": round(tp, 2), "range of 3 repeats": f"{min(tps_list):.1f} to {max(tps_list):.1f}",
                 "Cost / 1M tokens (Rs)": round(cost(tp), 2), "weights (GB)": round(w, 2), "extra during generation (GB)": round(max(extra_gb[name]), 2)})
c3 = pd.DataFrame(rows).set_index("Configuration")
c3["speed vs bfloat16"] = (c3["Throughput (tok/s)"] / c3.iloc[0]["Throughput (tok/s)"]).round(2)
c3["cost vs bfloat16"] = (c3["Cost / 1M tokens (Rs)"] / c3.iloc[0]["Cost / 1M tokens (Rs)"]).round(2)
c3.to_csv(REPORTS_DIR / "c3_quantization_cost.csv")
print(f"cost = (1,000,000 / throughput) / 3600 x Rs {RATE:.2f} per hour; total tokens per pass: {sum(r['new_tokens'] for r in passes['bfloat16 (Part A baseline)'][0])}")
display(c3)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4)); labels = ["bf16", "4-bit", "4-bit + spec"]
ax[0].bar(labels, c3["Peak VRAM (GB)"], color="#4c78a8"); ax[0].set_title("Peak VRAM (GB)")
ax[1].bar(labels, c3["Throughput (tok/s)"], color="#54a24b"); ax[1].set_title("Throughput (tokens/s)")
ax[2].bar(labels, c3["Cost / 1M tokens (Rs)"], color="#e45756"); ax[2].set_title("Cost per 1M tokens (Rs)")
plt.tight_layout(); plt.show()
""")

    md(r"""
### C3.3 Output quality on the 10 prompts

Does 4-bit quantization or speculative decoding change the answers? Rubric facts are counted over all 10 prompts (see the rubric above), together with unsupported claims, numbers found nowhere in the source file, and invented `### ...` sections.
""")
    code(r"""
texts = {n: [r["text"] for r in passes[n][0]] for n in CONFIGS}
rows = []
for name in CONFIGS:
    sc = [score_output(p, t) for p, t in zip(PROMPTS10, texts[name])]
    rows.append({"Configuration": name, "rubric facts mentioned": sum(len(s["facts_hit"]) for s in sc), "of": sum(s["facts_total"] for s in sc), "unsupported-claim flags": sum(len(s["flags"]) for s in sc),
                 "ungrounded numbers": sum(len(s["ungrounded_numbers"]) for s in sc), "prompts with invented sections": sum(s["extra_sections"] > 0 for s in sc),
                 "identical to bfloat16 output": sum(a == b for a, b in zip(texts[name], texts["bfloat16 (Part A baseline)"])),
                 "identical to 4-bit output": sum(a == b for a, b in zip(texts[name], texts["4-bit NF4"]))})
qual = pd.DataFrame(rows).set_index("Configuration")
qual.to_csv(REPORTS_DIR / "c3_quality.csv")
display(qual)
per = pd.DataFrame({n: [len(score_output(p, t)["facts_hit"]) for p, t in zip(PROMPTS10, texts[n])] for n in CONFIGS}, index=[p["id"] for p in PROMPTS10])
print("rubric facts mentioned per prompt:"); display(per)

# where does speculative decoding on the 4-bit model first differ from plain 4-bit?
a_name, b_name = "4-bit NF4", "4-bit NF4 + speculative decoding"
shown = 0
for p, x, y in zip(PROMPTS10, texts[a_name], texts[b_name]):
    if x != y and shown < 2:
        k = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        print(f"[{p['id']}] first difference at character {k} of {max(len(x), len(y))}")
        print("   plain 4-bit      :", repr(x[max(0, k - 50):k + 80])); print("   4-bit + speculative:", repr(y[max(0, k - 50):k + 80])); shown += 1
print("outputs identical, plain 4-bit vs 4-bit + speculative:", sum(x == y for x, y in zip(texts[a_name], texts[b_name])), "of", len(PROMPTS10))
""")

    md(r"""
### C3.4 Optional extension (no marks): adapter + 4-bit + speculative decoding

The realistic production setup combines all three techniques: the fine-tuned Adapter B (B2) on the 4-bit model with the draft model. The adapter is attached to a second 4-bit copy so that the clean 4-bit model above is untouched. The same 10 prompts and greedy decoding are used; the adapter's answers are not
expected to equal the base model's, so quality is read from the rubric and the format (disclaimer, stops by itself).
""")
    code(r"""
ext = {}
try:
    q4b = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=BNB, device_map={"": 0}).eval()
    q4a = PeftModel.from_pretrained(q4b, ADAPTER_DIR).eval()
    W_AD = W_4BIT                                                   # the LoRA matrices add about 27 MB
    for name, ex, w in [("4-bit + adapter", {}, W_AD), ("4-bit + adapter + speculative", spec_extra(draft, tok_d, best_n, best_sched), W_AD + W_DRAFT)]:
        run_one(q4a, tok_c, PROMPTS10[0], dict(do_sample=False), **ex)
        res_list, eg_list = [], []
        for _ in range(2):
            r, eg = run_pass(q4a, ex); res_list.append(r); eg_list.append(eg)
        tp = sum(thr(r) for r in res_list) / len(res_list)
        sc = [score_output(p, r["text"]) for p, r in zip(PROMPTS10, res_list[0])]
        ext[name] = {"Peak VRAM (GB)": round(w + max(eg_list), 2), "Throughput (tok/s)": round(tp, 2), "Cost / 1M tokens (Rs)": round(cost(tp), 2),
                     "rubric facts": sum(len(s["facts_hit"]) for s in sc), "flags": sum(len(s["flags"]) for s in sc),
                     "ends with disclaimer": sum(r["text"].strip().endswith(DISCLAIMER) for r in res_list[0]), "mean new tokens": round(sum(r["new_tokens"] for r in res_list[0]) / 10, 1)}
    ext_tbl = pd.DataFrame.from_dict(ext, orient="index")
    ext_tbl.to_csv(REPORTS_DIR / "c3_extension_adapter.csv")
    display(ext_tbl)
    print("sample adapter + 4-bit + speculative answer (prompt 1):\n" + res_list[0][0]["text"])
except Exception as e:                                              # optional extension: report what happened instead of stopping the notebook
    print("EXTENSION FAILED:", type(e).__name__, str(e)[:400])
""")


    md(r"""
### C3.5 Deployment recommendation

For this domain assistant on an A100 I would deploy the **bfloat16 model with greedy decoding**: it is the cheapest configuration (about Rs 107 per 1M tokens against about Rs 182 for 4-bit with speculative decoding and Rs 236 for plain 4-bit) and it mentions the most corpus facts (24 of 44, against 20 and 19). The 4-bit model cuts peak VRAM 3.3x (4.4 GB against 14.5 GB) at about 2.2x the cost per token and
changes every answer, so it should be used only when memory is the constraint (for example a 16 GB T4), where speculative decoding then speeds it up by about 1.3x.
""")

    md(r"""
### Inference: C3

**Memory.** 4-bit NF4 shrinks the weights from 14.5 GB to 4.1 GB (3.5x) and the peak VRAM from 14.5 GB to 4.4 GB (3.3x), so the 7B model fits on a 16 GB GPU such as the T4 in the brief with room for the KV cache, where the bfloat16 model barely would. Adding the 1.7B draft model for speculative decoding brings the 4-bit configuration to about 7.9 GB. Generation itself adds little (0.03 to 0.3 GB) at batch size 1 and 150 tokens.

**Throughput and cost.** The 4-bit model is **slower, not faster**: about 0.45x the bfloat16 throughput, so the cost per 1M tokens is about **2.2x higher** at the same hourly rate. Each step has to de-quantize the 4-bit weights before the matrix product, and in this overhead-bound regime (see C2) the extra work costs more than the smaller memory traffic saves.
Quantization therefore saves **memory, not money**, when one model already fits comfortably on the GPU. Speculative decoding behaves differently here than in C2: a 4-bit target step is about 70 ms against about 18 ms for a draft token, so drafting is cheap relative to verification and speculative decoding recovers about **1.3x** over plain 4-bit (cost about 1.7x the bfloat16 cost instead of 2.2x).
The cost formula uses single-stream throughput; a production server that batches many requests would divide these costs, and the ranking could change, but that was not measured.

**Quality.** 4-bit changes every greedy answer (none of the 10 outputs is identical to bfloat16) and the rubric facts drop from 24 to 20 of 44 (the sepsis answer lost both of its facts, the zinc answer two of three); unsupported-claim flags are not higher. With ten prompts and a pattern-based rubric this is a screen, not a benchmark, but the direction is clear: no gain in correctness, a small loss in coverage.
**Speculative decoding on the 4-bit model is not exactly output-preserving**: four of the ten answers differ from plain 4-bit. The target verifies several tokens in one matrix product, whose rounding differs slightly from one token at a time, and greedy decoding amplifies that when two candidate tokens are nearly tied; the diagnostic cell above shows where the first answers diverge. This is unlike the bfloat16 target in C2, where the outputs were identical.

**Optional extension: adapter + 4-bit + speculative decoding.** The fine-tuned adapter on the 4-bit model with the draft runs at about 13 tokens/s (about Rs 250 per 1M tokens): the adapter shortens answers (about 113 tokens) and speculative decoding helps only slightly (about 1.1x) on such short outputs. Its answers have the required form (most end with the disclaimer and stop by themselves; the rest hit the token cap) and similar rubric coverage (about 25 of 44 facts),
but more unsupported claims than the base model (5 flags against 2 to 3), the same pattern as B3: the adapter improves the form, not the reliability. The extension is therefore not a better production configuration than bfloat16 greedy decoding for this domain.
""")


def add(md, code):
    add_intro(md, code)
    add_c1(md, code)
    add_c2(md, code)
    add_c3(md, code)

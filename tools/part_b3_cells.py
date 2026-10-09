"""Part B3 cells for assignment_1b.ipynb. add(md, code, analysis=True) appends them. Cell sources are raw strings."""


def add(md, code, analysis=True):
    md(r"""
## B3: Evaluation and comparative analysis (1 mark)

**Requirement.** Run the trained adapter on the same 3 domain prompts used in Part A Step 2, present a side-by-side comparison (baseline vs adapter) and describe what improved or degraded (correctness, domain terminology, completeness, hallucinations).

**Method.**
- **Same settings as the baseline:** greedy decoding, `max_new_tokens = 150`, no system prompt, the same `### Instruction` / `### Response` block, the same three prompts.
- **Primary comparison: bfloat16 base + adapter.** The baseline ran in bfloat16, so with the same dtype the only thing that changes is the fine-tuning. The adapter was *trained* on a 4-bit base, so a **4-bit base + adapter** run is reported next to it as the matching-condition check.
- **The baseline is regenerated in the same session** (before the adapter is attached) and compared with the saved Part A outputs, so any difference between the columns comes from the adapter and nothing else. Speeds are measured in the same process after a warm-up generation.
- **Scoring against the corpus.** For each prompt a short rubric lists the facts a correct answer contains and the claims the corpus does not support. The rubric is tied to the corpus by **exact quotes from the prompt's own corpus file**, which the first cell verifies. Numbers in an output that appear
  neither in the prompt nor in that corpus file are listed as ungrounded. A pattern match is a screening tool and favours fluent answers, so the written analysis reads the outputs themselves.
""")

    code(r"""
from peft import PeftModel

# Ground truth per prompt. quotes: must appear in the corpus file (verified below). facts: regexes for what a correct answer contains.
# flags: regexes for claims the corpus does not support.
RUBRIC = {
    "malaria_uncomplicated_pf": {
        "quotes": ["should be treated with one of the following ACTs", "no longer generally recommended"],
        "facts": [("act", "names artemisinin-based combination therapy (ACT)", r"artemisinin[- ]based combination|\bACTs?\b"),
                  ("al", "artemether-lumefantrine", r"artemether[- ]lumefantrine|\bAL\b"),
                  ("asaq", "artesunate-amodiaquine", r"artesunate[- ]amodiaquine|AS\s*\+\s*AQ"),
                  ("asmq", "artesunate-mefloquine", r"artesunate[- ]mefloquine|ASMQ"),
                  ("dhap", "dihydroartemisinin-piperaquine", r"dihydroartemisinin[- ]piperaquine|\bDHA?P\b"),
                  ("assp", "artesunate + sulfadoxine-pyrimethamine", r"artesunate\s*(\+|plus|and|with)?\s*sulfadoxine|AS\s*\+\s*SP"),
                  ("aspy", "artesunate-pyronaridine", r"artesunate[- ]pyronaridine|ASPY")],
        "flags": [("single_drug_only", "presents one ACT as the recommendation (the corpus lists six options)", None),
                  ("quinine_second_line", "presents a quinine regimen as second line (corpus: quinine plus doxycycline or clindamycin is no longer generally recommended)",
                   r"second[- ]line[^.]{0,80}quinine|quinine[^.]{0,80}second[- ]line")]},
    "sepsis_iv_antibiotics": {
        "quotes": ["within 1 hour of calculating", "for whom the source of infection is unknown"],
        "facts": [("one_hour", "IV antibiotics within 1 hour", r"within\s+(1|one)\s+hour"),
                  ("risk_based", "tied to high risk / NEWS2 / risk category", r"high[- ]risk|NEWS2|risk category|risk level"),
                  ("broad_spectrum", "broad-spectrum antibiotics when the source is unknown", r"broad[- ]spectrum"),
                  ("source_specific", "switch to source-specific antibiotics once the source is confirmed", r"source[- ]specific|once the source|source is (identified|confirmed|known)|targeted")],
        "flags": [("external_guideline", "attributes the advice to a guideline that is not the corpus source (NICE)", r"Surviving Sepsis|\bSSC\b|SEP-1|Sepsis-3")]},
    "hypertension_first_line": {
        "quotes": ["Offer an ACE inhibitor or an ARB to adults starting step 1", "Offer a calcium-channel blocker (CCB) to adults starting step 1"],
        "facts": [("acei_arb", "ACE inhibitor or ARB at step 1", r"ACE inhibitor|angiotensin[- ]converting|\bARBs?\b|angiotensin II receptor|\bACEIs?\b"),
                  ("ccb", "calcium-channel blocker at step 1", r"calcium[- ]channel|\bCCBs?\b"),
                  ("age_55", "age 55 as the dividing line", r"\b55\b"),
                  ("family_origin", "Black African or African-Caribbean family origin", r"Black African|African[–-]Caribbean"),
                  ("t2dm", "type 2 diabetes", r"type 2 diabetes|T2DM")],
        "flags": [("thiazide_first_line", "lists a thiazide diuretic as first line (NICE: only if a CCB is not tolerated or in heart failure)", r"thiazide"),
                  ("wrong_age_rule", "gives a thiazide for people aged 55 or over (NICE: a calcium-channel blocker)", r"55[^.]{0,80}thiazide")]},
}
OPTION_IDS = ["al", "asaq", "asmq", "dhap", "assp", "aspy"]
squash = lambda s: " ".join(s.replace("’", "'").split())

# design check: every rubric quote exists in the corpus file, the adapter exists, the prompt ids match the Part A baseline
problems = []
for p in PROMPTS:
    corpus_text = squash((ROOT / p["corpus_target"]).read_text(encoding="utf-8"))
    problems += [f"{p['id']}: quote not found: {q}" for q in RUBRIC[p["id"]]["quotes"] if squash(q) not in corpus_text]
problems += [f"missing {ADAPTER_DIR / n}" for n in ("adapter_model.safetensors", "adapter_config.json") if not (ADAPTER_DIR / n).is_file()]
saved_a = {r["prompt_id"]: r for r in csv.DictReader((REPORTS_LAB / "baseline_outputs.csv").open(encoding="utf-8", newline=""))}
problems += [] if list(saved_a) == [p["id"] for p in PROMPTS] else ["baseline_outputs.csv prompt ids differ"]
print("design check problems:", problems or "none", "| rubric quotes verified in the corpus:", sum(len(r["quotes"]) for r in RUBRIC.values()))
assert not problems

def score_output(p, text):
    rub = RUBRIC[p["id"]]
    hits = [f for f, _, pat in rub["facts"] if re.search(pat, text, re.I)]
    flags = [f for f, _, pat in rub["flags"] if pat and re.search(pat, text, re.I)]
    if p["id"] == "malaria_uncomplicated_pf" and sum(o in hits for o in OPTION_IDS) == 1: flags.append("single_drug_only")
    source = (ROOT / p["corpus_target"]).read_text(encoding="utf-8")
    ungrounded = sorted({n for n in numbers_in(text)} - numbers_in(source) - numbers_in(p["instruction"]), key=float)
    return {"facts_hit": hits, "facts_total": len(rub["facts"]), "flags": flags, "ungrounded_numbers": ungrounded,
            "has_disclaimer": text.strip().endswith(DISCLAIMER), "extra_sections": len(re.findall(r"^#{2,3}\s*\w+", text, re.M)),
            "has_references_section": bool(re.search(r"#+\s*References", text))}

def generate_all(m, t):
    # greedy, 150 new tokens, no system prompt: the Part A baseline settings. One short warm-up so CUDA start-up is not billed to prompt 1.
    with torch.inference_mode():
        m.generate(**t(fmt_prompt("What is hypertension?"), return_tensors="pt").to(m.device), max_new_tokens=8, do_sample=False, pad_token_id=t.pad_token_id)
    rows = []
    for p in PROMPTS:
        enc = t(fmt_prompt(p["instruction"]), return_tensors="pt").to(m.device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            ids = m.generate(**enc, max_new_tokens=150, do_sample=False, pad_token_id=t.pad_token_id)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        new = ids[0, enc["input_ids"].shape[1]:]
        rows.append({"prompt_id": p["id"], "output": t.decode(new, skip_special_tokens=True).strip(), "new_tokens": int(new.shape[0]),
                     "tokens_per_second": round(int(new.shape[0]) / dt, 3), "stopped_on_eos": bool(new[-1].item() == t.eos_token_id)})
    return rows
""")

    md(r"""
### B3.1 Run: baseline again (bf16), adapter on bf16, adapter on 4-bit
""")
    code(r"""
tok_e = AutoTokenizer.from_pretrained(MODEL_ID)
tok_e.pad_token = tok_e.eos_token                      # same as the baseline; batch size 1, so padding never happens
results, mem = {}, {}

torch.cuda.reset_peak_memory_stats()
bf16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
results["baseline"] = generate_all(bf16, tok_e)                                   # adapter not attached yet
adapted = PeftModel.from_pretrained(bf16, ADAPTER_DIR).eval()                     # unmerged adapter on the bfloat16 base
results["adapter_bf16"] = generate_all(adapted, tok_e)
mem["bf16 base (+ adapter)"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
del adapted, bf16; gc.collect(); torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
q4 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=BNB, device_map={"": 0}).eval()
adapted4 = PeftModel.from_pretrained(q4, ADAPTER_DIR).eval()                      # the training condition: 4-bit base + adapter
results["adapter_4bit"] = generate_all(adapted4, tok_e)
mem["4-bit base + adapter"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
del adapted4, q4; gc.collect(); torch.cuda.empty_cache()

# the regenerated baseline must equal the saved Part A outputs (this lab run) and the earlier run
earlier_a = {r["prompt_id"]: r for r in csv.DictReader((ROOT / "reports" / "baseline_outputs.csv").open(encoding="utf-8", newline=""))}
same_lab = {r["prompt_id"]: r["output"].split() == saved_a[r["prompt_id"]]["output"].split() for r in results["baseline"]}
same_old = {r["prompt_id"]: r["output"].split() == earlier_a[r["prompt_id"]]["output"].split() for r in results["baseline"]}
print("baseline regenerated == Part A output on this lab :", same_lab)
print("baseline regenerated == earlier run (A6000)       :", same_old)
print("peak GPU memory (GB):", mem)
""")

    md(r"""
### B3.2 Side-by-side outputs

For each prompt: the Part A baseline, the adapter on bf16 (the primary comparison), and the adapter on 4-bit.
""")
    code(r"""
for i, p in enumerate(PROMPTS):
    print("=" * 110); print(f"PROMPT [{p['id']}]: {p['instruction']}"); print("=" * 110)
    for label, key in (("BASELINE (bf16)", "baseline"), ("ADAPTER B on bf16", "adapter_bf16"), ("ADAPTER B on 4-bit", "adapter_4bit")):
        r = results[key][i]
        print(f"\n--- {label}  [{r['new_tokens']} tokens, stopped on EOS: {r['stopped_on_eos']}]\n{r['output']}")
    print()
""")

    md(r"""
### B3.3 Scores against the corpus
""")
    code(r"""
score_rows = []
for system in ("baseline", "adapter_bf16", "adapter_4bit"):
    for p, r in zip(PROMPTS, results[system]):
        score_rows.append({"system": system, "prompt_id": p["id"], **r, **score_output(p, r["output"])})
sc = pd.DataFrame(score_rows)
sc["facts"] = sc.apply(lambda r: f"{len(r['facts_hit'])}/{r['facts_total']}", axis=1)
sc["flags_s"] = sc["flags"].apply(lambda x: "|".join(x)); sc["hits_s"] = sc["facts_hit"].apply(lambda x: "|".join(x)); sc["ungrounded"] = sc["ungrounded_numbers"].apply(lambda x: "|".join(x))
display(sc[["system", "prompt_id", "new_tokens", "stopped_on_eos", "has_disclaimer", "extra_sections", "has_references_section", "facts", "hits_s", "flags_s", "ungrounded"]])
summary = sc.groupby("system", sort=False).agg(ends_with_disclaimer=("has_disclaimer", "sum"), stops_by_itself=("stopped_on_eos", "sum"),
          invented_sections=("extra_sections", lambda s: int((s > 0).sum())), with_references=("has_references_section", "sum"),
          rubric_facts_mentioned=("facts_hit", lambda s: sum(len(x) for x in s)), unsupported_claim_flags=("flags", lambda s: sum(len(x) for x in s)),
          ungrounded_numbers=("ungrounded_numbers", lambda s: sum(len(x) for x in s)))
summary["of"] = "3 prompts (16 rubric facts in total)"
summary
""")

    md(r"""
### B3.4 Speed and memory
""")
    code(r"""
spd = pd.DataFrame({"baseline tok/s (this run)": [r["tokens_per_second"] for r in results["baseline"]],
                    "adapter bf16 tok/s": [r["tokens_per_second"] for r in results["adapter_bf16"]],
                    "adapter 4-bit tok/s": [r["tokens_per_second"] for r in results["adapter_4bit"]],
                    "baseline tok/s (Part A)": [float(saved_a[p["id"]]["tokens_per_second"]) for p in PROMPTS]}, index=[p["id"] for p in PROMPTS])
display(spd.round(2)); print("mean tok/s:", spd.mean().round(2).to_dict()); print("peak GPU memory (GB):", mem)
""")

    code(r"""
# save the comparison (lab outputs go to reports_lab/) and compare the adapter outputs with the earlier run's
import difflib
cmp_rows = []
for i, p in enumerate(PROMPTS):
    b, a, a4 = (results[s][i] for s in ("baseline", "adapter_bf16", "adapter_4bit"))
    cmp_rows.append({"prompt_id": p["id"], "baseline_output": saved_a[p["id"]]["output"], "adapter_output": a["output"], "baseline_tps": b["tokens_per_second"],
                     "adapter_tps": a["tokens_per_second"], "baseline_new_tokens": b["new_tokens"], "adapter_new_tokens": a["new_tokens"],
                     "adapter_stopped_on_eos": a["stopped_on_eos"], "adapter_has_disclaimer": a["output"].strip().endswith(DISCLAIMER),
                     "adapter_4bit_output": a4["output"], "adapter_4bit_tps": a4["tokens_per_second"]})
pd.DataFrame(cmp_rows).to_csv(REPORTS_LAB / "baseline_vs_adapter.csv", index=False)
sc.drop(columns=["facts_hit", "flags", "ungrounded_numbers"]).to_csv(REPORTS_LAB / "adapter_eval_scores.csv", index=False)
old_cmp = pd.read_csv(ROOT / "reports" / "baseline_vs_adapter.csv").set_index("prompt_id")
print("adapter (bf16) outputs identical to the earlier run's:", {r["prompt_id"]: r["adapter_output"].split() == old_cmp.loc[r["prompt_id"], "adapter_output"].split() for r in cmp_rows})
print("adapter (4-bit) outputs identical to the earlier run's:", {r["prompt_id"]: r["adapter_4bit_output"].split() == old_cmp.loc[r["prompt_id"], "adapter_4bit_output"].split() for r in cmp_rows})
for r in cmp_rows:
    old = old_cmp.loc[r["prompt_id"], "adapter_output"]
    if old.split() != r["adapter_output"].split():
        print(f"\n[{r['prompt_id']}] earlier-run adapter output for comparison:\n{old}")
""")

    md(r"""
### Analysis: what improved and what degraded

**Scores** (rubric facts are mention-based and flatter fluent answers; the per-prompt reading below is what counts; 3 prompts, so everything here is qualitative).

| | Baseline (bf16) | Adapter B (bf16) | Adapter B (4-bit) |
|---|---|---|---|
| Ends with the required disclaimer | 0 of 3 | **3 of 3** | **3 of 3** |
| Stops by itself | 0 of 3 (all hit the 150-token cap, cut mid-sentence) | **3 of 3** (97, 128, 119 tokens) | **3 of 3** (79, 108, 119) |
| Invented `### Evidence` / `### References` / `### Notes` sections | 3 of 3 (2 with References) | **0 of 3** | **0 of 3** |
| Rubric facts mentioned (of 16) | 5 | 8 | 7 |
| Claims the corpus contradicts or does not support (flags) | 3 | 4 | 5 |

**Malaria, uncomplicated *P. falciparum* (WHO).** The corpus says children and adults should be treated with *one of six* ACTs (AL, AS+AQ, ASMQ, DHAP, AS+SP, ASPY), notes that AS+SP and ASPY are not recommended in the first trimester, and says the second line for failure within 28 days is *an alternative ACT*;
the 7-day artesunate or quinine regimens with doxycycline or clindamycin are "no longer generally recommended".
- *Baseline:* names AL, which is one of the six, so it is correct but incomplete. It then repeats the same sentence three times, adds invented Evidence, References and Notes sections (the reference is to a 2015 edition, not the 2024 guideline in our corpus) and is cut off mid-word.
- *Adapter (bf16):* concise, source named, and it uses the right class term "artemisinin-based combination therapy" (terminology improved). But it describes artesunate plus sulfadoxine-pyrimethamine "or amodiaquine plus SP", which is not one of the six treatment options, and an intravenous artesunate course for 3 days then oral therapy, which is the
  *severe*-malaria pattern (the corpus uses parenteral artesunate for severe malaria and then completes with an oral ACT), not uncomplicated malaria. The 4-bit adapter presents AS+SP as *the* first line and "quinine plus clindamycin" as second line, which the corpus says is no longer generally recommended; the correct second line is another ACT.
  Correctness got worse (confident and wrong); completeness did not improve (it names one or two of six options either way).

**Sepsis, IV antibiotics (NICE NG253).** The corpus says: for people at high risk give broad-spectrum IV antibiotics **within 1 hour of calculating the NEWS2 score**; when the source is unknown, broad-spectrum antibiotics within the timeframe for the person's risk category; once the source is confirmed, switch to source-specific antibiotics.
- *Baseline:* "as soon as possible, ideally within 1 hour of recognition" and broad-spectrum if the source is uncertain is the right gist, but it anchors the hour to "recognition", credits the Surviving Sepsis Campaign instead of NICE and adds fake Evidence and References sections.
- *Adapter (bf16):* the same gist and one genuine gain, "the regimen should be adjusted as soon as the source is identified", which matches the corpus's source-specific step. But it attributes the advice to "Sepsis-3 guidelines" and adds gram-negative, gram-positive and anaerobic coverage. The 4-bit adapter credits "SSC" and adds MRSA, Pseudomonas and local resistance patterns.
  **None of these terms (Sepsis-3, Surviving Sepsis, MRSA, Pseudomonas, gram-negative, anaerobic, local resistance) appears anywhere in the NICE text.** Neither system mentions risk-based timing, which is the actual NICE content. Fake sections are gone, but unsupported detail and a wrong attribution replace them.

**Hypertension, first line (NICE NG136).** The corpus: offer an **ACE inhibitor or ARB** to adults with type 2 diabetes (any age) or aged **under 55** and not of Black African or African-Caribbean origin; offer a **calcium-channel blocker** from age **55** (without type 2 diabetes) or for Black African or African-Caribbean origin; a thiazide-like diuretic **only** if a CCB is not tolerated or there is heart failure.
- *Baseline:* vague. It lists a thiazide-type diuretic, a CCB or an ACE inhibitor and says the choice depends on age and comorbidities, with no rule. Unhelpful, but it asserts nothing specific that is wrong beyond listing a thiazide as first line.
- *Adapter (both dtypes):* names the right source and the right dividing age (55), which is better terminology and a partial hit on the rubric. But it states a **thiazide-like diuretic as first line, and as the first choice "for people aged 55 or older"**, with a CCB or ACE inhibitor as alternatives. That inverts the guideline: at 55 or over the corpus says CCB, and a thiazide only if a CCB is not tolerated.
  This is the most serious degradation: a vague answer became a specific wrong one, on exactly the part of the question about age.

**Overall.** *Improved:* the response **form**. Every adapter answer opens with "Per ...", stays at two sentences, ends with the exact Variant 4 disclaimer and stops by itself; the invented sections and cut-off answers are gone; domain terms (ACT, ACE inhibitor or ARB, CCB, age 55) appear more often (rubric 5 to 8 of 16).
*Degraded or unchanged:* **correctness and hallucination**. Unsupported or contradicted claims rose from 3 to 4 (bf16) and 5 (4-bit), and the errors are now confident and specific (a wrong first line, a wrong age rule, invented antibiotic coverage and guideline names) in the tone of an authoritative source. *Completeness* did not improve: the adapter still omits five of the six ACTs,
the risk-based timing and the type 2 diabetes and family-origin rules.

**Why.** 40 training pairs, each teaching one fact, change *how* the model writes but cannot reliably change *what* it knows, and the model fills the learned answer shape with plausible clinical text from its pre-training. The eval loss fell by a third because the form is easy to learn; the facts are not.

**Precision, memory and speed.** The 4-bit adapter (the training condition) is close to bf16 in quality (7 versus 8 rubric facts; one more flagged claim) and needs about **4.4 GB instead of 14.6 GB** of GPU memory (the measured peaks are in the table above). It is also the slowest: in every run of this cell the 4-bit adapter generated at roughly **half to 60 %** of the bf16 adapter's speed, because each step dequantises the weights and applies the unmerged LoRA matrices. The bf16 adapter is itself **10 to 20 % slower than the bf16 baseline**, again because it is attached unmerged (extra matrix products per layer); merging it into a bf16 model before serving removes that cost. Absolute tokens per second are **not stable on this shared A100**: the same baseline cell measured about 24 tokens/s in the final run and about 31 tokens/s in an earlier run of identical code, so read the ratios and the table above, not single values. Part C repeats timings with warm-up and several runs.

**Reproducibility.** The regenerated baseline matches the Part A outputs and the earlier run on the other GPU token for token. The adapter outputs match the earlier run for 2 of 3 prompts in each precision; the malaria answer (bf16) and the sepsis answer (4-bit) differ in wording because the retrained adapter on a different library version differs by tiny weight differences that flip a greedy choice. The conclusions are the same in both runs.

**Implication for Variant 4 (clinical protocol lookup).** Fine-tuning on this set made the assistant *look* more like a protocol assistant (source cited, disclaimer, clean stop) without making it more correct, which in a clinical setting is a risk: a fluent, cited, confident wrong dose or first-line choice is worse than a vague answer. The disclaimer is necessary but not sufficient. A deployable assistant needs grounding (retrieving the guideline passage into the prompt) and a larger, verified instruction set; the pairs here are a demonstration of the pipeline, not a clinical tool.
""")

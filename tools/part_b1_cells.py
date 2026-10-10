"""Part B1 cells for assignment_1b.ipynb. add(md, code) appends them. Cell sources are raw strings."""


def add(md, code):
    md(r"""
---
# Part B: Instruction fine-tuning with QLoRA (5 marks)

Part A gave a clean clinical corpus and a fluent but ungrounded baseline. Part B teaches the base model the *shape* of a protocol-lookup answer with a small
supervised set, then compares it with the baseline. Marks: B1 dataset (2), B2 QLoRA training (2), B3 evaluation (1). This build contains **B1**; B2 and B3 follow.

## B1: Instruction dataset creation (2 marks)

**Requirements from the brief and the instructor.** Source: only the cleaned `.txt` files of Part A. Format: JSONL with `instruction` and `response`, answer grounded in
the corpus, no outside facts. Method: synthetic generation is allowed, but then the **exact prompt template** and **5 sample pairs** must be shown. Split 80/20 with a
fixed seed and report both counts. The instructor's conditions: 40 to 50 pairs, a spot-check of 10 to 15 pairs, paraphrased answers of 1 to 3 sentences with at least 30 words
of clinical content, no single instruction template above 20 %, and question types spread across sources. Variant 4 adds the disclaimer *"This output is for educational/reference
purposes only and must not replace professional clinical judgment."* to every response.

**Our design (fixed before drafting).** Exactly **50 pairs**; 7 question types in a fixed mix (definition 8, explanation 7, procedure 8, comparison 7, troubleshooting 6, scenario 7,
advantages/limitations 7); 8 source families (WHO malaria, CDC STI, ICMR antimicrobial, NICE sepsis, NICE hypertension, ICMR STW vol 1 and vol 3, PubChem monographs), none above 30 % of the
pairs. Each pair is drafted from one page of the corpus (a *chunk*), chosen by a seeded procedure, by an LLM (Claude) using one committed prompt; the raw drafts are committed unedited, so
everything after drafting is deterministic code that can be re-run without any LLM call.
""")

    code(r"""
import random
from transformers import AutoTokenizer

SEED, PAIR_COUNT, EVAL_COUNT = 42, 50, 10
DISCLAIMER = "This output is for educational/reference purposes only and must not replace professional clinical judgment."
MIN_CLINICAL_WORDS, MAX_SENTENCES = 30, 3
MAX_TEMPLATE_SHARE, MAX_FAMILY_SHARE = 0.20, 0.30
VERBATIM_NGRAM, LEAKAGE_JACCARD = 8, 0.70
CHUNK_MIN_WORDS, CHUNK_MAX_WORDS = 250, 650

CORPUS_DIR = ROOT / "domain_corpus"
INSTR_DIR = ROOT / "data" / "instruction"
TYPE_MIX = {"definition": 8, "explanation": 7, "procedure": 8, "comparison": 7,
            "troubleshooting": 6, "scenario": 7, "advantages_limitations": 7}
FAMILIES = {
    "who_malaria": ["who_malaria_2024-11-30.txt"], "cdc_sti": ["cdc_sti_2021.txt"],
    "icmr_antimicrobial": ["icmr_antimicrobial_2019.txt"], "nice_sepsis": ["nice_ng253_sepsis.txt"],
    "nice_hypertension": ["nice_ng136_hypertension.txt"], "icmr_stw_vol1": ["icmr_stw_vol1_2019.txt"],
    "icmr_stw_vol3": ["icmr_stw_vol3_2022.txt"], "pubchem": sorted(p.name for p in CORPUS_DIR.glob("pubchem_*.txt")),
}
# pair type of each of the 50 slots, per family
SLOT_TYPES = {
    "who_malaria": ["procedure", "procedure", "troubleshooting", "troubleshooting", "scenario", "comparison", "definition", "advantages_limitations"],
    "cdc_sti": ["procedure", "procedure", "troubleshooting", "comparison", "comparison", "scenario", "explanation"],
    "icmr_antimicrobial": ["procedure", "procedure", "comparison", "explanation", "explanation", "definition", "troubleshooting"],
    "nice_sepsis": ["procedure", "scenario", "scenario", "explanation", "troubleshooting"],
    "nice_hypertension": ["procedure", "scenario", "comparison", "explanation", "advantages_limitations"],
    "icmr_stw_vol1": ["troubleshooting", "scenario", "comparison", "definition"],
    "icmr_stw_vol3": ["definition", "definition", "advantages_limitations", "advantages_limitations"],
    "pubchem": ["definition", "definition", "definition", "explanation", "explanation",
                "advantages_limitations", "advantages_limitations", "advantages_limitations", "comparison", "scenario"],
}
assert Counter(t for ts in SLOT_TYPES.values() for t in ts) == Counter(TYPE_MIX) and sum(TYPE_MIX.values()) == PAIR_COUNT

ABBREVIATIONS = ["e.g.", "i.e.", "vs.", "Dr.", "No.", "approx.", "etc.", "cf.", "spp.", "sp.", "subsp.", "St.", "Fig.", "ca.", "al."]
GENUS_INITIAL = re.compile(r"\b([A-Z])\.(?=\s+[a-z])")            # P. falciparum is not a sentence end

def b_words(text):      return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-/.%]*", text)
def split_clinical(r):  t = r.strip(); return t[: -len(DISCLAIMER)].strip() if t.endswith(DISCLAIMER) else None
def count_sentences(text):
    p = text
    for a in ABBREVIATIONS: p = p.replace(a, a.replace(".", "\u0000"))
    p = GENUS_INITIAL.sub("\\1\u0000", p)
    p = re.sub(r"(?<=\d)\.(?=\d)", "\u0000", p)                   # decimals
    return len([x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", p.strip()) if x.strip()])
def numbers_in(text):   return set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d{3})", "", text)))
def ngrams(tokens, n):  return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
def norm_tokens(text):  return re.findall(r"[a-z0-9]+", text.lower())
def jac(a, b):          return len(a & b) / len(a | b) if a | b else 0.0
def template_family(instruction):
    toks = re.sub(r"\b\d+(\.\d+)?\b", "<num>", instruction).split()
    kept = ["<term>" if i > 0 and t.strip("?,.;:()")[:1].isupper() else t.strip("?,.;:()").lower() for i, t in enumerate(toks)]
    return " ".join(kept[:4])

def read_jsonl(path):   return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
print("constants and helpers defined; slots:", sum(len(v) for v in SLOT_TYPES.values()), "| families:", len(FAMILIES))
""")

    md(r"""
### B1.1 Choosing the source pages (chunks), reproducibly

For each of the 50 slots the code ranks the usable pages of the slot's family by keyword fit to the slot's pair type, shuffles the top 6 with `random.Random(42)`, and keeps two pages: a primary and a
spare (the spare replaces a primary that turns out to be a contents page or an evidence table). Unusable pages (too short, mostly digits, references, dotted contents lines) are excluded, and no two chosen
pages of a family are adjacent. The result is compared with the committed `data/instruction/chunks.jsonl`.
""")
    code(r"""
PAGE_MARKER = re.compile(r"^<<<PAGE (\d+)>>>$", re.M)
def load_pages(fn):
    pieces = PAGE_MARKER.split((CORPUS_DIR / fn).read_text(encoding="utf-8"))
    return [(int(pieces[i]), pieces[i + 1].strip()) for i in range(1, len(pieces), 2)]
def truncate_page(text):
    toks = text.split()
    if len(toks) <= CHUNK_MAX_WORDS: return text
    cut = " ".join(toks[:CHUNK_MAX_WORDS]); end = cut.rfind(". ")
    return cut[: end + 1] if end > len(cut) // 2 else cut
def page_is_usable(text):
    if len(text.split()) < CHUNK_MIN_WORDS: return False
    text = truncate_page(text)
    L, D = sum(c.isalpha() for c in text), sum(c.isdigit() for c in text)
    if L / max(len(text), 1) < 0.70 or D / max(len(text), 1) > 0.08: return False
    low = text.lower()
    return not (low.count("doi") > 2 or low.count("et al") > 3 or "contents" in low[:200] or len(re.findall(r"\.{4,}", text)) > 2)

CLINICAL_TERMS = r"dose|mg|treat|recommend|should|patient|days|antibiotic|therapy|infection|drug"
NOISE_TERMS = r"certainty of (the )?evidence|95% ci|crct|research needs|judgement of the panel|references"
TYPE_TERMS = {
    "definition": r"is defined|defined as|refers to|is a |are a |known as|classified",
    "explanation": r"because|due to|mechanism|results? in|leads? to|reduces?|works?",
    "procedure": r"should be (given|started|offered|taken|administered)|dose|administer|step|regimen",
    "comparison": r"compared|versus|whereas|alternative|rather than|instead of|than",
    "troubleshooting": r"failure|resistan|adverse|contraindicat|avoid|vomit|not recommended|should not|if ",
    "scenario": r"pregnan|child|women|infant|allerg|elderly|aged|patients? (with|who)",
    "advantages_limitations": r"benefit|limit|risk|advantage|concern|efficacy|caution|side effect",
}
def type_score(text, pair_type):
    low = text.lower()
    return 3 * len(re.findall(TYPE_TERMS[pair_type], low)) + len(re.findall(CLINICAL_TERMS, low)) - 4 * len(re.findall(NOISE_TERMS, low))

def build_chunks():
    rng, chunks, slot_no = random.Random(SEED), [], 0
    for family, slot_types in SLOT_TYPES.items():
        cands = [(f, pg, truncate_page(t)) for f in FAMILIES[family] for pg, t in load_pages(f) if page_is_usable(t)]
        used = []
        for pair_type in slot_types:
            slot_no += 1
            free = [c for c in cands if all(not (c[0] == u[0] and abs(c[1] - u[1]) < 2) for u in used)]
            free.sort(key=lambda c: (-type_score(c[2], pair_type), c[0], c[1]))
            pool = free[:6]; rng.shuffle(pool)
            for rank, (fn, pg, text) in enumerate(pool[:2], 1):
                used.append((fn, pg))
                chunks.append({"chunk_id": f"s{slot_no:02d}r{rank}", "slot": slot_no, "rank": rank, "family": family,
                               "type": pair_type, "source_file": fn, "page": pg, "text": text})
    return chunks

chunks = build_chunks()
committed_chunks = read_jsonl(INSTR_DIR / "chunks.jsonl")
print(f"{len(chunks)} chunks (50 slots x primary + spare); identical to committed data/instruction/chunks.jsonl: {chunks == committed_chunks}")
assert chunks == committed_chunks
pd.DataFrame(chunks).groupby(["family", "type"]).size().unstack(fill_value=0)
""")

    md(r"""
### B1.2 Generation method: the exact prompt template and the drafter

The drafter is **Claude** (not Mistral), called once per chunk with the template below. It follows the brief's tip ("based ONLY on this text", `instruction` and `response` keys), adds the locked sentence
*"Each response must be a concise 1-3 sentence answer in your own words. Do not copy text verbatim."*, requires the source guideline to be named, the disclaimer to end the answer, and forbids any fact not in
the chunk. `{pair_type}`, `{source_file}`, `{page}` and `{chunk_text}` are filled per chunk.
""")
    code(r"""
print((ROOT / "prompts" / "instruction_draft_prompt.txt").read_text(encoding="utf-8"))
log = json.loads((INSTR_DIR / "drafting_log.json").read_text(encoding="utf-8"))
print("-" * 80); print("drafter       :", log["drafter"]); print("attempts      :", log["attempts_total"], "drafted,", log["attempts_selected"], "selected;",
      len(log["spares_used"]), "slots used the spare page"); print("raw drafts    : data/instruction/drafts_raw.jsonl (committed, unedited)")
""")

    md(r"""
### B1.3 Automated checks, 50-pair selection and the 40/10 split

Every draft must pass these per-pair checks, in code: the response **ends with the exact disclaimer** (once); the clinical text before it has **1 to 3 sentences** (one sentence splitter protects abbreviations,
genus initials such as *P. falciparum*, and decimals) and **at least 30 words**; **no 8-word run** is copied from the source page; **every number** (doses, days, thresholds) appears in the source page; the response
**names its source guideline**; the instruction is not a near-duplicate (word-set Jaccard >= 0.70) of any of the three fixed prompts used in Part A, B3 and C. Selection takes, per slot, the primary chunk's draft if it
passes, otherwise the spare's. Dataset-level checks then require exactly 50 rows with only the two keys, the locked type mix, no family above 30 %, **no instruction template above 20 %** (first 4 tokens after masking numbers
and capitalised terms) and at least two templates per type, no duplicate instructions, and all source files present. The split shuffles the ids with seed 42 and holds out 10.
""")
    code(r"""
def check_draft(draft, chunk):
    instruction, response = (draft.get("instruction") or "").strip(), (draft.get("response") or "").strip()
    if not instruction or not response: return ["empty instruction or response"]
    clinical = split_clinical(response)
    if clinical is None: return ["response does not end with the exact disclaimer"]
    errs = []
    if response.count(DISCLAIMER) != 1: errs.append("disclaimer appears more than once")
    if len(b_words(clinical)) < MIN_CLINICAL_WORDS: errs.append(f"only {len(b_words(clinical))} clinical words")
    if not 1 <= count_sentences(clinical) <= MAX_SENTENCES: errs.append(f"{count_sentences(clinical)} sentences")
    copied = ngrams(norm_tokens(clinical), VERBATIM_NGRAM) & ngrams(norm_tokens(chunk["text"]), VERBATIM_NGRAM)
    if copied: errs.append(f"{len(copied)} verbatim {VERBATIM_NGRAM}-word runs copied")
    missing = numbers_in(clinical) - numbers_in(chunk["text"])
    if missing: errs.append(f"numbers not in source: {sorted(missing)}")
    if not re.search(r"\b(WHO|CDC|NICE|ICMR|PubChem)\b", clinical): errs.append("source guideline not named")
    return errs

def check_leakage(instruction):
    toks = set(norm_tokens(instruction))
    for p in PROMPTS:                                              # the 3 fixed prompts from Part A
        if jac(toks, set(norm_tokens(p["instruction"]))) >= LEAKAGE_JACCARD: return f"near-duplicate of fixed prompt {p['id']}"
    return None

def dataset_errors(rows, meta, chunks_by_id):
    errs = []
    if len(rows) != PAIR_COUNT: errs.append(f"{len(rows)} rows")
    errs += [f"row {i} keys {sorted(r)}" for i, r in enumerate(rows) if set(r) != {"instruction", "response"}]
    if Counter(m["type"] for m in meta) != Counter(TYPE_MIX): errs.append("type mix differs from the locked mix")
    fam = Counter(m["family"] for m in meta)
    errs += [f"family {f} has {n} pairs (> 30%)" for f, n in fam.items() if n / PAIR_COUNT > MAX_FAMILY_SHARE]
    if set(fam) != set(FAMILIES): errs.append("not every family covered")
    tpl = Counter(m["template_family"] for m in meta)
    errs += [f"template '{t}' has {n} of {PAIR_COUNT} (> 20%)" for t, n in tpl.items() if n / PAIR_COUNT > MAX_TEMPLATE_SHARE]
    errs += [f"type {t} uses fewer than 2 templates" for t in TYPE_MIX if len({m["template_family"] for m in meta if m["type"] == t}) < 2]
    for m, row in zip(meta, rows):
        if not (CORPUS_DIR / m["source_file"]).is_file(): errs.append(f"{m['id']}: missing source file")
        errs += [f"{m['id']}: {e}" for e in check_draft(row, chunks_by_id[m["chunk_id"]])]
        if check_leakage(row["instruction"]): errs.append(f"{m['id']}: {check_leakage(row['instruction'])}")
    if len({r["instruction"].strip().lower() for r in rows}) != len(rows): errs.append("duplicate instructions")
    return errs

drafts = {}
for dr in read_jsonl(INSTR_DIR / "drafts_raw.jsonl"):             # latest attempt per chunk
    if dr["chunk_id"] not in drafts or dr.get("attempt", 1) >= drafts[dr["chunk_id"]].get("attempt", 1): drafts[dr["chunk_id"]] = dr

rejected, selected = [], []
for slot in sorted({c["slot"] for c in chunks}):
    for chunk in sorted((c for c in chunks if c["slot"] == slot), key=lambda c: c["rank"]):
        dr = drafts.get(chunk["chunk_id"])
        if dr is None: continue
        errs = check_draft(dr, chunk) + ([check_leakage(dr["instruction"])] if check_leakage(dr["instruction"]) else [])
        if errs: rejected.append({"chunk_id": chunk["chunk_id"], "reason": "; ".join(errs)}); continue
        selected.append((chunk, dr)); break
    else:
        raise SystemExit(f"slot {slot}: no passing draft")

rows = [{"instruction": dr["instruction"].strip(), "response": dr["response"].strip()} for _, dr in selected]
meta = [{"id": f"p{n:02d}", "type": c["type"], "family": c["family"], "source_file": c["source_file"], "page": c["page"],
         "chunk_id": c["chunk_id"], "template_family": template_family(dr["instruction"])} for n, (c, dr) in enumerate(selected, 1)]
errors = dataset_errors(rows, meta, {c["chunk_id"]: c for c in chunks})
print("dataset-level and per-pair check failures:", errors or "none")
assert not errors

# 80/20 split, seed 42
ids = [m["id"] for m in meta]; shuffled = ids[:]; random.Random(SEED).shuffle(shuffled)
train_ids, eval_ids = sorted(shuffled[EVAL_COUNT:]), sorted(shuffled[:EVAL_COUNT])
by_id = {m["id"]: r for m, r in zip(meta, rows)}
train = [{"id": i, **by_id[i]} for i in train_ids]; evals = [{"id": i, **by_id[i]} for i in eval_ids]

# reproduce the committed submission file exactly
same = {"instruction_dataset.jsonl": rows == read_jsonl(ROOT / "instruction_dataset.jsonl"),
        "pairs_meta.jsonl": meta == read_jsonl(INSTR_DIR / "pairs_meta.jsonl"),
        "train.jsonl": train == read_jsonl(INSTR_DIR / "train.jsonl"), "eval.jsonl": evals == read_jsonl(INSTR_DIR / "eval.jsonl")}
print("rebuilt files identical to the committed ones:", same)
assert all(same.values())
print(f"\ndrafts rejected by the checks and replaced by the spare page's draft or a later attempt: {len(rejected)}")
pd.DataFrame(rejected)
""")

    md(r"""
### B1.4 Five sample instruction-response pairs (required when an LLM drafts the data)

One pair each from five different pair types and five different source families, printed from `instruction_dataset.jsonl`.
""")
    code(r"""
picked, seen_t, seen_f = [], set(), set()
for m, r in zip(meta, rows):
    if m["type"] not in seen_t and m["family"] not in seen_f:
        picked.append((m, r)); seen_t.add(m["type"]); seen_f.add(m["family"])
    if len(picked) == 5: break
for m, r in picked:
    print(f"[{m['id']}] type={m['type']}  source={m['source_file']} p.{m['page']}\nINSTRUCTION: {r['instruction']}\nRESPONSE   : {r['response']}\n")
""")

    md(r"""
### B1.5 Dataset validation: counts, coverage, length

The tables below are computed from the rebuilt rows (all assertions above passed). They show the type mix, the source-family shares (cap 30 %), the template concentration (cap 20 %) and the length of
the clinical text (rule: 1 to 3 sentences, at least 30 words).
""")
    code(r"""
mm = pd.DataFrame(meta)
print("Pairs by type (locked mix):", dict(Counter(mm["type"])), "| mix matches:", Counter(mm["type"]) == Counter(TYPE_MIX))
print("Pairs by source family   :", dict(Counter(mm["family"])), "| max share:", f"{mm['family'].value_counts().max() / PAIR_COUNT:.0%} (cap 30%)")
tc = mm["template_family"].value_counts()
print(f"Distinct instruction templates: {len(tc)} of 50; most common '{tc.index[0]}' x{tc.iloc[0]} = {tc.iloc[0] / PAIR_COUNT:.0%} (cap 20%)")
cw = [len(b_words(split_clinical(r["response"]))) for r in rows]; cs = [count_sentences(split_clinical(r["response"])) for r in rows]
print(f"Clinical words per response: min {min(cw)}, mean {sum(cw) / len(cw):.1f}, max {max(cw)} (rule >= 30) | sentences: {dict(Counter(cs))} (rule 1-3)")
print("Responses ending with the exact disclaimer:", sum(r["response"].endswith(DISCLAIMER) for r in rows), "of", len(rows))
print("Instructions that near-duplicate a fixed prompt:", sum(bool(check_leakage(r["instruction"])) for r in rows))
mm.groupby(["family", "type"]).size().unstack(fill_value=0)
""")

    md(r"""
### B1.6 Dataset size and token budget (justification)

The Mistral-7B tokenizer is applied to the full training text (instruction block, answer and end-of-sequence token). The result sets the maximum sequence length for B2 and shows how small the job is.
""")
    code(r"""
tok_b = AutoTokenizer.from_pretrained(MODEL_ID)
fmt_text = lambda r: f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['response']}"
lengths = sorted(len(tok_b(fmt_text(r)).input_ids) + 1 for r in rows)          # +1 for the EOS appended in training
p95 = lengths[int(0.95 * len(lengths))]
tok_report = {"mean_tokens": round(sum(lengths) / len(lengths), 1), "p95_tokens": p95, "max_tokens": lengths[-1], "min_tokens": lengths[0],
              "tokens_one_epoch_on_40_train_rows": sum(len(tok_b(fmt_text(r)).input_ids) + 1 for r in train)}
print(tok_report)
print("max_seq_length 256 covers every example:", lengths[-1] <= 256)
trainable = 32 * 16 * ((4096 + 4096) + (4096 + 1024))      # Adapter B on Mistral-7B: q_proj 4096->4096, v_proj 4096->1024, r = 16
print(f"Adapter B trainable parameters (r=16, q_proj+v_proj): {trainable:,} = {trainable / 7_241_732_096:.3%} of the base model")
""")

    md(r"""
**Why 50 pairs (40 for training) is the right size here.** The aim is to teach the *form* of an answer (name the guideline, 1 to 3 sentences, disclaimer, stop), not new medicine: Mistral-7B already knows the vocabulary, and the
brief caps the dataset to what the GPU and time allow. The measured mean is about 174 tokens per example (p95 219, max 239), so one epoch over 40 rows is about 7,000 tokens: at batch 1 with gradient accumulation 4, three epochs are
30 optimizer steps, which fits in 4-bit on the A100 in about a minute. A maximum sequence length of **256** covers every example without truncation (512 would only add padding). A larger set would add LLM-drafting cost and drafting
noise without changing what the adapter can learn at this scale, and a higher-capacity adapter (rank 32) on 40 rows would over-fit; **Adapter B (r=16, alpha=32, q_proj and v_proj)** is the balanced choice, with about 6.8 M trainable
parameters (0.09 % of the model).

Mistral-7B-v0.1 is a **base model with no chat template**, so the "model's chat/instruction format" for SFT is the plain block `### Instruction:` / `### Response:` used here, in the Part A baseline, and in later parts.
""")

    md(r"""
### B1.7 Manual spot-check (12 pairs, seeded sample)

The 12 ids come from `random.Random(42).sample` over the 50 ids, so the sample is reproducible. Each pair was read against its source page for correctness, grounding, the disclaimer and copying
(the instructor asked for 10 to 15 pairs). The code checks above already cover the disclaimer and copying for all 50 pairs; the reading adds a judgement of factual correctness against the page. Three pairs are shown next to their source text.
""")
    code(r"""
spot = pd.read_csv(ROOT / "reports" / "instruction_spotcheck.csv")
expected = sorted(random.Random(SEED).sample([m["id"] for m in meta], 12))
print("spot-check ids equal the seeded sample:", list(spot["id"]) == expected, "| pairs reviewed:", len(spot))
print("verdicts:", dict(Counter(spot["verdict"])), "\n")
print(spot.drop(columns=["source_file"]).to_string(index=False, max_colwidth=110))

chunk_by_id = {c["chunk_id"]: c for c in chunks}
print("\n" + "=" * 100 + "\nThree reviewed pairs next to their source page:")
for pid in list(spot["id"])[:3]:
    m = next(x for x in meta if x["id"] == pid)
    print(f"\n[{pid}] {m['type']} | {m['source_file']} p.{m['page']}\nINSTRUCTION: {by_id[pid]['instruction']}\nRESPONSE   : {by_id[pid]['response']}\nSOURCE PAGE (first 700 chars): {chunk_by_id[m['chunk_id']]['text'][:700]}")
""")

    md(r"""
### B1.8 Train / eval split (80/20, seed 42)
""")
    code(r"""
tmeta = {m["id"]: m["type"] for m in meta}
print(f"train: {len(train)} examples | eval: {len(evals)} examples | seed {SEED} | disjoint ids: {not set(train_ids) & set(eval_ids)}")
print("train types:", dict(Counter(tmeta[i] for i in train_ids)))
print("eval types :", dict(Counter(tmeta[i] for i in eval_ids)))
print("eval ids   :", eval_ids)
""")

    md(r"""
### Inference: B1

- **What the checks prove.** Format, counts, copying, number grounding, the disclaimer and the template cap are verified for all 50 pairs by code, and the whole dataset is rebuilt from the committed raw drafts and matches the
  submission file exactly, so the dataset is reproducible without calling any LLM. The 12-pair reading adds a human judgement of correctness against the source page.
- **What they do not prove.** A paraphrase can still drop a caveat (the spot-check notes one such case, a hepatitis C screening exception). The pairs are LLM-drafted and inherit its wording habits; the 20 % template cap, the
  family cap and the committed raw drafts limit and expose that. The number check also cannot catch a correct number attached to the wrong claim.
- **Page choice.** Pages were chosen by seeded keyword fit to the pair type to avoid contents pages and evidence tables; nine slots used the spare page because the primary was one of those or too close to what the three fixed
  prompts ask. None of the 50 instructions near-duplicates the three fixed prompts, so B3 and Part C measure generalisation, not recall.
- **Eval split.** Ten rows are too few for a stable loss estimate and the held-out set happens to contain no `definition` pair, so B3 relies on the three fixed prompts and reports eval loss only as a rough signal.
- **Next.** B2 trains Adapter B on the 40 training rows with maximum sequence length 256, EOS appended to every text and loss on the response only.
""")

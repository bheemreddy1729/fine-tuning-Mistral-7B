"""Part B1: instruction dataset (50 pairs) built only from domain_corpus/.

Steps, all from the repo root:

    python src/instruction_dataset.py chunks     # pick source chunks (seed 42)
    python src/instruction_dataset.py build      # drafts -> 50 pairs, checks, split
    python src/instruction_dataset.py validate   # re-run every check on the files on disk

Drafts come from Claude. The exact prompt is prompts/instruction_draft_prompt.txt.
Raw drafts are kept unedited in data/instruction/drafts_raw.jsonl, so `build`
runs without calling any LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import BASELINE_PROMPTS, MODEL_ID, format_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "domain_corpus"
INSTR_DIR = ROOT / "data" / "instruction"
REPORTS_DIR = ROOT / "reports"
PROMPT_PATH = ROOT / "prompts" / "instruction_draft_prompt.txt"
CHUNKS_PATH = INSTR_DIR / "chunks.jsonl"
DRAFTS_PATH = INSTR_DIR / "drafts_raw.jsonl"
DRAFT_LOG_PATH = INSTR_DIR / "drafting_log.json"
META_PATH = INSTR_DIR / "pairs_meta.jsonl"
TRAIN_PATH = INSTR_DIR / "train.jsonl"
EVAL_PATH = INSTR_DIR / "eval.jsonl"
SUBMISSION_PATH = ROOT / "instruction_dataset.jsonl"
REJECTED_PATH = REPORTS_DIR / "instruction_rejected.csv"
CHECKS_PATH = REPORTS_DIR / "instruction_checks.json"
SPOTCHECK_PATH = REPORTS_DIR / "instruction_spotcheck.csv"

SEED = 42
PAIR_COUNT = 50
EVAL_COUNT = 10
DISCLAIMER = (
    "This output is for educational/reference purposes only and must not replace "
    "professional clinical judgment."
)
MIN_CLINICAL_WORDS = 30
MAX_SENTENCES = 3
MAX_TEMPLATE_SHARE = 0.20
MAX_FAMILY_SHARE = 0.30
VERBATIM_NGRAM = 8
LEAKAGE_JACCARD = 0.70
CHUNK_MIN_WORDS, CHUNK_MAX_WORDS = 250, 650
MAX_SEQ_CANDIDATE = 512

TYPE_MIX = {
    "definition": 8,
    "explanation": 7,
    "procedure": 8,
    "comparison": 7,
    "troubleshooting": 6,
    "scenario": 7,
    "advantages_limitations": 7,
}

# family -> source files and the pair type of each slot (50 slots in total).
FAMILIES = {
    "who_malaria": ["who_malaria_2024-11-30.txt"],
    "cdc_sti": ["cdc_sti_2021.txt"],
    "icmr_antimicrobial": ["icmr_antimicrobial_2019.txt"],
    "nice_sepsis": ["nice_ng253_sepsis.txt"],
    "nice_hypertension": ["nice_ng136_hypertension.txt"],
    "icmr_stw_vol1": ["icmr_stw_vol1_2019.txt"],
    "icmr_stw_vol3": ["icmr_stw_vol3_2022.txt"],
    "pubchem": sorted(p.name for p in CORPUS_DIR.glob("pubchem_*.txt")),
}
SLOT_TYPES = {
    "who_malaria": ["procedure", "procedure", "troubleshooting", "troubleshooting",
                    "scenario", "comparison", "definition", "advantages_limitations"],
    "cdc_sti": ["procedure", "procedure", "troubleshooting", "comparison", "comparison",
                "scenario", "explanation"],
    "icmr_antimicrobial": ["procedure", "procedure", "comparison", "explanation",
                           "explanation", "definition", "troubleshooting"],
    "nice_sepsis": ["procedure", "scenario", "scenario", "explanation", "troubleshooting"],
    "nice_hypertension": ["procedure", "scenario", "comparison", "explanation",
                          "advantages_limitations"],
    "icmr_stw_vol1": ["troubleshooting", "scenario", "comparison", "definition"],
    "icmr_stw_vol3": ["definition", "definition", "advantages_limitations",
                      "advantages_limitations"],
    "pubchem": ["definition", "definition", "definition", "explanation", "explanation",
                "advantages_limitations", "advantages_limitations", "advantages_limitations",
                "comparison", "scenario"],
}
PAGE_MARKER = re.compile(r"^<<<PAGE (\d+)>>>$", re.M)
ABBREVIATIONS = ["e.g.", "i.e.", "vs.", "Dr.", "No.", "approx.", "etc.", "cf.", "spp.", "sp.",
                 "subsp.", "St.", "Fig.", "ca.", "al."]
# A letter + dot at a word start (P. falciparum, S. aureus) is a genus initial, not a sentence end.
GENUS_INITIAL = re.compile(r"\b([A-Z])\.(?=\s+[a-z])")


# ---------------------------------------------------------------- text helpers
def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-/.%]*", text)


def split_clinical(response: str) -> str | None:
    """Return the text before the disclaimer, or None if it does not end with it."""
    text = response.strip()
    if not text.endswith(DISCLAIMER):
        return None
    return text[: -len(DISCLAIMER)].strip()


def count_sentences(text: str) -> int:
    """One splitter for the whole project. Protects abbreviations, genus initials, decimals."""
    protected = text
    for abbr in ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "\u0000"))
    protected = GENUS_INITIAL.sub("\\1\u0000", protected)
    protected = re.sub(r"(?<=\d)\.(?=\d)", "\u0000", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", protected.strip())
    return len([p for p in parts if p.strip()])


def numbers_in(text: str) -> set[str]:
    cleaned = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return set(re.findall(r"\d+(?:\.\d+)?", cleaned))


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def template_family(instruction: str) -> str:
    """First 4 tokens after lowercasing and replacing digits and capitalised terms with slots."""
    masked = re.sub(r"\b\d+(\.\d+)?\b", "<num>", instruction)
    tokens = masked.split()
    kept = []
    for i, tok in enumerate(tokens):
        word = tok.strip("?,.;:()")
        if i > 0 and word[:1].isupper():
            kept.append("<term>")
        else:
            kept.append(word.lower())
    return " ".join(kept[:4])


# ---------------------------------------------------------------- corpus pages
def load_pages(file_name: str) -> list[tuple[int, str]]:
    text = (CORPUS_DIR / file_name).read_text(encoding="utf-8")
    pieces = PAGE_MARKER.split(text)
    return [(int(pieces[i]), pieces[i + 1].strip()) for i in range(1, len(pieces), 2)]


def truncate_page(text: str) -> str:
    """Long pages are cut to their first CHUNK_MAX_WORDS words at a sentence end."""
    tokens = text.split()
    if len(tokens) <= CHUNK_MAX_WORDS:
        return text
    cut = " ".join(tokens[:CHUNK_MAX_WORDS])
    end = cut.rfind(". ")
    return cut[: end + 1] if end > len(cut) // 2 else cut


def page_is_usable(text: str) -> bool:
    if len(text.split()) < CHUNK_MIN_WORDS:
        return False
    text = truncate_page(text)
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    if letters / max(len(text), 1) < 0.70 or digits / max(len(text), 1) > 0.08:
        return False
    lowered = text.lower()
    if lowered.count("doi") > 2 or lowered.count("et al") > 3 or "contents" in lowered[:200]:
        return False
    if len(re.findall(r"\.{4,}", text)) > 2:  # dotted table-of-contents lines
        return False
    return True


# ---------------------------------------------------------------- chunks step
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


def type_score(text: str, pair_type: str) -> int:
    lowered = text.lower()
    return (3 * len(re.findall(TYPE_TERMS[pair_type], lowered))
            + len(re.findall(CLINICAL_TERMS, lowered))
            - 4 * len(re.findall(NOISE_TERMS, lowered)))


def build_chunks() -> list[dict]:
    """Per slot, one primary chunk (rank 1) and one spare (rank 2).

    Candidates are usable pages of the family. For each slot the pages are ranked by
    keyword fit to the slot's pair type, the top 6 unused pages are shuffled with seed 42,
    and the first two become the primary and the spare. Pages are never adjacent.
    """
    rng = random.Random(SEED)
    chunks, slot_no = [], 0
    for family, slot_types in SLOT_TYPES.items():
        candidates = [(f, page, truncate_page(text))
                      for f in FAMILIES[family] for page, text in load_pages(f)
                      if page_is_usable(text)]
        used: list[tuple[str, int]] = []
        for pair_type in slot_types:
            slot_no += 1
            free = [c for c in candidates
                    if all(not (c[0] == u[0] and abs(c[1] - u[1]) < 2) for u in used)]
            free.sort(key=lambda c: (-type_score(c[2], pair_type), c[0], c[1]))
            pool = free[:6]
            rng.shuffle(pool)
            if len(pool) < 2:
                raise SystemExit(f"not enough usable pages for {family}")
            for rank, (file_name, page, text) in enumerate(pool[:2], 1):
                used.append((file_name, page))
                chunks.append({
                    "chunk_id": f"s{slot_no:02d}r{rank}",
                    "slot": slot_no,
                    "rank": rank,
                    "family": family,
                    "type": pair_type,
                    "source_file": file_name,
                    "page": page,
                    "text": text,
                })
    return chunks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_chunks() -> list[dict]:
    chunks = build_chunks()
    expected = Counter(t for types in SLOT_TYPES.values() for t in types)
    if expected != Counter(TYPE_MIX) or sum(expected.values()) != PAIR_COUNT:
        raise SystemExit(f"slot types do not match the locked mix: {dict(expected)}")
    write_jsonl(CHUNKS_PATH, chunks)
    print(f"wrote {len(chunks)} chunks ({PAIR_COUNT} slots x 2) to {CHUNKS_PATH}")
    return chunks


# ---------------------------------------------------------------- checks
def check_draft(draft: dict, chunk: dict) -> list[str]:
    """Per-pair checks. Returns the list of failures (empty = pass)."""
    errors = []
    instruction = (draft.get("instruction") or "").strip()
    response = (draft.get("response") or "").strip()
    if not instruction or not response:
        return ["empty instruction or response"]
    clinical = split_clinical(response)
    if clinical is None:
        return ["response does not end with the exact disclaimer"]
    if response.count(DISCLAIMER) != 1:
        errors.append("disclaimer appears more than once")
    n_words = len(words(clinical))
    if n_words < MIN_CLINICAL_WORDS:
        errors.append(f"only {n_words} clinical words (<{MIN_CLINICAL_WORDS})")
    n_sent = count_sentences(clinical)
    if not 1 <= n_sent <= MAX_SENTENCES:
        errors.append(f"{n_sent} sentences (need 1-{MAX_SENTENCES})")
    chunk_tokens = normalize_tokens(chunk["text"])
    resp_tokens = normalize_tokens(clinical)
    copied = ngrams(resp_tokens, VERBATIM_NGRAM) & ngrams(chunk_tokens, VERBATIM_NGRAM)
    if copied:
        errors.append(f"{len(copied)} verbatim {VERBATIM_NGRAM}-word runs copied from the chunk")
    missing = numbers_in(clinical) - numbers_in(chunk["text"])
    if missing:
        errors.append(f"numbers not in source chunk: {sorted(missing)}")
    if not re.search(r"\b(WHO|CDC|NICE|ICMR|PubChem)\b", clinical):
        errors.append("response does not name its source guideline (WHO, CDC, NICE, ICMR or PubChem)")
    return errors


def prompt_token_sets() -> list[set[str]]:
    return [set(normalize_tokens(p["instruction"])) for p in BASELINE_PROMPTS]


def check_leakage(instruction: str) -> str | None:
    tokens = set(normalize_tokens(instruction))
    for prompt, fixed in zip(BASELINE_PROMPTS, prompt_token_sets()):
        if jaccard(tokens, fixed) >= LEAKAGE_JACCARD:
            return f"near-duplicate of fixed prompt {prompt['id']}"
    return None


def dataset_errors(rows: list[dict], meta: list[dict], chunks_by_id: dict) -> list[str]:
    """Dataset-level checks 1-9. Every one must pass."""
    errors = []
    if len(rows) != PAIR_COUNT:
        errors.append(f"{len(rows)} rows, expected {PAIR_COUNT}")
    for i, row in enumerate(rows):
        if set(row) != {"instruction", "response"}:
            errors.append(f"row {i} keys are {sorted(row)}")
        if not row.get("instruction", "").strip() or not row.get("response", "").strip():
            errors.append(f"row {i} has an empty field")
    types = Counter(m["type"] for m in meta)
    if types != Counter(TYPE_MIX):
        errors.append(f"type counts {dict(types)} differ from {TYPE_MIX}")
    families = Counter(m["family"] for m in meta)
    for family, n in families.items():
        if n / PAIR_COUNT > MAX_FAMILY_SHARE:
            errors.append(f"family {family} has {n} pairs (>{MAX_FAMILY_SHARE:.0%})")
    if set(families) != set(FAMILIES):
        errors.append(f"families covered {sorted(families)} != {sorted(FAMILIES)}")
    templates = Counter(m["template_family"] for m in meta)
    for tpl, n in templates.items():
        if n / PAIR_COUNT > MAX_TEMPLATE_SHARE:
            errors.append(f"template '{tpl}' has {n} of {PAIR_COUNT} (>{MAX_TEMPLATE_SHARE:.0%})")
    for pair_type in TYPE_MIX:
        distinct = {m["template_family"] for m in meta if m["type"] == pair_type}
        if len(distinct) < 2:
            errors.append(f"type {pair_type} uses only {len(distinct)} template family")
    for m, row in zip(meta, rows):
        if not (CORPUS_DIR / m["source_file"]).is_file():
            errors.append(f"{m['id']}: source file {m['source_file']} not in domain_corpus/")
        chunk = chunks_by_id[m["chunk_id"]]
        for err in check_draft(row, chunk):
            errors.append(f"{m['id']}: {err}")
        leak = check_leakage(row["instruction"])
        if leak:
            errors.append(f"{m['id']}: {leak}")
    if len({row["instruction"].strip().lower() for row in rows}) != len(rows):
        errors.append("duplicate instructions")
    return errors


# ---------------------------------------------------------------- build step
def latest_drafts() -> dict[str, dict]:
    """Latest attempt per chunk_id from drafts_raw.jsonl."""
    best: dict[str, dict] = {}
    for draft in read_jsonl(DRAFTS_PATH):
        cid = draft["chunk_id"]
        if cid not in best or draft.get("attempt", 1) >= best[cid].get("attempt", 1):
            best[cid] = draft
    return best


def split_rows(ids: list[str]) -> tuple[list[str], list[str]]:
    """80/20 split, seed 42, stratification-free but each type must appear in train."""
    rng = random.Random(SEED)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    return shuffled[EVAL_COUNT:], shuffled[:EVAL_COUNT]


def run_build() -> dict:
    chunks = read_jsonl(CHUNKS_PATH)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    drafts = latest_drafts()
    rejected, selected = [], []
    for slot in sorted({c["slot"] for c in chunks}):
        slot_chunks = sorted((c for c in chunks if c["slot"] == slot), key=lambda c: c["rank"])
        chosen = None
        for chunk in slot_chunks:
            draft = drafts.get(chunk["chunk_id"])
            if draft is None:
                continue
            errors = check_draft(draft, chunk)
            leak = check_leakage(draft.get("instruction", ""))
            if leak:
                errors.append(leak)
            if errors:
                rejected.append({"chunk_id": chunk["chunk_id"], "reason": "; ".join(errors)})
                continue
            chosen = (chunk, draft)
            break
        if chosen is None:
            raise SystemExit(f"slot {slot}: no passing draft (see {REJECTED_PATH})")
        selected.append(chosen)
    with REJECTED_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chunk_id", "reason"])
        writer.writeheader()
        writer.writerows(rejected)
    rows, meta = [], []
    for n, (chunk, draft) in enumerate(selected, 1):
        rows.append({"instruction": draft["instruction"].strip(), "response": draft["response"].strip()})
        meta.append({
            "id": f"p{n:02d}",
            "type": chunk["type"],
            "family": chunk["family"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "template_family": template_family(draft["instruction"]),
        })
    errors = dataset_errors(rows, meta, chunks_by_id)
    if errors:
        raise SystemExit("dataset checks failed:\n  " + "\n  ".join(errors))
    write_jsonl(SUBMISSION_PATH, rows)
    write_jsonl(META_PATH, meta)
    train_ids, eval_ids = split_rows([m["id"] for m in meta])
    by_id = {m["id"]: row for m, row in zip(meta, rows)}
    write_jsonl(TRAIN_PATH, [{"id": i, **by_id[i]} for i in sorted(train_ids)])
    write_jsonl(EVAL_PATH, [{"id": i, **by_id[i]} for i in sorted(eval_ids)])
    return run_validate()


# ---------------------------------------------------------------- validate + report
SPOTCHECK_COUNT = 12
SPOTCHECK_FIELDS = ["id", "type", "source_file", "correct", "grounded", "disclaimer", "not_verbatim", "verdict", "note"]


def spotcheck_ids(meta: list[dict]) -> list[str]:
    """Seeded sample of pairs for manual review (reproducible)."""
    return sorted(random.Random(SEED).sample([m["id"] for m in meta], SPOTCHECK_COUNT))


def spotcheck_errors(meta: list[dict]) -> list[str]:
    """The manual review file must exist, cover exactly the seeded sample, and give a verdict per row."""
    if not SPOTCHECK_PATH.is_file():
        return [f"missing {SPOTCHECK_PATH.name}"]
    with SPOTCHECK_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    if [r["id"] for r in rows] != spotcheck_ids(meta):
        errors.append("spot-check ids do not match the seeded sample")
    for r in rows:
        if r["verdict"] not in {"pass", "fail"} or not r["note"].strip():
            errors.append(f"spot-check row {r['id']} has no verdict or note")
        if r["verdict"] == "fail":
            errors.append(f"spot-check row {r['id']} failed: {r['note']}")
    return errors


def token_report(rows: list[dict]) -> dict:
    """Token lengths of the full training text with the Mistral tokenizer (+1 for EOS)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    lengths = sorted(len(tok(format_prompt(r["instruction"]) + r["response"]).input_ids) + 1
                     for r in rows)
    p95 = lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]
    return {
        "tokenizer": MODEL_ID,
        "mean_tokens": round(sum(lengths) / len(lengths), 1),
        "p95_tokens": p95,
        "max_tokens": lengths[-1],
        "min_tokens": lengths[0],
        "total_tokens_one_epoch": sum(lengths),
        "proposed_max_seq_length": 256 if lengths[-1] <= 256 else (MAX_SEQ_CANDIDATE if p95 <= MAX_SEQ_CANDIDATE else None),
    }


def run_validate(with_tokens: bool = True) -> dict:
    rows = read_jsonl(SUBMISSION_PATH)
    meta = read_jsonl(META_PATH)
    chunks_by_id = {c["chunk_id"]: c for c in read_jsonl(CHUNKS_PATH)}
    errors = dataset_errors(rows, meta, chunks_by_id)
    train, evals = read_jsonl(TRAIN_PATH), read_jsonl(EVAL_PATH)
    if (len(train), len(evals)) != (PAIR_COUNT - EVAL_COUNT, EVAL_COUNT):
        errors.append(f"split is {len(train)}/{len(evals)}")
    if {r["id"] for r in train} & {r["id"] for r in evals}:
        errors.append("train and eval share ids")
    errors += spotcheck_errors(meta)
    _, exp_eval = split_rows([m["id"] for m in meta])
    if {r["id"] for r in evals} != set(exp_eval):
        errors.append("eval ids do not match the seed-42 split")
    if errors:
        raise SystemExit("validation failed:\n  " + "\n  ".join(errors))
    clinical_words = [len(words(split_clinical(r["response"]))) for r in rows]
    report = {
        "seed": SEED,
        "pairs": len(rows),
        "train": len(train),
        "eval": len(evals),
        "type_counts": dict(Counter(m["type"] for m in meta)),
        "family_counts": dict(Counter(m["family"] for m in meta)),
        "template_counts": dict(Counter(m["template_family"] for m in meta)),
        "max_template_share": round(max(Counter(m["template_family"] for m in meta).values()) / len(rows), 3),
        "clinical_words_mean": round(sum(clinical_words) / len(rows), 1),
        "clinical_words_min": min(clinical_words),
        "clinical_words_max": max(clinical_words),
        "eval_type_counts": dict(Counter(
            next(m["type"] for m in meta if m["id"] == r["id"]) for r in evals)),
        "checks_passed": True,
    }
    if with_tokens:
        report["tokens"] = token_report(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    CHECKS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["chunks", "build", "validate"])
    args = parser.parse_args()
    {"chunks": run_chunks, "build": run_build, "validate": run_validate}[args.command]()


if __name__ == "__main__":
    main()

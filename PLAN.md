# Assignment 1B plan

Reply with one id, for example `accept A1`. Work starts only for that item. The next item stays pending until you accept it.

Current item: **B1 accepted (in progress).** Part A is complete (corpus + bfloat16 baseline). B2 and B3 stay closed until the previous item is done. The Part B plan below was revised after review against `Assignment-1B.pdf` and Variant 4 of `Enterprise_Variants_All_Assignments.pdf`.

## Locked already

- Domain: Medical and Clinical Literature. Use case: Variant 4 Clinical Protocol Lookup Assistant.
- Model: `mistralai/Mistral-7B-v0.1` (base; no official chat template). Tokenizer: `AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")` for every Mistral load.
- Adapter for Part B: **Adapter B only** (`r=16`, `alpha=32`, targets `q_proj`, `v_proj`).
- Draft model for Part C speculative decoding: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, with its own tokenizer. Do not swap tokenizers.
- Instruction block for training and evaluation (Mistral has no chat template, so this is the SFT format):

```text
### Instruction:
{instruction}

### Response:
{response}
```

- JSONL target: exactly **50 pairs** with this mix — definition 8 (16%), explanation 7, procedure 8, comparison 7, troubleshooting 6, scenario 7, advantages/limitations 7.
- Each response: 1–3 sentences, ≥30 words of clinical content (disclaimer does not count toward the 30), paraphrased from `domain_corpus/` only, then the exact Variant 4 disclaimer: `This output is for educational/reference purposes only and must not replace professional clinical judgment.`
- No instruction template may account for more than 20% of the 50 instructions.
- Split: 80/20 with seed 42 → **40 train / 10 eval**.
- Spot-check: 10–15 pairs reviewed in the notebook (correctness, grounding, disclaimer, no verbatim copy).
- If an LLM drafts pairs, the generation prompt must include: `Each response must be a concise 1–3 sentence answer in your own words. Do not copy text verbatim.`
- The same 3 clinical prompts from Part A Step 2 are reused in Part B3 and Part C.
- Kept PDF pages after the three filters: 1,122 (≥300). Public repo: https://github.com/bheemreddy1729/fine-tuning-Mistral-7B
- GPU used for baseline: NVIDIA RTX A6000 (46 GB). Part B QLoRA can run on that pod or on BITS A100/L40S. Cost tables in Part C must use the hourly rate that matches the GPU actually used (brief lists ₹12/hour for A100).

## Project-wide implementation rule

- **Execution environment.** The user runs the code in their own Jupyter instance (BITS or pod GPU; the browser token is shared in chat when needed). Every item therefore ships as a `.ipynb` whose cells call `src/` and keep their outputs, an `.html` export of that notebook, and its files under `reports/`. Notebooks must run top to bottom on that instance. `requirements.txt` is the single source of dependencies: the first code cell of every notebook finds the repo root and runs `%pip install -q -r {ROOT}/requirements.txt`, so no manual install is needed. Each item that adds libraries (for example B2: torch, peft, trl==0.12.1, bitsandbytes, accelerate) adds them, pinned, to `requirements.txt` in the same change. Part A's notebooks get the same first cell the next time they are re-run. The token is used only for the actions the user asks for.

- **Cross-machine check (done).** The corpus pipeline and both Part A notebooks were re-run on the RTX A6000 pod after adding the install cell. Extraction now collapses runs of spaces, because pymupdf versions differ in how many spaces they emit; after that change `domain_corpus/`, `data/sources.json` and every corpus report are byte-identical on the Windows machine and the pod. The baseline re-run reproduced the three greedy outputs exactly (only timings differ).
- Every data transformation, validation, model run, benchmark, metric, and report must be produced by committed programmatic logic in this repository.
- Terminal commands may invoke project code or install dependencies, but must not contain the only implementation of an assignment step.
- Reusable logic belongs in `src/`; the notebook imports and calls it, displays results, and records interpretations.
- Inputs, configuration, thresholds, fixed seeds, and source metadata must be explicit and version-controlled.
- Generated artifacts must identify the code/configuration that produced them and must be reproducible by rerunning the documented command or notebook cell.
- Each accepted item includes its implementation, checks, and generated evidence. An item is not done if only its output files exist.
- A final clean-run check will rebuild all non-GPU artifacts from the documented entry point. GPU steps will use the same rule on Prayogshala.

## Part A — corpus

Code will live in `src/corpus_pipeline.py`. The notebook only calls that pipeline and displays the tables. Reports go to `reports/corpus_stats.csv` and `reports/filter_checks.json`. Source metadata goes to `data/sources.json`. Raw PDFs go to `data/raw_pdfs/`. Kept text goes to `domain_corpus/`.

Every filter row uses the same columns: PDF count, page count, word count, character count, documents removed, pages removed. A check re-reads the kept text and fails the script when the rule is broken.

### A1 — Download the seven PDFs

Status: done

Download only these files and write `data/sources.json` with title, URL, license, and the page count measured from the file.

Measured with PyMuPDF: WHO malaria 463, CDC STI guidelines 192, NICE NG253 97, NICE NG136 52, ICMR antimicrobial guidelines 206, ICMR STW Volume 1 75, ICMR STW Volume 3 80. Total 1,165 pages across 7 PDFs. The CDC file is the CDC Stacks original because `cdc.gov` returned HTTP 403 to a scripted download.

| Source | Document |
|---|---|
| WHO | Guidelines for malaria, 30 November 2024 (CC BY-NC-SA 3.0 IGO; attribution required) |
| CDC | STI Treatment Guidelines, 2021 (US government work) |
| NICE | NG253 suspected sepsis in adults |
| NICE | NG136 hypertension in adults |
| ICMR | Treatment Guidelines for Antimicrobial Use in Common Syndromes, 2019 |
| ICMR | Standard Treatment Workflows, Volume 1, 2019 |
| ICMR | Standard Treatment Workflows, Volume 3, 2022 |

Done when all seven files are on disk and `sources.json` matches the measured page counts. No text extraction in this item.

### A2 — Extract text and record extraction stats

Status: done

Extract each PDF page by page. Write one text file per PDF under a staging folder, with a page marker for every page.

The extraction row records total PDF pages, pages with fewer than 30 letters, and characters extracted. Pages with fewer than 30 letters stay in this row. They are not counted as language-filter removals.

Run with `python src/corpus_pipeline.py extract`. Result: 7 PDFs, 1,165 pages, 609,037 words, 4,242,287 characters (spaces collapsed so every machine gives the same count), 27 pages with fewer than 30 letters. Staged text is in `data/extracted/`. The short-page list is `reports/extraction_short_pages.csv`.

Done when the extraction table exists for all seven PDFs and short pages are listed with file name and page number.

### A3 — Language filter and check

Status: done

Run with `python src/corpus_pipeline.py language`. Before: 7 PDFs, 1,165 pages. After: 7 PDFs, 1,137 pages. Twenty-seven pages under 30 letters were listed as non-content in `reports/language_noncontent_pages.csv` and were not counted as language removals. One page was removed as non-English (`reports/language_removed_pages.csv`: ICMR STW Volume 1, page 72, detected as `id`). The recheck passed: every kept page is English. Kept text is in `data/staged/after_language/`.

Library: `langdetect`. Set `DetectorFactory.seed = 0` before every detection, including the recheck.

A page is removed by this filter only when it has at least 30 letters and the detector does not return English. Pages under 30 letters were already counted in A2 and are dropped here as non-content, in their own list, outside the language-removal count.

Check: run the detector again on every kept page that has at least 30 letters. Every one of those pages must be English. The script stops on a failure.

Done when `corpus_stats.csv` has a before and after language row, and `filter_checks.json` records the language check.

### A4 — Length filter and check

Status: done

Run with `python src/corpus_pipeline.py length`. Before and after are the same: 7 PDFs, 1,137 pages, 608,691 words. No document was under 3 pages or 1,000 words, so `reports/length_removed_documents.csv` has a header and no dropped rows. The recheck passed. Kept text is in `data/staged/after_length/`.

Run this on the text that survived A3. Keep a document only when it has at least 3 pages and at least 1,000 words.

Check: recompute pages and words on every remaining file. A file under either cutoff fails the check. The report names every dropped file with its page count and word count.

The notebook prose for this step will justify the 3-page and 1,000-word rule.

Done when the before and after length row is saved and the length check passes or names the failures.

### A5 — Deduplication and check

Status: done

Run with `python src/corpus_pipeline.py dedup`. Before: 7 PDFs, 1,137 pages. After: 7 PDFs, 1,122 pages. Removed 2 exact pages and 13 near-duplicate pages. No whole document was removed. The recheck found no repeated hash and no kept page or document pair at Jaccard 0.90 or higher. Kept text is in `data/staged/after_dedup/`.

Normalize page text by lowercasing and collapsing whitespace.

- Exact copy: SHA-256 of the normalized page. Keep the first page. Drop later copies inside the same document and across documents.
- Near copy: Jaccard similarity on 5-word shingles of 0.90 or higher. Drop the later page.
- Whole documents at 0.90 or higher: drop the shorter document.
- "First" follows a fixed sort of source file name, then page number, so a rerun drops the same copy.

Check: rebuild hashes and pairwise similarities on the kept pages. No hash may occur twice. No kept pair may score 0.90 or higher. The script stops on a failure.

Done when the before and after dedup row is saved and the dedup check passes.

### A6 — PubChem monographs

Status: done

Run with `python src/corpus_pipeline.py pubchem`. Nine drugs, 104 section-pages, 38,327 words. Every drug is its own file. Language, length, and dedup checks passed. Dedup removed 0 section-pages. PDF gate is unchanged: 7 PDFs, 1,122 pages. PubChem does not count toward either gate.

Files are `domain_corpus/pubchem_*.txt`. Counts are in `reports/pubchem_drug_counts.csv` and `reports/pubchem_stats.csv`. `sources.json` labels the source as not a PDF.

Artemether-lumefantrine combination CID 6450800 has almost no clinical text, so the corpus uses the component records artemether CID 68911 and lumefantrine CID 6437380.

The first 13 clinical headings leave doxycycline at 759 words. Protein Binding, Record Description, and GHS Classification are English drug-information sections on the same record and bring doxycycline to 1,109 words, so it passes the same 1,000-word rule.

PubChem is not a PDF. Fetch the clinical sections for amoxicillin, doxycycline, vancomycin, ceftriaxone, azithromycin, metronidazole, artesunate, artemether, and lumefantrine.

Keep a drug as its own file when it has at least 1,000 words and at least 3 section-pages. Merge shorter drugs into one file. Run the same language, length, and dedup checks on that text.

PubChem files do not count toward the minimum of five PDFs and do not count toward the 300-page gate.

### A7 — Final gate, notebook, and push

Status: done

Run with `python src/corpus_pipeline.py gate`, or the full rebuild `python src/corpus_pipeline.py part-a`. The notebook `part_a_corpus.ipynb` calls `run_part_a()` and stores the output. Gate result: 7 surviving PDFs and 1,122 kept PDF pages. PubChem's 104 section-pages stay out of that total. WHO attribution in `sources.json` includes CC BY-NC-SA 3.0 IGO.

The download check lives in `verify_or_download_pdfs()`. It confirms each raw PDF against `sources.json` and downloads a file only when the copy on disk does not match.

Deduplication removed the most pages among the three filters: 15 pages, 13 of them near-duplicates in NICE NG253. Extraction found 27 pages with fewer than 30 letters. Those are non-content, not language removals. The language filter removed 1 non-English page.

Gate: at least five surviving PDF documents, and at least 300 kept pages from those PDFs only. A failed gate stops the push.

The notebook displays the extraction row and the before/after table for language, length, and dedup. The prose names `langdetect`, states SHA-256 plus Jaccard of at least 0.90 on 5-word shingles, justifies the length rule, and says which step removed the most pages and why.

Push to `main` only after every check passes: pipeline, reports, `domain_corpus/`, `data/sources.json`, and `data/raw_pdfs/`. Model weights stay out of the repo. WHO text is pushed with the CC BY-NC-SA 3.0 IGO attribution in `sources.json`.

### Part A Step 2 — Baseline model output

Status: done

Assignment Part A Step 2 (1 mark). Ran on NVIDIA RTX A6000. Notebook `part_a_baseline.ipynb` stores the architecture printout and the three greedy outputs (`max_new_tokens=150`, no system prompt). Reports: `reports/baseline_architecture.json`, `reports/baseline_outputs.csv`. Measured: 7,241,732,096 parameters, 32 layers, hidden size 4096, vocab 32,000.

Hidden size is `model.config.hidden_size`: the width of each token representation in the residual stream.

These three prompts are fixed for Part B3 and Part C:

| Id | Corpus target | Instruction |
|---|---|---|
| `malaria_uncomplicated_pf` | `who_malaria_2024-11-30.txt` | What treatment does the malaria guidance recommend for uncomplicated Plasmodium falciparum malaria in adults and in children weighing at least 25 kg? |
| `sepsis_iv_antibiotics` | `nice_ng253_sepsis.txt` | In an adult with suspected sepsis, when should intravenous antibiotics be started, and what should happen if the source of infection is still uncertain? |
| `hypertension_first_line` | `nice_ng136_hypertension.txt` | What first-line antihypertensive treatment does the adult hypertension guidance recommend, and how does age change that choice? |

---

## Part B — Instruction Fine-Tuning with QLoRA [5 Marks]

Assignment page 3. Fine-tune the same base model with QLoRA on an instruction dataset built only from `domain_corpus/`. Same tokenizer and model family throughout. Every step follows the project-wide rule: logic in `src/`, notebook calls it and keeps cell outputs, reports are reproducible.

### What Part B is about

Part A gave us cleaned clinical text and a weak baseline (untuned Mistral answers that drift and invent structure). Part B teaches the model to answer clinical-protocol questions in our locked response shape, using a small supervised set:

1. **B1** turns the corpus into 50 grounded instruction–response pairs and an 80/20 split.
2. **B2** loads the base model in 4-bit NF4 and trains **one** LoRA adapter (Adapter B) with PEFT + TRL SFT.
3. **B3** runs the same 3 Part A prompts through the adapter and writes a side-by-side comparison with the bfloat16 baseline.

Marks: B1 = 2, B2 = 2, B3 = 1.

### B1 — Instruction dataset creation [2 Marks]

Status: implemented, awaiting your review. Files: `instruction_dataset.jsonl`, `data/instruction/`, `prompts/instruction_draft_prompt.txt`, `reports/instruction_checks.json`, `reports/instruction_spotcheck.csv`, `part_b_dataset.ipynb` and `.html`, `tests/test_instruction_dataset.py`.

**Assignment requirements** (checked against `Assignment-1B.pdf` page 3)

- Source: only cleaned `.txt` files from Part A Step 1 (`domain_corpus/`). No external clinical facts.
- Format: JSONL; each row has `instruction` and `response`.
- Method: synthetic generation via an external LLM. The brief requires **the exact prompt template** and **5 sample instruction–response pairs** shown in the notebook.
- Split: 80% train / 20% eval, fixed seed; report example counts for both.
- Justify dataset size and average tokens per example relative to model/GPU.
- Variant 4: the exact disclaimer must be in **every** pair.

**Our locked extras (binding)**

| Rule | Value |
|---|---|
| Pair count | Exactly 50 |
| Type mix | definition 8, explanation 7, procedure 8, comparison 7, troubleshooting 6, scenario 7, advantages/limitations 7 |
| Response shape | 1–3 sentences of clinical content (the disclaimer is **not** counted as a sentence), ≥30 clinical words, naming the source guideline ("Per WHO malaria guidelines, …") like the Variant 4 samples, then the full Variant 4 disclaimer sentence |
| Grounding | Paraphrase from a named corpus file; no verbatim long copy |
| Template diversity | No single instruction template >20% of the 50 |
| Split | seed 42 → 40 train / 10 eval |
| Spot-check | 10–15 pairs reviewed in the notebook |

**Proposed implementation**

- Code: `src/instruction_dataset.py` with subcommands `chunks`, `build`, `validate`, `split`, `report`. Notebook: `part_b_dataset.ipynb` (imports the module, displays results).
- Drafter: **Claude** (exact model id recorded in `data/instruction/drafting_log.json`), separate from Mistral. Pipeline so the step stays reproducible:
  1. `chunks` selects source chunks programmatically (seed 42) from `domain_corpus/`, covering every family, and writes `data/instruction/chunks.jsonl` (chunk id, source file, text, target pair type).
  2. The exact prompt template lives in `prompts/instruction_draft_prompt.txt` (committed). One call per chunk, with the type and chunk filled in. It contains the assignment's tip adapted to our format ("based ONLY on this text", `instruction` / `response` keys), the locked sentence `Each response must be a concise 1–3 sentence answer in your own words. Do not copy text verbatim.`, the source-naming rule, the requirement to end with the exact disclaimer, and a ban on facts outside the chunk.
  3. Raw Claude outputs are committed unedited as `data/instruction/drafts_raw.jsonl` (with model id, date, temperature). Anyone can re-run `build` from them without calling the LLM.
  4. `build` programmatically selects and trims drafts to the locked 50-pair mix, rejecting any draft that fails a check below. Rejected drafts go to `reports/instruction_rejected.csv` with the reason.
- Files: `instruction_dataset.jsonl` at repo root (submission name, only `instruction` and `response`), `data/instruction/pairs_meta.jsonl` (`id`, `type`, `source_file`, `chunk_id`, `template_family`), `data/instruction/train.jsonl`, `data/instruction/eval.jsonl`.
- Coverage: every family (WHO malaria, CDC STI, NICE sepsis, NICE hypertension, ICMR antimicrobial, ICMR STW vols 1 and 3, PubChem). No single PDF above 30% of pairs; PubChem counts as one family for that cap.
- Do **not** copy the Variant 4 sample rows from the enterprise PDF into the dataset; they are style examples only.

**Automated checks (script fails on violation)**

1. Exactly 50 rows; keys `instruction` and `response` present and non-empty; no other keys in `instruction_dataset.jsonl`.
2. Type counts match the locked mix (from metadata).
3. Every response ends with the exact disclaimer string.
4. Clinical text before the disclaimer: ≥30 words and 1–3 sentences. The splitter is one defined function in code that protects `e.g.`, `i.e.`, `P. falciparum`, `mg/kg`, decimals and `vs.`.
5. Template family: lowercase the instruction, replace drug, disease and number slots with a placeholder, and take the first 4 tokens. No family above 20% (max 10 of 50), and at least 2 families per type.
6. Every `source_file` exists under `domain_corpus/`.
7. **Grounding:** (a) no 8-word sequence in a response appears verbatim in its source chunk; (b) every number in a response (doses, days, thresholds) appears in its source chunk; (c) the response names its source guideline.
8. **No leakage:** no instruction in train or eval equals or near-duplicates (Jaccard ≥0.7 on word sets) any of the 3 fixed Part A prompts; no duplicate instructions inside the 50.
9. Split is 40/10 with seed 42; disjoint ids; both splits written; type coverage reported for each.
10. Token-length report: mean, p95 and max tokens of `format_prompt(instruction)+response+EOS` with the Mistral tokenizer; justify `max_seq_length` for B2 from it.

**Notebook must show** (each item is in the brief or in our own rules)

- Justification for 50 pairs and 40 train rows on Mistral-7B + Adapter B, including the **measured average tokens per example** and what that means for VRAM and steps (small, high-quality set; the aim is format and grounding, not new knowledge).
- **The exact generation prompt template**, printed from `prompts/instruction_draft_prompt.txt`, plus the drafter model id.
- **5 sample instruction–response pairs** (one printed per distinct type, from different source families).
- Validation table (counts by type, source family share, template share, length stats, check pass/fail).
- Spot-check table for 10–15 pairs (id, correct / grounded / disclaimer / not verbatim, pass or fail, short reason). Spot-check results are saved to `reports/instruction_spotcheck.csv`.
- Train and eval counts (40 and 10) with the seed.
- A sentence stating that the Mistral base has no chat template, so the SFT format is `### Instruction / ### Response`, as the brief allows ("where applicable").

**Done when** `instruction_dataset.jsonl` exists, all checks pass, train/eval splits exist, the notebook has saved outputs (and an HTML export), and the files are ready to commit.

### B2 — QLoRA fine-tuning with Adapter B [2 Marks]

Status: done, awaiting your review. Ran on the RTX A6000 pod: 30 optimizer steps in 45.5 s, peak 5.25 GB, eval loss 2.127 (untuned 4-bit base) to 1.427; a second same-seed run agrees within 0.0015. Files: `src/qlora_train.py`, `part_b_qlora.ipynb` and `.html`, `reports/qlora_training.json`, `reports/qlora_loss.csv`, `reports/qlora_smoke.json`, `reports/qlora_design_check.json`. The adapter is in `adapters/adapter_b/` on the pod (gitignored). The adapter learned the format (stops on EOS, adds the disclaimer) but not the facts; see the notebook inferences.

**Assignment requirements**

- QLoRA via `transformers`, `peft`, `bitsandbytes` (4-bit base).
- Train **one** of Adapter A/B/C — we train **Adapter B**.
- SFT setup matching the model’s instruction format (for Mistral-7B-v0.1: our `### Instruction` / `### Response` block; no chat template).
- Report batch size, learning rate, epochs/steps, max sequence length.

**Proposed training design (confirm on accept)**

| Setting | Proposed value | Why |
|---|---|---|
| Base load | 4-bit NF4, `bnb_4bit_compute_dtype=bfloat16`, double quant on | Assignment Part C uses the same NF4 pattern; QLoRA standard |
| LoRA | r=16, alpha=32, dropout 0.05, targets `q_proj`,`v_proj` | Locked Adapter B |
| Trainer | Plain `transformers.Trainer` with our own response-only collator (no TRL: the brief names only transformers, peft, bitsandbytes, and TRL pins are tied to transformers versions) | Assignment SFT setup, fewer version risks |
| Dataset | 40 train / 10 eval from B1 | Locked split |
| Max seq length | **256** (B1 measured max 239 tokens, p95 219) | Covers every example; 512 would only add padding |
| Epochs | **3** | Small set; more epochs overfit |
| Learning rate | **2e-4** | Common QLoRA LoRA LR |
| Batch | per-device **1**, grad accum **4** (effective 4); tune if VRAM allows 2 | Safe on 46 GB with 7B 4-bit |
| LR schedule / warmup | linear, warmup ratio 0.03 | Stable short run |
| Seed | 42 | Match data split |
| Logging | loss each step; eval loss each epoch | Notebook table |
| Output | `adapters/adapter_b/` (gitignored weights) + `reports/qlora_training.json` hyperparameters and final metrics | Weights stay out of GitHub |
| EOS | Every training text ends with `</s>` (the tokenizer's EOS) after the disclaimer | Without it the adapter never learns to stop and runs on to `max_new_tokens` |
| Padding | `pad_token` set to `unk` (not EOS), `padding_side="right"` for training | Pad = EOS makes the collator mask the EOS label |
| Loss masking | `DataCollatorForCompletionOnlyLM` with `response_template="### Response:"`; `packing=False` | Loss on the answer only; 40 rows cannot afford wasted signal |
| Eval loss | Also compute eval loss of the **untuned** base on the same 10 rows in the same format | Gives a quantitative before/after for B3 |
| Dependencies | Pinned in `requirements.txt` to the pod's versions: torch 2.8.0, transformers 4.46.3, accelerate 1.1.1, peft 0.13.2, bitsandbytes 0.50.2. The module sets `USE_TF=0` because the pod's TensorFlow/Keras 3 breaks the transformers import | Reproducible installs |

Code: `src/qlora_train.py` with `check` and `run`. Notebook: `part_b_qlora.ipynb` (calls train, prints hyperparams and train/eval loss).

The notebook must state the four hyperparameters the brief asks for (batch size, learning rate, epochs and steps, max sequence length), say why Mistral-7B-v0.1 uses a hand-built instruction format instead of a chat template, and justify Adapter B (r=16, alpha=32, q_proj and v_proj).

**Done when** training finishes, adapter files exist on the GPU machine, hyperparameter report is in the notebook and `reports/qlora_training.json`, and a smoke generate on one train-style prompt succeeds and stops on its own with the disclaimer.

### B3 — Evaluation and comparative analysis [1 Mark]

Status: done, awaiting your review. Ran on the RTX A6000 pod. Files: `src/adapter_eval.py`, `part_b_eval.ipynb` and `.html`, `reports/baseline_vs_adapter.csv`, `reports/adapter_eval_scores.csv`, `reports/adapter_eval.json`. Result: the adapter fixes the form (disclaimer 3 of 3, stops by itself 3 of 3, no fake reference sections) but not the content; all three answers contain a claim the corpus contradicts or does not support. Primary comparison is bf16 base + adapter against the regenerated baseline, which reproduced the Part A outputs exactly; a 4-bit + adapter run is reported next to it.

**Assignment requirements**

- Run the trained adapter on the **same 3 domain prompts** from Part A Step 2.
- Side-by-side comparison: baseline vs adapter.
- Describe what improved or degraded (correctness, domain terminology, completeness, hallucinations).

**Proposed implementation**

- Code: `src/adapter_eval.py` loads the **bf16 base + Adapter B** (same dtype as the Part A baseline, so the only change is fine-tuning). Same generate settings as the baseline: greedy, `max_new_tokens=150`, no system prompt, same instruction block, same 3 prompts from Part A Step 2.
- A second, optional row set uses the 4-bit base + adapter (the training dtype), labelled separately.
- Save `reports/baseline_vs_adapter.csv` with columns: `prompt_id`, `baseline_output`, `adapter_output`, `baseline_tps`, `adapter_tps`, plus `adapter_has_disclaimer`, `adapter_stopped_on_eos`, `adapter_new_tokens`.
- Also report eval loss of base vs adapter on the 10 held-out pairs (from B2).
- Notebook `part_b_eval.ipynb` displays the side-by-side table and a written analysis: one short paragraph per prompt and an overall judgment, each covering **correctness, domain terminology, completeness and hallucinations** (the four items the brief names), with quoted examples from the outputs.
- Analysis must note: baseline drift and hallucinated “Evidence/References” sections vs whether the adapter stays closer to protocol language and includes the disclaimer. Also state honestly what **degraded** (for example, wrong or invented doses, over-confident short answers, or content the 50 pairs never covered).

**Done when** the three-prompt comparison is saved, the written analysis is in the notebook output, and HTML export for Part B work is produced (or included in the main submission notebook export).

### Part B acceptance sequence

Reply with one id. Work starts only for that item.

| Id | Work | Marks |
|---|---|---|
| B1 | Build and validate the 50-pair JSONL, spot-check, 40/10 split, notebook with outputs | 2 |
| B2 | QLoRA train Adapter B; report hyperparameters and losses | 2 |
| B3 | Same 3 prompts: baseline vs adapter + written comparison | 1 |

Decisions made on review:

1. **Pair drafting:** LLM-assisted by Claude, with the reproducibility steps in B1.
2. **Submission JSONL:** only `{instruction, response}`; metadata in `data/instruction/pairs_meta.jsonl`.
3. **Training hyperparameters:** 3 epochs, lr 2e-4, batch 1 × accum 4; max sequence length 256 (measured in B1: max 239 tokens, p95 219).
4. **B3 generate dtype:** bf16 base + adapter against the bf16 baseline.

## Final submission (single notebook, required by the brief)

`Assignment-1B.pdf` (Submission Deliverables) asks for **one** notebook covering Parts A–C and **one** HTML export of it. Our per-part notebooks (`part_a_corpus`, `part_a_baseline`, `part_b_*`, later `part_c_*`) are the working notebooks. Before submission, build `assignment_1b.ipynb` that imports `src/`, shows every part's outputs and inferences in order, and states the final prompts, hyperparameters and benchmark settings. Export `assignment_1b.html` with outputs visible. The submission set is: `assignment_1b.ipynb`, `assignment_1b.html`, `instruction_dataset.jsonl`, `domain_corpus/*.txt`. The GPU-run cells (baseline, QLoRA, Part C) must be executed on the pod so their outputs are saved in the notebook.

---

## Part C — closed until Part B is done

Assignment pages 4–6. Benchmarks are on the **base** model (not the adapter), except the unmarked optional extension.

| Id | Work |
|---|---|
| C1 | Greedy, beam, top-k, top-p, temperature 0.3 / 0.7 / 1.2. Same 3 prompts. `max_new_tokens=150`. Tokens/sec + 100-word deployment choice. |
| C2 | Speculative decoding with SmolLM2-1.7B-Instruct and its own tokenizer. |
| C3 | 4-bit NF4 on 10 prompts with the winning decoder. VRAM, throughput, cost per 1M tokens (use the hourly rate for the GPU actually used). HTML export. |

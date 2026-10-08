# Assignment 1B plan

Reply with one id, for example `accept A1`. Work starts only for that item. The next item stays pending until you accept it.

Current item: **Part A Step 2 is done.** The bfloat16 baseline ran on an NVIDIA RTX A6000. Saved outputs are in `part_a_baseline.ipynb`. B1 stays closed.

## Locked already

- Domain: Medical and Clinical Literature. Use case: Clinical Protocol Lookup Assistant.
- Model for later parts: `mistralai/Mistral-7B-v0.1` on the BITS A100, with that model's own tokenizer.
- Adapter for later training: Adapter B (`r=16`, `alpha=32`, `q_proj`, `v_proj`).
- Draft model for later speculative decoding: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, with its own tokenizer.
- Instruction block for later training and evaluation:

```text
### Instruction:
{instruction}

### Response:
{response}
```

- After Part A, the JSONL target is 50 pairs: definition 8, explanation 7, procedure 8, comparison 7, troubleshooting 6, scenario 7, advantages and limitations 7.
- Each response is 1–3 sentences, at least 30 words of clinical content, paraphrased from the corpus, then the full disclaimer: "This output is for educational/reference purposes only and must not replace professional clinical judgment."
- Kept PDF pages after the three filters must total at least 300.
- Public repo: https://github.com/bheemreddy1729/fine-tuning-Mistral-7B

Parts B and C are listed at the bottom so the sequence stays visible. They are not open for acceptance until Part A is pushed.

## Project-wide implementation rule

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

Run with `python src/corpus_pipeline.py extract`. Result: 7 PDFs, 1,165 pages, 609,037 words, 4,248,733 characters, 27 pages with fewer than 30 letters. Staged text is in `data/extracted/`. The short-page list is `reports/extraction_short_pages.csv`.

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

Status: code ready, GPU run pending

This is the assignment's Part A Step 2 (1 mark). The plan table below calls the same work B2.

`python src/baseline.py check` confirms three clinical prompts, no system prompt, the shared instruction block, greedy decoding, and `max_new_tokens=150`. It does not download the model.

`python src/baseline.py run` on the A100 loads `mistralai/Mistral-7B-v0.1` in bfloat16 with `AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")`. It prints parameter count, decoder layers, hidden size, and vocabulary size from the loaded model, then saves `reports/baseline_architecture.json` and `reports/baseline_outputs.csv`. The notebook is `part_a_baseline.ipynb`.

Hidden size is `model.config.hidden_size`: the width of each token representation in the residual stream. The code reads it from the loaded model.

## Later, after the baseline is saved

These stay closed until you accept them.

| Id | Work |
|---|---|
| B1 | Build the 50-pair JSONL from `domain_corpus/` only, with the locked mix, prompt, length filter, template check, and your 15-pair spot check. Split 80/20 with seed 42. |
| B2 | Part A baseline: load Mistral-7B-v0.1 in bfloat16, print parameter count, layers, hidden size, and vocabulary size, and save 3 clinical prompts. |
| B3 | QLoRA with Adapter B on the training split. Report batch size, learning rate, epochs or steps, and max sequence length. |
| B4 | Same 3 prompts, baseline versus adapter, with a written comparison. |
| C1 | Greedy, beam search, top-k, top-p, and temperature at 0.3, 0.7, and 1.2. Same 3 prompts. `max_new_tokens=150`. Record text and tokens per second. Write the 100-word deployment choice. |
| C2 | Speculative decoding with SmolLM2-1.7B-Instruct and its own tokenizer. |
| C3 | 4-bit NF4 benchmark on 10 prompts using the winning decoder. Fill VRAM, throughput, and cost per 1 million tokens at ₹12/hour for bfloat16, 4-bit, and 4-bit plus speculative decoding. Export the notebook to HTML. |

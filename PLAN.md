# PLAN.md: status and next steps (read this first)

Assignment 1B (Domain LLM Adaptation and Production Optimization), Medical and Clinical Literature, Variant 4 (Clinical Protocol Lookup Assistant),
`mistralai/Mistral-7B-v0.1`. Branch `lab-run`. Last updated 2026-10-09. Marks: Part A 2, Part B 5, Part C 8 (total 15).
Operational setup of the BITS Kubeflow lab is in `lab_runbook/LAB_RUNBOOK.md`; this file tracks **what is delivered and what is left**.

## 1. If you are an agent resuming this work

1. Read this file, then `lab_runbook/LAB_RUNBOOK.md` (namespace quota, server settings, DON'T list, driving the lab through the browser).
2. The deliverable is **one self-contained notebook**, `assignment_1b.ipynb`, plus `assignment_1b.html`, `instruction_dataset.jsonl` and `domain_corpus/*.txt`. All code lives in notebook cells.
   The notebook is **generated** from `tools/build_assignment_notebook.py` (Part A) and `tools/part_b1_cells.py`, `part_b2_cells.py`, `part_b3_cells.py`. Edit those, rebuild, execute on the lab; never edit the `.ipynb` by hand.
3. Rules from the user, which override defaults:
   - **Instructor's pinned environment wins** (torch 2.5.1+cu124, transformers 4.46.3, peft 0.13.2, accelerate 1.1.1, bitsandbytes 0.44.1, trl 0.12.1). Never upgrade or downgrade these.
   - **One step at a time**, user approval between steps. Aim for full marks and the performance-best configuration within the namespace quota.
   - **Never mention any earlier or other machine or software stack** (anything from before the lab run) in the notebook, HTML, reports or docs. The assignment is accounted for on the lab A100 only; if you find such text, remove it.
   - **Do not push `main` or open the PR without asking.** `lab-run` may be pushed. Never put secrets or the user's token in files.
   - Take screenshots of the notebook, memory and storage at each milestone into `lab_evidence/` (git tracked).
   - Analysis must be honest and checked against the corpus text; verify every claim before writing it (the user wants "detailed inferences and justification").
4. Execute long jobs on the pod in the background (`nohup ... > log &`) and poll the log; the browser tool can drop. See runbook section 6.

## 2. Delivered and verified on the lab (A100-SXM4-80GB)

`assignment_1b.ipynb`: 72 cells, executed end to end, 0 errors; HTML exported; reports in `reports/`; proof in `lab_evidence/`.

| Item | Result (so you can sanity-check a re-run) | Where |
|---|---|---|
| Part A Step 1, corpus | 7 PDFs (WHO malaria, CDC STI, NICE sepsis NG253, NICE hypertension NG136, ICMR antimicrobial, ICMR STW vol 1 and 3), 1,165 pages extracted, 1,122 kept (gate: 5 PDFs and 300 pages, passed). Language filter removed 1 page (a names-only list falsely detected as Indonesian), length filter 0, dedup 15 (2 exact, 13 near; 13 from NICE sepsis). 9 PubChem monograph files are supplementary. Notebook output equals `reports/corpus_stats.csv` and the committed `domain_corpus/` byte for byte. | `domain_corpus/`, `data/`, notebook |
| Part A Step 2, baseline | bfloat16, greedy, 150 tokens, no system prompt. 7,241,732,096 parameters, 32 layers, hidden size 4096, vocab 32,000. The 3 fixed prompts (malaria, sepsis, hypertension) all hit the 150-token cap and invent `### Evidence/References` sections. | `reports/baseline_*` |
| B1 dataset | 50 pairs (type mix 8/7/8/7/6/7/7, 8 families, none above 20 %, max template share 6 %), disclaimer 50/50, 1-2 sentences, 46-91 clinical words, split 40/10 seed 42. Rebuilt from committed raw drafts, identical to `instruction_dataset.jsonl`. 12-pair spot-check (done by the assistant, 12/12 pass). Tokens mean 173.7, max 239, so max length 256. | `instruction_dataset.jsonl`, `data/instruction/`, `prompts/` |
| B2 QLoRA | Adapter B (r 16, alpha 32, q_proj and v_proj), 4-bit NF4, lr 2e-4, 3 epochs, batch 1 x 4, max length 256, loss on answer + EOS. Eval loss 2.126 to 1.426 (-32.9 %), 30 steps, 53 s, peak 5.32 GB. A same-seed repeat was bit-identical. Learned the form, not the facts. | `reports/qlora_*` |
| B3 evaluation | Baseline vs adapter (bf16) vs adapter (4-bit) on the 3 prompts: disclaimer 0/3 to 3/3, stops by itself 0/3 to 3/3, invented sections 3 to 0, rubric facts 5/8/7 of 16, unsupported claims 3/4/5. Adapter fixes the form but gives confident wrong content (e.g. thiazide first line for age 55+). Memory 4.4 GB (4-bit) vs 14.6 GB (bf16). | `reports/baseline_vs_adapter.csv`, `adapter_eval_scores.csv` |
| Lab proof and docs | 7 screenshots and the instructor's `setup_env`/`check_env` logs in `lab_evidence/`; `README.md`; runbook; repo cleaned of obsolete files. | repo root |

**The trained adapter is committed** in `adapters/adapter_b/` (7 files, 27 MB of float32 weights, 128 tensors, 6,815,744 parameters, sha256 of `adapter_model.safetensors` = `0ab4a10a9df7c4502ad3193fd09671a314f5b8979dbe2f84ea33d8d77b9d18c2`), so B3 and the optional adapter extension can load it on any pod without retraining. It is the adapter produced by the notebook's B2 cells that B3 evaluated; rerunning B2 recreates it identically (same seed). The Mistral base weights are not committed (downloaded from Hugging Face).

## 3. To do: Part C (8 marks), all on the **base** model, never the adapter

Add `tools/part_c_cells.py` (same pattern as the B modules, raw-string cells, hooked into the builder), keep every cell executed with outputs, and write detailed inferences for each step.
Use the same 3 fixed prompts, `max_new_tokens = 150`, the same `### Instruction/### Response` format, and the base model loaded in bfloat16 as in Part A.

1. **C1 decoding strategies (3 marks).** Greedy, beam search (4 beams), top-k (50), top-p (0.9), temperature 0.3, 0.7 and 1.2 (the brief's table has three temperature columns). 3 prompts x 7 settings. Record the generated text and tokens/s (warm-up first; repeat timings
   because the A100 is shared and speeds vary ~25 % run to run, so report ratios). Fill the brief's comparison table. Add a corpus-based quality rubric as in B3 (facts and unsupported claims). Write the **100-word** deployment recommendation citing specific cells of the table (factual domain: expect greedy or low temperature).
2. **C2 speculative decoding (2 marks).** Draft model `HuggingFaceTB/SmolLM2-1.7B-Instruct`; its vocabulary (~49k) differs from Mistral's (32k), so use transformers 4.46.3 universal assisted generation (`assistant_tokenizer`). Both models must fit in VRAM. Benchmark on the C1 prompts, report speed-up
   and any quality difference (greedy speculative output should match plain greedy). Biggest technical risk; fallback if the cross-tokenizer path fails: a draft with Mistral's vocabulary, with the reason documented.
3. **C3 4-bit quantisation and cost (3 marks).** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`. **10 domain prompts** = the 3 fixed ones plus 7 new ones written from the corpus (none overlapping the 50 training instructions) with the best decoder from C1.
   Table of bfloat16 (Part A baseline), 4-bit NF4 and 4-bit + speculative: peak VRAM, throughput, **cost per 1M tokens = (1,000,000 / tok_s) / 3600 x 12.00** (the lab GPU is an A100, rate Rs 12/hr). Add a 2-3 sentence recommendation citing the measured cost and quality trade-off.
   **The 7 new prompts must be shown to the user for approval before running.**
4. **Optional, no marks.** Adapter + 4-bit + speculative decoding, as the brief's extension; a short run is worthwhile because B3 showed the adapter hurts content.

## 4. To do: finish

1. Rebuild and execute the full notebook on the lab (runbook section 6), export HTML, pull the results, validate (0 errors, no earlier-machine text, key checks True).
2. Final screenshots into `lab_evidence/`; update `README.md` and this file.
3. Clean-run test on the pod with the removed files absent (the notebook must run top to bottom with only the files in this repo plus the instructor's environment).
4. Ask the user before opening the PR `lab-run` to `main`. After the PR: `main` holds the final submission; old history remains in git.

## 5. Open items and known soft spots

- 7 extra Part C prompts: draft from the corpus, **show the user for approval**.
- The B1 spot-check was done by the assistant, not a human; the user may want to read a few pairs (the notebook shows three beside their source text).
- B3 rests on 3 prompts and a pattern-based rubric; the analysis says so and relies on reading the outputs.
- The instructor's rule is 300 PDF pages in total, not per document (the per-document reading in an old note was wrong). The corpus gate is correct as it stands.
- The lab registry image `bits-sudo-jupyter-pytorch-cuda-full` has no tag; the server uses `kubeflownotebookswg/jupyter-pytorch-cuda-full:v1.10.0-rc.1` (runbook section 1).

## 6. File map

| Path | Purpose |
|---|---|
| `assignment_1b.ipynb`, `assignment_1b.html` | The executed notebook and its HTML export (deliverables) |
| `instruction_dataset.jsonl`, `domain_corpus/*.txt` | Deliverables: fine-tuning data and cleaned corpus |
| `tools/` | Generators of the notebook (`build_assignment_notebook.py`, `part_b1_cells.py`, `part_b2_cells.py`, `part_b3_cells.py`) |
| `data/` | `sources.json`, `raw_pdfs/`, `instruction/` (chunks, raw drafts, metadata, train/eval), `pubchem/` |
| `adapters/adapter_b/` | The trained Adapter B (committed, 27 MB) |
| `prompts/` | Exact LLM prompt template for drafting the pairs |
| `reports/` | Notebook outputs (baseline, QLoRA, B3 scores) and two reference inputs (`corpus_stats.csv`, `instruction_spotcheck.csv`) |
| `lab_evidence/` | Screenshots and logs proving the lab run |
| `lab_runbook/LAB_RUNBOOK.md` | How to set up and drive a new lab session, with all lessons learned |

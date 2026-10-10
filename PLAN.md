# PLAN.md: status and next steps (read this first)

Assignment 1B (Domain LLM Adaptation and Production Optimization), Medical and Clinical Literature, Variant 4 (Clinical Protocol Lookup Assistant),
`mistralai/Mistral-7B-v0.1`. The final work is on `main` (the working branch `lab-run` was merged on 2026-10-10). Last updated 2026-10-10. Marks: Part A 2, Part B 5, Part C 8 (total 15).
Operational setup of the BITS Kubeflow lab is in `lab_runbook/LAB_RUNBOOK.md`; this file tracks **what is delivered and what is left**.

## 1. If you are an agent resuming this work

1. Read this file, then `lab_runbook/LAB_RUNBOOK.md` (namespace quota, server settings, DON'T list, driving the lab through the browser).
2. The deliverable is **one self-contained notebook**, `assignment_1b.ipynb`, plus `assignment_1b.html`, `instruction_dataset.jsonl` and `domain_corpus/*.txt`. All code lives in notebook cells.
   The notebook is **generated** from `tools/build_assignment_notebook.py` (Part A) and `tools/part_b1_cells.py`, `part_b2_cells.py`, `part_b3_cells.py`. Edit those, rebuild, execute on the lab; never edit the `.ipynb` by hand.
3. Rules from the user, which override defaults:
   - **Instructor's pinned environment wins** (torch 2.5.1+cu124, transformers 4.46.3, peft 0.13.2, accelerate 1.1.1, bitsandbytes 0.44.1, trl 0.12.1). Never upgrade or downgrade these.
   - **One step at a time**, user approval between steps. Aim for full marks and the performance-best configuration within the namespace quota.
   - **Never mention any earlier or other machine or software stack** (anything from before the lab run) in the notebook, HTML, reports or docs. The assignment is accounted for on the lab A100 only; if you find such text, remove it.
   - **Ask before pushing to `main`** (it holds the final submission): work on a branch and open a pull request. Never put secrets or the user's token in files.
   - Take screenshots of the notebook, memory and storage at each milestone into `lab_evidence/` (git tracked).
   - Analysis must be honest and checked against the corpus text; verify every claim before writing it (the user wants "detailed inferences and justification").
4. Execute long jobs on the pod in the background (`nohup ... > log &`) and poll the log; the browser tool can drop. See runbook section 6.

## 2. Delivered and verified on the lab (A100-SXM4-80GB)

`assignment_1b.ipynb`: 112 cells (Parts A, B and C), executed end to end on the lab in about one hour, 0 errors; HTML exported; reports in `reports/`; proof in `lab_evidence/`.

| Item | Result (so you can sanity-check a re-run) | Where |
|---|---|---|
| Part A Step 1, corpus | 7 PDFs (WHO malaria, CDC STI, NICE sepsis NG253, NICE hypertension NG136, ICMR antimicrobial, ICMR STW vol 1 and 3), 1,165 pages extracted, 1,122 kept (gate: 5 PDFs and 300 pages, passed). Language filter removed 1 page (a names-only list falsely detected as Indonesian), length filter 0, dedup 15 (2 exact, 13 near; 13 from NICE sepsis). 9 PubChem monograph files are supplementary. Notebook output equals `reports/corpus_stats.csv` and the committed `domain_corpus/` byte for byte. | `domain_corpus/`, `data/`, notebook |
| Part A Step 2, baseline | bfloat16, greedy, 150 tokens, no system prompt. 7,241,732,096 parameters, 32 layers, hidden size 4096, vocab 32,000. The 3 fixed prompts (malaria, sepsis, hypertension) all hit the 150-token cap and invent `### Evidence/References` sections. | `reports/baseline_*` |
| B1 dataset | 50 pairs (type mix 8/7/8/7/6/7/7, 8 families, none above 20 %, max template share 6 %), disclaimer 50/50, 1-2 sentences, 46-91 clinical words, split 40/10 seed 42. Rebuilt from committed raw drafts, identical to `instruction_dataset.jsonl`. 12-pair spot-check (done by the assistant, 12/12 pass). Tokens mean 173.7, max 239, so max length 256. | `instruction_dataset.jsonl`, `data/instruction/`, `prompts/` |
| B2 QLoRA | Adapter B (r 16, alpha 32, q_proj and v_proj), 4-bit NF4, lr 2e-4, 3 epochs, batch 1 x 4, max length 256, loss on answer + EOS. Eval loss 2.126 to 1.426 (-32.9 %), 30 steps, 53 s, peak 5.32 GB. A same-seed repeat was bit-identical. Learned the form, not the facts. | `reports/qlora_*` |
| B3 evaluation | Baseline vs adapter (bf16) vs adapter (4-bit) on the 3 prompts: disclaimer 0/3 to 3/3, stops by itself 0/3 to 3/3, invented sections 3 to 0, rubric facts 5/8/7 of 16, unsupported claims 3/4/5. Adapter fixes the form but gives confident wrong content (e.g. thiazide first line for age 55+). Memory 4.4 GB (4-bit) vs 14.6 GB (bf16). | `reports/baseline_vs_adapter.csv`, `adapter_eval_scores.csv` |
| C1 decoding (3 marks) | 7 settings x 3 prompts, 150 tokens. Speed is the same for greedy and sampling (within 3 %, about 31 tok/s); beam search about 10 % slower. Greedy gives the most rubric facts (5 of 16, tied with top-p) and is deterministic; temperature 1.2 collapses into gibberish (0 facts); diversity (distinct-2) rises 0.52 to 0.99 from temperature 0.3 to 1.2. 104-word recommendation: deploy greedy. | notebook |
| C2 speculative (2 marks) | SmolLM2-1.7B-Instruct draft through transformers 4.46.3 universal assisted generation works; output identical to plain greedy (3 of 3); 2.2 to 2.9 tokens per target pass; **speed-up about 1.0x** (median 0.97, range 0.82 to 1.08) because per-token overhead dominates: a draft token costs 0.58 of a target token; the 360M draft has 32 layers like Mistral and is slower (0.77x). | `reports/c2_speculative.csv` |
| C3 4-bit and cost (3 marks) | 10 prompts (3 fixed + 7 corpus-based, user-approved), greedy: bfloat16 14.5 GB, 31.1 tok/s, Rs 107 per 1M tokens; 4-bit NF4 4.4 GB, 14.1 tok/s, Rs 236 (2.2x); 4-bit + speculative 7.9 GB, 18.3 tok/s, Rs 182. Rubric facts 24, 20, 19 of 44. Recommendation: bfloat16 greedy; 4-bit only when VRAM is the constraint. Optional extension adapter + 4-bit + speculative: 13.5 tok/s, Rs 247, 25 facts, 5 flags. | `reports/c3_*.csv` |
| Lab proof and docs | 7 screenshots and the instructor's `setup_env`/`check_env` logs in `lab_evidence/`; `README.md`; runbook; repo cleaned of obsolete files. | repo root |

**The trained adapter is committed** in `adapters/adapter_b/` (7 files, 27 MB of float32 weights, 128 tensors, 6,815,744 parameters, sha256 of `adapter_model.safetensors` = `0ab4a10a9df7c4502ad3193fd09671a314f5b8979dbe2f84ea33d8d77b9d18c2`), so B3 and the optional adapter extension can load it on any pod without retraining. It is the adapter produced by the notebook's B2 cells that B3 evaluated; rerunning B2 recreates it identically (same seed). The Mistral base weights are not committed (downloaded from Hugging Face).

## 3. Finish: status

1. **Done.** Notebook rebuilt and executed on the lab (about 55 minutes), HTML exported, reports pulled and validated (0 errors, no earlier-machine text, key checks True).
2. **Done (2026-10-10), clean-run test.** A fresh clone of `lab-run` (commit `b2e3c40`, 84 tracked files, none of the removed files) on the pod, notebook executed top to bottom: 0 errors. Deterministic results identical to the committed run (corpus gate and statistics, B1 rebuild,
   B2 eval losses 2.1264 to 1.5047 / 1.4245 / 1.4264, B3 outputs, C3 quality table). Timings within about 7 % (bfloat16 33.0 against 31.1 tokens/s, 4-bit 15.0 against 14.1, speculative speed-ups 1.01 / 0.97 / 0.90 against 1.03 / 1.00 / 0.89). The test folder `~/data/clean_test` on the pod can be deleted.
3. **Done (2026-10-10).** Final milestone screenshots in `lab_evidence/` (08 to 10): finished run, pod memory, storage and GPU, Kubeflow server and volumes.
4. **Done (2026-10-10).** Pull request #1 (`lab-run` into `main`) was merged with a merge commit (`871eadb`); `main` holds the final submission and the old history stays in git.
5. **Done (2026-10-10), deliverables check against the brief's Submission Deliverables.** `assignment_1b.ipynb` (112 cells plus a closing summary of the final prompts, hyperparameters and benchmarking settings; every code cell has saved output, 0 errors), `assignment_1b.html` (outputs visible), `instruction_dataset.jsonl` (50 rows, only `instruction` and `response`), `domain_corpus/*.txt` (16 files): all present on `main`; the 18 per-part requirements were checked in the notebook.
6. **Open decision (user).** The brief's C3 load line is `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)`; the notebook adds `bnb_4bit_use_double_quant=True` (the QLoRA recipe, stated in the C3 text). Removing it would raise the 4-bit memory figure by a few tenths of a GB and needs a rerun of the notebook (about one hour) to keep outputs consistent.

## 4. Open items and known soft spots

- Part C finding to keep in mind: speculative decoding gives no wall-clock gain on the bfloat16 target in this setup (about 1.0x) and 4-bit costs about 2.2x more per token; both are explained in the notebook, and the recommendation is bfloat16 greedy.
- The B1 spot-check was done by the assistant, not a human; the user may want to read a few pairs (the notebook shows three beside their source text).
- B3 rests on 3 prompts and a pattern-based rubric; the analysis says so and relies on reading the outputs.
- The instructor's rule is 300 PDF pages in total, not per document (the per-document reading in an old note was wrong). The corpus gate is correct as it stands.
- The lab registry image `bits-sudo-jupyter-pytorch-cuda-full` has no tag; the server uses `kubeflownotebookswg/jupyter-pytorch-cuda-full:v1.10.0-rc.1` (runbook section 1).

## 5. File map

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

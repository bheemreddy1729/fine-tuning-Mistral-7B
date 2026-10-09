# Lab re-run plan (BITS Kubeflow, A100-SXM4-80GB)

Goal: re-run Parts A, B and C on the lab pod with the instructor's pinned environment, and produce the single
deliverable set the brief asks for: `assignment_1b.ipynb` (executed, outputs visible), `assignment_1b.html`,
`instruction_dataset.jsonl`, `domain_corpus/*.txt`. Proof of lab use goes in `lab_evidence/`.

## Ground rules (from the user)

1. The instructor guide's versions win over `requirements.txt`: torch 2.5.1, transformers 4.46.3, peft 0.13.2,
   accelerate 1.1.1, bitsandbytes 0.44.1, trl 0.12.1 (our code does not use TRL).
2. Re-run everything on the lab. Never delete or overwrite existing reports or artifacts until that part is finished
   and the user approves. Mechanism: all lab work happens on git branch `lab-run` on the pod. `main` (commit `24707a7`)
   keeps the old outputs untouched, and the new outputs are compared against them before anything replaces them.
3. One step at a time, user approval between steps. Performance-best config within quota.
4. Screenshots of notebook, memory and storage at each milestone go to `lab_evidence/` (git tracked).

## Known facts that shape the plan

| Fact | Consequence |
|---|---|
| GPU is A100 80 GB (about 68 GB free, 12 GB held by another tenant) | The brief's A100 rate of Rs 12/hr is now the correct rate for Part C costs. Use `torch.cuda.max_memory_allocated` for VRAM (per process), repeat timings, and say the card is shared. |
| Pod limits: 4 CPU, 15.5 Gi RAM (limit 18.6), 15 Gi data volume, home dir is ephemeral | Venv and repo live on `~/data` (about 4-6 GB). Hugging Face cache goes to the container disk (873 GB free), re-downloaded after a restart. Load models with `device_map="cuda"`, `low_cpu_mem_usage=True`. |
| Mistral weights 14.5 GB and SmolLM2-1.7B 3.4 GB | Downloads happen once per pod life. GitHub and Hugging Face are reachable (HTTP 200). |
| Repo `requirements.txt` pins torch 2.8.0 and bitsandbytes 0.50.2, and every notebook's first cell runs `pip install -r` | Replace with a lab-aware install cell: install only the extras the guide lacks (pymupdf, langdetect, nbformat, nbclient, nbconvert, hf_transfer) and never touch pinned packages. `check_env.py` is the gate. |
| Browser sessions can drop | Run long jobs from the terminal (`jupyter nbconvert --execute` or `python -m src.x`) with `nohup` and logs, not from a browser-attached kernel. |
| transformers 4.46.3 | Supports universal assisted generation (`assistant_tokenizer`), which handles the Mistral/SmolLM2 vocabulary mismatch (C2 risk). |

## Phases

### L0. Environment (approved, in progress)
- `bash ./setup_env.sh`, then `./venv/bin/python ./check_env.py` must end with "All checks passed".
- Add the extras for Part A, create branch `lab-run`, fix the notebooks' install cell.
- Evidence: screenshot of the check summary.

### L1. Part A re-run (2 marks)
- Corpus pipeline from `data/raw_pdfs/` (committed), then the A2 to A7 gates, then the bf16 baseline on 3 prompts.
- Compare new `reports/*` and `domain_corpus/` with `main` byte-for-byte. Earlier runs were reproducible across
  machines, so expect identical corpus files and identical greedy outputs (only timings differ).
- PubChem monographs are already committed under `data/pubchem/sections`. Do not re-fetch unless the user asks.

### L2. Part B re-run (5 marks)
- B1: rebuild the dataset from the committed raw drafts (`drafts_raw.jsonl`), validate, split 40/10. Must reproduce
  `instruction_dataset.jsonl` exactly. No new LLM calls.
- B2: QLoRA Adapter B on the A100 (r=16, alpha=32, q_proj and v_proj, lr 2e-4, 3 epochs, batch 1 x accum 4,
  max length 256, seed 42). Plain `Trainer`, no TRL. Record loss, time, peak VRAM, with `bitsandbytes` 0.44.1.
- B3: bf16 base + adapter vs baseline on the same 3 prompts, plus the 4-bit row. Keep the honest finding if it repeats
  (format fixed, content not).
- Evidence: screenshot of `nvidia-smi` during training.

### L3. Part C (8 marks), new code in `src/`
- C1 (3): greedy, beam (4 beams), top-k 50, top-p 0.9, temperature 0.3, 0.7, 1.2 (the brief's table has Temp 1/2/3 columns).
  3 prompts x 7 settings, `max_new_tokens=150`, tokens/s recorded, a corpus-based quality rubric as in B3, a 100-word
  deployment analysis citing table cells.
- C2 (2): speculative decoding, SmolLM2-1.7B-Instruct as draft with universal assisted generation. Check that
  outputs match plain greedy (they should, up to numerics), report speed-up and acceptance. Fallback if the cross-tokenizer
  path misbehaves: a Mistral-vocabulary draft, with the reason documented.
- C3 (3): 4-bit NF4 on 10 prompts (the 3 fixed plus 7 new, written from the corpus, none overlapping the 50 training
  instructions), with the best decoder from C1. Table: bf16, 4-bit, 4-bit + speculative, giving peak VRAM, tok/s and
  cost per 1M tokens at Rs 12/hr, plus a 2 to 3 sentence recommendation.
- Optional extension (no marks): adapter + 4-bit + speculative, if time allows.

### L4. Unified notebook and submission
- Build `assignment_1b.ipynb`: one install cell, then Parts A, B, C in order, each calling `src/` and keeping outputs,
  with the final prompts, hyperparameters, benchmark settings and written inferences. It embeds the `lab_evidence/`
  screenshots in a short "Execution environment" section.
- Execute on the pod from the terminal (`jupyter nbconvert --to notebook --execute`), export `assignment_1b.html`.
- Check the four deliverables, update `PLAN.md`, and commit on `lab-run`. Pushing and replacing `main` only on user approval.

## Decisions needed from the user
1. OK to work on branch `lab-run` (old outputs stay on `main`)?
2. Part C temperatures 0.3 / 0.7 / 1.2 as above?
3. Write the 7 extra C3 prompts from the corpus myself, and show them for approval before running?

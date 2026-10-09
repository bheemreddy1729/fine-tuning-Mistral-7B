# Assignment 1B: Domain LLM adaptation and production optimization

Domain: Medical and Clinical Literature (Variant 4, Clinical Protocol Lookup Assistant). Model: `mistralai/Mistral-7B-v0.1`.
Executed on the BITS Prayogshala / Kubeflow lab, NVIDIA A100-SXM4-80GB, with the instructor's pinned environment
(torch 2.5.1, transformers 4.46.3, peft 0.13.2, bitsandbytes 0.44.1). Proof of the lab run is in `lab_evidence/`.

## Deliverables

| File | Contents |
|---|---|
| `assignment_1b.ipynb` | The notebook with outputs: Part A (corpus, baseline), Part B (dataset, QLoRA, evaluation), Part C (decoding, speculative decoding, 4-bit and cost). All code is in the cells. |
| `assignment_1b.html` | HTML export of the executed notebook |
| `instruction_dataset.jsonl` | The 50 instruction-response pairs used for fine-tuning |
| `domain_corpus/*.txt` | The cleaned domain text files (7 PDF documents and 9 PubChem drug monographs) |

## Other folders

| Folder | Contents |
|---|---|
| `data/` | `sources.json` (title, URL, licence, size, pages of each PDF), `raw_pdfs/`, `instruction/` (chunks, raw LLM drafts, metadata, train/eval split), `pubchem/` (raw PubChem sections) |
| `prompts/` | The exact prompt template used to draft the instruction pairs |
| `reports/` | Results written by the notebook (baseline, QLoRA losses, baseline-vs-adapter scores) and two reference files it reads |
| `lab_evidence/` | Screenshots and logs proving the run on the lab (GPU, memory, storage, environment check) |
| `tools/` | The scripts that generate `assignment_1b.ipynb` |
| `lab_runbook/` | Status and a step-by-step guide to start a new lab session |

## Reproduce

On a GPU Kubeflow server (A100, the instructor's `setup_env.sh` already run), see `lab_runbook/LAB_RUNBOOK.md`. In short:
`../venv/bin/python -m nbconvert --to notebook --execute --inplace assignment_1b.ipynb`. The adapter weights are not committed
(they are produced by the notebook's QLoRA step).

## Status

Parts A and B are complete and verified on the lab; Part C is in progress. See the runbook.

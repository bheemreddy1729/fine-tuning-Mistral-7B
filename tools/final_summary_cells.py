"""Closing section of assignment_1b.ipynb: the final prompts, key hyperparameters and benchmarking settings in one place (required by the brief)."""


def add(md):
    md(r"""
---
## Final prompts, hyperparameters and benchmarking settings (summary)

The brief asks for the final prompts, the key hyperparameters and the benchmarking settings. They are used in the cells above; this section collects them in one place.

**Environment.** BITS Prayogshala / Kubeflow notebook server, one NVIDIA A100-SXM4-80GB (shared with other users), 4 CPU, 15.5 Gi RAM; Python 3.11, torch 2.5.1+cu124, transformers 4.46.3, peft 0.13.2, bitsandbytes 0.44.1. Model `mistralai/Mistral-7B-v0.1` with its own tokenizer throughout (a base model with no chat template);
draft model `HuggingFaceTB/SmolLM2-1.7B-Instruct` with its own tokenizer (and `SmolLM2-360M-Instruct` for the ablation).

**Prompt format (no system prompt, every part).**
```
### Instruction:
{instruction}

### Response:
```

**Final prompts.** The first three are the baseline and evaluation prompts of Parts A, B3, C1 and C2; all ten are used in C3. The seven new prompts are drawn from the corpus and do not overlap the training instructions (highest word-set similarity 0.35, limit 0.70).

| # | id | prompt |
|---|---|---|
| 1 | malaria_uncomplicated_pf | What treatment does the malaria guidance recommend for uncomplicated Plasmodium falciparum malaria in adults and in children weighing at least 25 kg? |
| 2 | sepsis_iv_antibiotics | In an adult with suspected sepsis, when should intravenous antibiotics be started, and what should happen if the source of infection is still uncertain? |
| 3 | hypertension_first_line | What first-line antihypertensive treatment does the adult hypertension guidance recommend, and how does age change that choice? |
| 4 | n01_malaria_second_line | A patient with uncomplicated P. falciparum malaria has treatment failure within 28 days. What second-line treatment does WHO recommend, and which longer regimens are no longer generally recommended? |
| 5 | n02_gonorrhoea_regimen | What single-dose regimen does CDC recommend for uncomplicated gonococcal infection of the cervix, urethra or rectum in adults, and how does the dose change with body weight? |
| 6 | n03_syphilis_primary | What is the recommended treatment for primary or secondary syphilis in adults? |
| 7 | n04_bacterial_vaginosis | Which regimens does CDC recommend for bacterial vaginosis? |
| 8 | n05_cystitis_icmr | What does ICMR list as the drug of choice for acute cystitis, and when should nitrofurantoin or fosfomycin be avoided? |
| 9 | n06_diarrhoea_zinc | How much zinc does the ICMR workflow advise for a child with acute diarrhoea, and for how long? |
| 10 | n07_metronidazole | Which infections is metronidazole indicated for, and what advice does PubChem give about alcohol? |

**Part A (corpus and baseline).** 7 PDFs; a page with fewer than 30 letters is non-content; language filter `langdetect` (seed 0), per page; length filter at least 3 pages and 1,000 words per document; deduplication by SHA-256 of the normalised page, then Jaccard similarity of at least 0.90 on 5-word shingles for pages and documents; corpus gate 5 PDFs and 300 kept PDF pages (1,122 kept).
Baseline: bfloat16, greedy, `max_new_tokens = 150`, no system prompt.

**Part B (dataset and QLoRA).** Instruction dataset: 50 pairs, type mix 8 / 7 / 8 / 7 / 6 / 7 / 7, 8 source families (none above 30 %), no instruction template above 20 %, 1 to 3 sentences and at least 30 clinical words per answer, Variant 4 disclaimer on every answer, split 80/20 with seed 42 (40 train, 10 eval), maximum token length 239.
Adapter B: LoRA r = 16, alpha = 32, dropout 0.05 on `q_proj` and `v_proj` (6,815,744 trainable parameters); 4-bit NF4 base with double quantization and bfloat16 compute; 3 epochs (30 optimizer steps); per-device batch size 1 with 4 gradient-accumulation steps (effective batch 4); learning rate 2e-4, linear schedule, 3 % warm-up; optimizer `paged_adamw_32bit`, weight decay 0, gradient clipping 1.0;
maximum sequence length 256; seed 42; bfloat16; gradient checkpointing; loss on the answer and end-of-sequence token only. Evaluation (B3): greedy, 150 new tokens, bfloat16 base + unmerged adapter (and 4-bit base + adapter).

**Part C (inference).**
- *C1 decoding:* greedy; beam search (4 beams, early stopping); top-k 50; top-p 0.9; temperature 0.3, 0.7 and 1.2 (top-k and top-p switched off for pure temperature scaling); `max_new_tokens = 150`; sampling seed 42 (diversity from seeds 1, 2 and 3); speed = mean of 2 identical-seed runs after a warm-up.
- *C2 speculative decoding:* target Mistral-7B in bfloat16, draft SmolLM2-1.7B-Instruct, universal assisted generation (target and draft tokenizers passed explicitly), greedy; draft tokens per round compared at 3, 5 and 8 (constant) and the adaptive heuristic schedule (selected: heuristic, 5); speed-up = median of 15 paired ratios from 5 alternating runs of plain and speculative decoding; ablation with the 360M draft.
- *C3 quantization and cost:* `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`; best decoder from C1 (greedy); the 10 prompts above; throughput = total new tokens / total time over the 10 prompts, median of 3 alternating repeats after a warm-up; peak VRAM = resident weights + largest extra allocation during generation;
  cost per 1M tokens = (1,000,000 / throughput) / 3600 x Rs 12.00 per hour (the brief's A100 rate). Optional extension: Adapter B + 4-bit + speculative decoding, 2 passes.

**Reproducibility.** Fixed seeds (42 throughout; 1 to 3 for the C1 diversity samples), the data and drafts under `data/`, and the trained adapter in `adapters/adapter_b/` are committed; the notebook was re-run from a fresh clone of the repository with identical results for everything deterministic.
""")

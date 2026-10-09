# BITS Kubeflow lab runbook: what worked, what failed, what to do next time

Written 2026-10-09 after the first lab session. Follow the checklist in order. Every "DON'T" below cost us time.
Namespace: `2025ae05142-s1-26-aimlzg536`. Dashboard: https://kubeflow-csisrlab.bits-pilani.ac.in/?ns=2025ae05142-s1-26-aimlzg536

## 0. Limits of this namespace (the root of most failures)

| Limit | Value | Consequence |
|---|---|---|
| CPU (requests) | 8 total | Request at most 7.5 (the Istio sidecar adds 0.1). Even 7.5 stayed `Unschedulable` ("Insufficient cpu" on the GPU nodes) the second time, so use **4 CPU**. |
| Memory (requests) | 16 Gi total | Request **15.5 Gi** (sidecar adds ~128 Mi). 32 Gi was rejected with `exceeded quota`. |
| Storage (PVCs) | 32 Gi total | A 15 Gi data volume plus two 6 Gi workspace volumes already used 27 Gi. A new 6 Gi volume was rejected with 403. |
| GPU | 1 NVIDIA on toleration group "NVIDIA A100 GPU Node" | Only A100 and "RTX 6000-GPU" exist. No L40S. The A100 is 80 GB and shared (another tenant held ~12 GB). |

## 1. Next-session checklist (about 10 min to a running server)

1. Book the slot in Prayogshala, Start Session, sign in again at the Dex page, open the Kubeflow dashboard.
2. **Volumes page first.** Existing volumes: `llm-data` (15 Gi NFS, holds the venv, repo and logs) plus two empty leftovers,
   `llm-lab-workspace` and `llm-lab-ws` (6 Gi each). Delete the two leftovers (red trash icon, you must click it yourself:
   my automated clicks never got a confirmation dialog). That frees 12 Gi of storage quota.
3. Notebooks, New Notebook, fill the form:
   - Name: `llm-gpu` (any name).
   - Type: JupyterLab. Image: Custom Notebook dropdown, pick `kubeflownotebookswg/jupyter-pytorch-cuda-full:v1.10.0-rc.1`.
     (Optional: try `10.152.183.210:5000/bits-sudo-jupyter-pytorch-cuda-full` first, it is the guide's image, but it failed with
     `ImagePullBackOff ... :latest not found`. Look at the Events tab within 1 minute and fall back immediately if it fails.)
   - CPU 4, memory 15.5. GPUs: 1, vendor NVIDIA. Affinity/Tolerations, Tolerations Group: NVIDIA A100 GPU Node.
   - **Workspace volume: delete it** (trash icon). Storage quota cannot take a new one and the guide says it is scratch only.
   - Data Volumes: Attach existing volume, Type Kubernetes Volume, name `llm-data`, mount path `/home/jovyan/data`.
   - Shared memory: leave enabled. Launch.
4. "Submitting new Notebook..." can hang for up to ~90 s. **Do not click Launch twice.** Open the Notebooks list instead.
   A green tick means ready. If it shows a warning, open the details page and read the message.
5. Connect, open a Terminal, then:
   ```bash
   cd ~/data && ls                      # venv, setup_env.sh, check_env.py, fine-tuning-Mistral-7B should still be there
   bash ./setup_env.sh                  # re-registers the kernel; fast if the venv survived (full install took ~10 min)
   ./venv/bin/python ./check_env.py     # must end with "All checks passed"
   ```
   If `venv/` is gone, `setup_env.sh` rebuilds it. Budget 10+ minutes on NFS (the guide's "2-4 minutes" is optimistic).
6. Install the extras Part A needs, into the venv only:
   `./venv/bin/pip install -q pymupdf langdetect nbformat nbclient nbconvert hf_transfer`
7. Repo: `cd ~/data/fine-tuning-Mistral-7B && git fetch && git switch lab-run` (or clone with `-b lab-run` once the branch is pushed).
8. Take the evidence screenshots again (notebook running, details, volumes, memory/storage/GPU, check_env) into `lab_evidence/`.

## 2. DON'T list

- **DON'T `pip install -r requirements.txt`** from the repo. It pins torch 2.8.0 and bitsandbytes 0.50.2 and the notebooks' first
  cell does exactly that. The instructor's pins win: torch 2.5.1+cu124, transformers 4.46.3, peft 0.13.2, accelerate 1.1.1,
  bitsandbytes 0.44.1, trl 0.12.1. Never upgrade or downgrade those. Only add missing packages (step 6).
- DON'T click **New Volume** on the Volumes page: it is disabled for this account. Volumes are created in the notebook form.
- DON'T request more than 4 CPU / 15.5 Gi, and DON'T add a second volume (storage quota).
- DON'T set `HF_HOME`/`TRANSFORMERS_CACHE` to `~/data`. The 15 Gi volume cannot hold the 14.5 GB Mistral weights next to the venv.
  The default cache on the container disk (873 GB free) is fine; it is re-downloaded after a restart.
- DON'T use `~` for outputs you want to keep: with no workspace volume the home directory is ephemeral. Use `./` under `~/data`.
- DON'T try to edit a launched server (the UI cannot). Delete it and recreate it; the data volume survives deletion.
- DON'T name a new server like an old one if it creates a `<name>-workspace` PVC that already exists.
- DON'T load models onto the CPU first (RAM is 15.5 Gi): use `device_map="cuda"`, `low_cpu_mem_usage=True`.
- DON'T rely on a browser-attached kernel for long runs. Use the terminal with `nohup ... > log 2>&1 &`
  or `jupyter nbconvert --to notebook --execute`, and poll the log.
- DON'T `| tail` a long command if you want live output: it buffers until the end. Use `tee file.log` and read the file.
- DON'T trust single timings: the A100 is shared. Warm up, repeat, and measure VRAM with `torch.cuda.max_memory_allocated`.
- DON'T delete old reports, adapters or outputs until the part is finished and the user has approved (user rule).
  Lab work goes on branch `lab-run`; `main` keeps the old outputs. PR to `main` only after verification.

## 3. Facts verified in session 1

- Pod: A100-SXM4-80GB, driver 535.309.01, CUDA 12.4, Python 3.11.11, torch 2.5.1+cu124, `cuda.is_available()` True.
- Network from the pod: github.com, huggingface.co and pypi.org all return HTTP 200, so git clone and model download work.
  (`cdn-lfs.huggingface.co` returned 000; that hostname is obsolete. A real model download is the true test.)
- Data volume mount: NFS, appears as 15T in `df`; the enforced quota is the 15 Gi PVC shown in Kubeflow.
- Pod limits seen inside the container: memory limit 18.6 GiB (request 15.5 x 1.2), 4 CPU request (`nproc` shows the host's 256).
- `check_env.py` result: all pinned versions OK, 4-bit bitsandbytes forward on GPU OK, peft LoRA OK.
- transformers 4.46.3 supports universal assisted generation (`assistant_tokenizer`), which Part C2 needs because SmolLM2's
  vocabulary differs from Mistral's.

## 4. Tips for driving the lab through browser automation (only if Claude does it again)

- Edge needs remote debugging enabled (`edge://inspect/#remote-debugging`) and an "Allow remote debugging?" prompt can reappear:
  click Allow when the tool stalls.
- The Kubeflow pages live in an iframe inside a shadow root. Coordinate clicks drift when the page scrolls. Drive the DOM instead.
- Angular Material selects open only with `mousedown/mouseup/click` events after `scrollIntoView`, and stale `mat-option`
  elements from earlier dropdowns stay in the DOM: always pick the **last** matching option.
- Fill inputs with the native value setter plus `input`/`change` events.
- In the Jupyter terminal, `type_text` the command and press Enter separately; a trailing newline does not run it.
- The Jupyter REST API can read files live: `/notebook/<ns>/<server>/api/contents/data/<file>?content=1`.

## 5. Use the next session's time well

Prepare locally, before booking, so lab minutes go to execution only: Part C code in `src/` (decoding, speculative, 4-bit cost),
the 7 extra Part C prompts, the lab-aware install cell, and the skeleton of `assignment_1b.ipynb`.
Then in the lab, in this order: environment, Part A re-run, Part B re-run (QLoRA about 1 minute of training, the model
download is the slow part), Part C, build and execute `assignment_1b.ipynb`, export HTML, screenshots, commit.
See `LAB_RUN_PLAN.md` for the phases and `lab_evidence/` for the proof screenshots.

# BITS Kubeflow lab runbook: what worked, what failed, what to do next time

Written 2026-10-09 after the first lab session. Follow the checklist in order. Every "DON'T" below cost us time.
Namespace: `2025ae05142-s1-26-aimlzg536`. Dashboard: https://kubeflow-csisrlab.bits-pilani.ac.in/?ns=2025ae05142-s1-26-aimlzg536

## Status and next steps

**See `../PLAN.md`** for what is delivered, what is left (Part C and the final steps), the working rules and the file map. This runbook covers only how to set up and drive a lab session.
Two facts you need before starting: the final work is on `main` (the branch `lab-run` was merged into it), and the trained adapter is **committed** in `adapters/adapter_b/`, so a new pod does not need to retrain it before B3.

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
   # if setup_env.sh / check_env.py are missing, the repo carries the instructor's copies (unchanged):
   #   cp fine-tuning-Mistral-7B/lab_runbook/instructor_scripts/* ~/data/
   bash ./setup_env.sh                  # re-registers the kernel; fast if the venv survived (full install took ~10 min)
   ./venv/bin/python ./check_env.py     # must end with "All checks passed"
   ```
   If `venv/` is gone, `setup_env.sh` rebuilds it. Budget 10+ minutes on NFS (the guide's "2-4 minutes" is optimistic).
6. Install the extras Part A needs, into the venv only:
   `./venv/bin/pip install -q pymupdf langdetect nbformat nbclient nbconvert hf_transfer`
7. Repo: `cd ~/data/fine-tuning-Mistral-7B && git pull` on `main` (on a fresh volume: `git clone https://github.com/bheemreddy1729/fine-tuning-Mistral-7B`).
8. Take the evidence screenshots again (notebook running, details, volumes, memory/storage/GPU, check_env) into `lab_evidence/`.

## 2. DON'T list

- **DON'T install or upgrade anything that is in the instructor's list**: torch 2.5.1+cu124, transformers 4.46.3, peft 0.13.2, accelerate 1.1.1,
  bitsandbytes 0.44.1, trl 0.12.1. `requirements.txt` in this repo now holds only the extras the image lacks (pymupdf, langdetect, nbformat, nbclient,
  nbconvert, hf_transfer), so `../venv/bin/pip install -r requirements.txt` is safe; any older copy that pins torch or bitsandbytes is not.
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
  Work on a branch and open a pull request to `main` only after verification on the lab (the final submission is already on `main`).

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

Prepare locally, before booking, so lab minutes go to execution only (the notebook generators in `tools/` were removed from `main`; restore them with `git checkout 4ebe3bc -- tools` if the notebook must change). Then in the lab, in this order: environment (section 1), check that `adapters/adapter_b/` is present after the clone (it is committed), build and execute
`assignment_1b.ipynb` (section 6), export HTML, screenshots into `lab_evidence/`, commit.

## 6. Lessons from building the unified notebook on the lab (session 1, later part)

- `assignment_1b.ipynb` was generated by `tools/build_assignment_notebook.py` (all code is inline in cells; the generators are no longer on `main`, restore with `git checkout 4ebe3bc -- tools`). To re-execute the committed notebook on the pod (skip the build step unless `tools/` was restored and edited):
  (optional) `../venv/bin/python tools/build_assignment_notebook.py`, then
  `nohup ../venv/bin/python -m nbconvert --to notebook --execute --inplace --allow-errors --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=llm-venv assignment_1b.ipynb > ../nb_run.log 2>&1 &`.
- The venv has no `jupyter` script: use `../venv/bin/python -m nbconvert` to execute, but export HTML with `/opt/conda/bin/jupyter nbconvert --to html assignment_1b.ipynb`
  (the venv's nbconvert cannot find the `lab` template).
- Run it with `--allow-errors` and read the log; nbconvert writes the notebook only at the end, so watch `du -sm ~/.cache/huggingface` to follow the model download.
- Model download took ~4 min for 14.5 GB (about 65 MB/s once running; slow for the first minutes). Cached loads take ~15 s. Set `HF_HUB_DISABLE_PROGRESS_BARS=1` so the notebook output stays clean.
- Moving files to the pod without GitHub: Jupyter contents API `PUT /notebook/<ns>/<server>/api/contents/<path>` with the `_xsrf` cookie as `X-XSRFToken`.
  The browser tool limits a single call to ~100 KB, so send large files (images) in 40 KB chunks into a `window` variable first.
  Do not return a promise from a long fetch (the tool waits 5 s); start it, then poll a window variable.
- Before every `type_text` into the terminal, redefine your command variable: a stale variable from an earlier call silently re-ran an old command once.
- The pod's repo clone does not contain files created locally after cloning (e.g. `lab_evidence/`); upload them or push the branch first.

## 7. Lessons from building Part C (session 1, evening)

- **The browser tool acts on the active tab.** If the user (or another tool) switches tabs, `fetch('/notebook/...')` runs against the wrong origin and every call times out. Switch to the JupyterLab tab (`switch_tab`) before each batch of API calls and re-check `page_info()['url']`.
- **The pod's API gets slow while generation saturates the 4 CPUs** (log fetches and uploads time out). Retry, poll with short calls, and upload large files while the pod is idle. A timed-out upload may leave the old file in place: verify it (for example that the file contains a new symbol) before starting a run.
- **Never type multi-line commands into the terminal**: newlines are flattened and a heredoc hangs at the `>` prompt. Upload a script through the contents API and run it, or use one-line commands (cancel a stuck prompt with `type_text('\x03')`).
- **Set `HF_HUB_ENABLE_HF_TRANSFER=1` in the shell before `nohup`**: the 3.4 GB SmolLM2-1.7B download took about 14 minutes without it and a few seconds once cached. The draft models (`HuggingFaceTB/SmolLM2-1.7B-Instruct`, `SmolLM2-360M-Instruct`) and Mistral live in `~/.cache/huggingface` (container disk), so a new pod downloads them again.
- **Develop one section at a time** (needs the restored `tools/`). `tools/_dev_c.py` (git-ignored, pod only) builds a mini notebook with only the Part C section you name (`c1`, `c1setup c2`, `c1setup c2setup c3`), so a change can be tested in minutes instead of rerunning the whole notebook. Run it with `nbconvert --execute --allow-errors`, pull the `*_executed.ipynb`, and read the outputs before writing any analysis.
- **Timings on this shared A100 vary by about 25 % between runs.** Part C therefore alternates the compared configurations and reports ratios. Plain Mistral-7B greedy decoding runs at 25 to 35 tokens/s here (overhead-bound, not bandwidth-bound).
- **Run times on the lab:** the full notebook (Parts A to C) ran in about 55 minutes in the final run; the model downloads (Mistral 14.5 GB, SmolLM2-1.7B 3.4 GB) are extra on a new pod.

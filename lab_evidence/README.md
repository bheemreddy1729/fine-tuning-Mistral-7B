# Lab evidence (BITS Prayogshala / Kubeflow)

Screenshots proving the assignment runs on the BITS lab infrastructure. Captured 2026-10-09.

| File | Shows |
|---|---|
| 01_kubeflow_notebook_running.png | Server `llm-gpu` Running: 1 GPU, 4 CPU, 15.5 Gi |
| 02_kubeflow_notebook_details_image_cpu_memory.png | Image, CPU/memory requests and limits, `llm-data` mounted, shared memory on |
| 03_kubeflow_volumes_llm-data_15Gi_nfs.png | Data volume `llm-data`: 15 Gi, nfs-client, used by `llm-gpu` |
| 04_pod_terminal_memory_storage_gpu.png | Pod memory limit/usage, data-volume storage, GPU memory |
| 05_pod_terminal_nvidia-smi_A100.png | `nvidia-smi`: NVIDIA A100-SXM4-80GB |
| 06_repo_cloned_on_data_volume.png | Repo and the instructor's setup scripts on the 15 GB data volume |
| 07_check_env_all_checks_passed.png | Instructor `check_env.py`: pinned versions OK, A100 visible, 4-bit GPU test and peft LoRA pass |
| 08_final_run_complete_memory_storage_gpu.png | After the final clean-clone run: log ends with `ALLDONE`; pod memory limit 18.60 GiB (15.59 GiB in use incl. cache); the 15 Gi data volume holds venv 1.2 GB + repo 193 MB; A100 80 GB idle |
| 09_kubeflow_notebook_running_after_full_run.png | Server `llm-gpu` still Running 11 hours after creation (1 GPU, 4 CPU, 15.5 Gi), last activity 1 minute earlier |
| 10_kubeflow_volumes_after_full_run.png | Volumes after the full run: `llm-data` 15 Gi nfs-client, used by `llm-gpu` |
| setup_env.log, check_env.log, llm_baseline_requirements.txt | Raw output of the instructor scripts and the frozen package versions from the pod |

Notes: the lab registry image `bits-sudo-jupyter-pytorch-cuda-full` had no tag available, so the server uses
`kubeflownotebookswg/jupyter-pytorch-cuda-full:v1.10.0-rc.1` (torch 2.5.1+cu124, Python 3.11, same stack as the guide).
CPU and memory are capped by the namespace quota (8 CPU, 16 Gi RAM, 32 Gi storage).

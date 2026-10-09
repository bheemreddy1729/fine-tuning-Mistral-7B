#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_env.sh
# LLM / NLP course baseline for Kubeflow notebook pods, installed into a
# Python venv that lives on your 15 GB NFS data volume.
#
# WHERE TO RUN:
#   From a terminal, INSIDE your data-volume working directory (the 15 GB one).
#   This script and check_env.py should sit in that same directory.
#
#   cd <your data-volume directory>
#   bash ./setup_env.sh
#
# WHAT IT DOES:
#   - Creates ./venv (once) using --system-site-packages, so the venv inherits
#     the image's working torch + CUDA instead of re-downloading ~3 GB of GPU
#     wheels. Only the LLM libraries are installed fresh into the venv.
#   - Installs the pinned course baseline into ./venv.
#   - Registers ./venv as a Jupyter kernel called "Python (LLM venv)".
#   - Saves exact versions to ./llm_baseline_requirements.txt.
#
#   The script is idempotent: the first run installs everything (2-4 min),
#   later runs are fast and mainly re-register the kernel.
#
# AFTER IT FINISHES:
#   1. In your notebook: Kernel -> Change Kernel -> "Python (LLM venv)"
#   2. Verify:           ./venv/bin/python ./check_env.py
# ---------------------------------------------------------------------------

set -e

VENV_DIR="./venv"
KERNEL_NAME="llm-venv"
KERNEL_DISPLAY="Python (LLM venv)"

echo "==========================================="
echo " LLM Course Environment Setup (venv on NFS)"
echo " Working directory: $(pwd)"
echo " Started: $(date)"
echo "==========================================="

echo ""
echo "--- [0/5] Location check ---"
echo "Installing the venv at: $(pwd)/venv"
echo "Make sure this is your 15 GB NFS data volume, NOT the pod home directory,"
echo "so the venv persists across pod restarts."

echo ""
echo "--- [1/5] GPU and Python check ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. QLoRA / 4-bit needs a GPU pod."
fi

if command -v python3.11 &> /dev/null; then
    PYBIN="python3.11"
else
    PYBIN="python3"
    echo "NOTE: python3.11 not found, falling back to: $($PYBIN --version 2>&1)"
fi
echo "Using interpreter: $($PYBIN --version 2>&1)"

echo ""
echo "--- [2/5] Virtual environment ---"
if [ -d "$VENV_DIR" ]; then
    echo "venv already exists at $VENV_DIR (reusing it)."
else
    echo "Creating venv at $VENV_DIR (with --system-site-packages)..."
    $PYBIN -m venv --system-site-packages "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"

"$PY" - <<'PYCHECK'
try:
    import torch
    print(f"Inherited torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"NOTE: torch not visible in venv yet ({e}). It will be used from the base image.")
PYCHECK

echo ""
echo "--- [3/5] Installing LLM baseline into venv (2-4 minutes) ---"
"$PIP" install --upgrade pip --quiet
"$PIP" install --quiet \
    "transformers==4.46.3" \
    "tokenizers==0.20.3" \
    "datasets==3.1.0" \
    "accelerate==1.1.1" \
    "peft==0.13.2" \
    "trl==0.12.1" \
    "bitsandbytes==0.44.1" \
    "safetensors==0.4.5" \
    "huggingface_hub==0.26.2" \
    "sentencepiece==0.2.0" \
    "protobuf==4.25.5" \
    "evaluate==0.4.3" \
    "sentence-transformers==3.3.1" \
    "faiss-cpu==1.9.0" \
    "scikit-learn==1.5.2" \
    "pandas==2.2.3" \
    "numpy>=2.0,<3" \
    "matplotlib==3.9.2" \
    "seaborn==0.13.2" \
    "tqdm==4.67.0" \
    "ipywidgets==8.1.5" \
    "wandb==0.18.7" \
    "ipykernel"

echo ""
echo "--- [4/5] Registering Jupyter kernel ---"
"$PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"
echo "Registered kernel: $KERNEL_DISPLAY (name: $KERNEL_NAME)"

echo ""
echo "--- [5/5] Freezing versions ---"
"$PIP" freeze > ./llm_baseline_requirements.txt
echo "Saved: $(pwd)/llm_baseline_requirements.txt"

echo ""
echo "==========================================="
echo " Setup complete"
echo "==========================================="
echo ""
echo "NEXT STEPS:"
echo "  1. In your notebook: Kernel -> Change Kernel -> \"$KERNEL_DISPLAY\""
echo "  2. Verify:           ./venv/bin/python ./check_env.py"
echo ""
echo "After a pod restart, the venv on the data volume PERSISTS."
echo "You normally only need to re-select the kernel. If the kernel is missing"
echo "from the menu, just re-run this script - it will re-register it quickly."
echo ""

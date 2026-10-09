#!/usr/bin/env python
"""
check_env.py
------------
Verify the LLM course baseline environment installed in your venv.

Run with the venv's interpreter (no activation needed):
    ./venv/bin/python ./check_env.py

Or, if you have already selected the "Python (LLM venv)" kernel and opened a
terminal with the venv activated:
    python ./check_env.py

Exits 0 on success, 1 if any check fails.
"""

import importlib
import sys

EXPECTED = {
    "torch":                 "2.",       # inherited from base image (2.5.x)
    "transformers":          "4.46.3",
    "tokenizers":            "0.20.3",
    "datasets":              "3.1.0",
    "accelerate":            "1.1.1",
    "peft":                  "0.13.2",
    "trl":                   "0.12.1",
    "bitsandbytes":          "0.44.1",
    "safetensors":           "0.4.5",
    "huggingface_hub":       "0.26.2",
    "sentencepiece":         "0.2.0",
    "evaluate":              "0.4.3",
    "sentence_transformers": "3.3.1",
    "faiss":                 None,       # faiss-cpu reports module name 'faiss'
    "sklearn":               "1.5.2",
    "pandas":                "2.2.3",
    "numpy":                 "2.",       # numpy 2.x matches the image's scipy ABI
    "matplotlib":            "3.9.2",
    "wandb":                 "0.18.7",
}


def line(char="-", n=60):
    print(char * n)


def check_interpreter():
    print("\n--- Interpreter ---")
    line()
    print(f"  python executable: {sys.executable}")
    if "venv" in sys.executable:
        print("  OK: running inside the venv.")
        return True
    else:
        print("  WARNING: this does not look like the venv interpreter.")
        print("  Run as: ./venv/bin/python ./check_env.py")
        print("  (or select the 'Python (LLM venv)' kernel in the notebook).")
        return True  # warning only, not a hard failure


def check_imports():
    print("\n--- Package imports & versions ---")
    line()
    ok = True
    for pkg, expected_prefix in EXPECTED.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "(no __version__)")
            marker = "OK"
            if expected_prefix and not str(ver).startswith(expected_prefix):
                marker = f"MISMATCH (expected starts with {expected_prefix})"
                ok = False
            print(f"  {pkg:<24} {ver:<15} {marker}")
        except ImportError as e:
            print(f"  {pkg:<24} MISSING       ({e})")
            ok = False
    return ok


def check_cuda():
    print("\n--- CUDA / GPU ---")
    line()
    try:
        import torch
        avail = torch.cuda.is_available()
        print(f"  torch.cuda.is_available(): {avail}")
        if avail:
            print(f"  CUDA build version:        {torch.version.cuda}")
            print(f"  GPU name:                  {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU memory:                {mem_gb:.1f} GB")
            return True
        else:
            print("  WARNING: No CUDA. LoRA on CPU works; QLoRA will not.")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_bitsandbytes_gpu():
    print("\n--- bitsandbytes 4-bit GPU test ---")
    line()
    try:
        import torch
        import bitsandbytes as bnb
        if not torch.cuda.is_available():
            print("  SKIPPED (no GPU available)")
            return True
        x = torch.randn(4, 4).cuda()
        layer = bnb.nn.Linear4bit(4, 4).cuda()
        y = layer(x)
        print(f"  4-bit Linear forward OK. Output shape: {tuple(y.shape)}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Likely a CUDA / bitsandbytes version mismatch.")
        return False


def check_peft_lora():
    print("\n--- peft LoRA wiring test ---")
    line()
    try:
        from peft import LoraConfig
        cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], task_type="CAUSAL_LM")
        print(f"  LoraConfig built OK: r={cfg.r}, alpha={cfg.lora_alpha}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print(" LLM Course Environment Verification (venv)")
    print("=" * 60)

    results = [
        ("Interpreter",                check_interpreter()),
        ("Package imports",            check_imports()),
        ("CUDA / GPU",                 check_cuda()),
        ("bitsandbytes 4-bit on GPU",  check_bitsandbytes_gpu()),
        ("peft LoRA",                  check_peft_lora()),
    ]

    print("\n--- Summary ---")
    line()
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print(" All checks passed. You're ready for LoRA / QLoRA experiments.")
        sys.exit(0)
    else:
        print(" Some checks failed. See messages above.")
        print(" Common fixes:")
        print("   - Did you select the 'Python (LLM venv)' kernel in the notebook?")
        print("   - Is your pod a GPU pod (L40S or A100) with the CUDA image?")
        print("   - Try re-running: bash ./setup_env.sh   (from your data volume dir)")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Preflight check - exercises EVERY point that has broken before, on the LOGIN
NODE (free, no GPU, no SBUs), so a broken env / data / model is caught before you
sbatch anything. Each check is isolated so it reports ALL failures, not just the first.

Covers, in order, the exact things that failed during setup:
  1. core package imports  (huggingface_hub, peft, sklearn 1.5.2, cv2.imread,
     numpy 1.x, torchvision rpn -> the Qwen model class)
  2. pipeline module imports (action_predict, pie_data, lora_prompt_variant, ...)
  3. Qwen model shards present and ~full size (scratch-purge / truncation)
  4. the Qwen processor loads
  5. the VAL set (set05/06) loads end-to-end through the real data pipeline
     (PIE annotations + PVIM predict + cv2.imread on real frames + contact sheets)

Run (login node):
    module purge && module load 2023 \\
      TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
    source .venv-qwen/bin/activate
    export PYTHONNOUSERSITE=1
    python future_work/preflight.py
Exit code 0 = safe to sbatch; non-zero = something to fix first.
"""
import sys
import os

# Run exactly like the SLURM job: from the repo root, with the root + future_work
# on the path — so pipeline modules (pie_data, ...) and relative data paths resolve
# no matter which directory the preflight was launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
sys.path[:0] = [_ROOT, _HERE]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

QWEN = os.environ.get("QWEN_MODEL", "models/Qwen2.5-VL-32B-Instruct")
PVIM_DIR = "data/models/pie/PVIM/ckpt_tte60_120"
VAL_CFG = "config_files/PVIM_eval_tte90.yaml"

FAILS = []


def check(name, fn):
    try:
        fn()
        print(f"  [ OK ] {name}", flush=True)
    except Exception as e:
        print(f"  [FAIL] {name}\n         -> {type(e).__name__}: {e}", flush=True)
        FAILS.append(name)


# 1 -------------------------------------------------------------------------
def _core():
    import numpy
    assert numpy.__version__.startswith("1."), f"numpy must be 1.x, got {numpy.__version__}"
    import torch, torchvision, tensorflow, transformers, peft, cv2, sklearn, yaml, pandas, huggingface_hub  # noqa
    assert hasattr(cv2, "imread"), "cv2 imported but has no imread (broken opencv)"
    from torchvision.transforms import InterpolationMode  # noqa  (torchvision rpn chain)
    from transformers import (Qwen2_5_VLForConditionalGeneration,  # noqa
                              AutoProcessor, get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, TaskType  # noqa
    from qwen_vl_utils import process_vision_info  # noqa


# 2 -------------------------------------------------------------------------
def _pipeline():
    import action_predict, pie_data, stratified_run_qwen_speed  # noqa
    import prompt_ablation, prompt_ablation_cues, lora_sweep, lora_validated  # noqa
    import lora_confirm_winner, lora_finetune_qwen, lora_prompt_variant  # noqa


# 3 -------------------------------------------------------------------------
def _model_shards():
    import glob
    shards = glob.glob(os.path.join(QWEN, "*.safetensors"))
    assert shards, f"no *.safetensors in {QWEN} (purged? wrong path?)"
    total_gb = sum(os.path.getsize(s) for s in shards) / 1e9
    assert total_gb > 50, f"only {total_gb:.1f} GB across {len(shards)} shards (expected ~64; truncated?)"
    print(f"         {len(shards)} shards, {total_gb:.1f} GB", flush=True)


# 4 -------------------------------------------------------------------------
def _processor():
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(QWEN, use_fast=False)


# 5 -------------------------------------------------------------------------
def _val_data():
    from prompt_ablation import load_raw
    from prompt_ablation_cues import assemble
    raw = load_raw(VAL_CFG, PVIM_DIR, "val", None)      # set05/06 via PIE native split
    n = len(raw["gt"])
    assert n > 0, "val loaded 0 samples"
    samples = assemble(raw, frozenset({"pvim"}))         # no-PVIM 6-cue prompt
    assert len(samples) == n and samples[0]["prompt"], "assemble produced empty prompts"
    print(f"         val={n} samples, {len(samples)} prompts assembled (cv2/PVIM/sheets all ran)", flush=True)


print("=== PREFLIGHT (login node, no SBUs) ===", flush=True)
print(f"    QWEN model : {QWEN}", flush=True)
check("1. core package imports (numpy1.x, torch, torchvision, transformers, peft, cv2.imread, sklearn, Qwen class)", _core)
check("2. pipeline modules (action_predict, pie_data, lora_prompt_variant, ...)", _pipeline)
check("3. Qwen model shards present + ~64 GB", _model_shards)
check("4. Qwen processor loads", _processor)
check("5. VAL set05/06 loads end-to-end (PIE + PVIM + cv2 + contact sheets)", _val_data)

print("", flush=True)
if FAILS:
    print(f"PREFLIGHT FAILED ({len(FAILS)}/5): " + "; ".join(FAILS), flush=True)
    print("Fix the above before sbatch (do NOT waste SBUs).", flush=True)
    sys.exit(1)
print("PREFLIGHT OK (5/5) - safe to sbatch the ablation.", flush=True)

#!/bin/bash
# Build a clean per-project venv for the PVIM + Qwen2.5-VL pipeline.
# Run on the LOGIN NODE. ~5 minutes total.
#
# Layout (Snellius best practice):
#   - System modules supply TF, torch, numpy, Pillow, scipy (via --system-site-packages)
#   - This venv supplies only project deps: torchvision, transformers, qwen-vl-utils, accelerate
#   - ~/.local is bypassed (venv with --system-site-packages does NOT include user-site)
#   - Models live on scratch (already done): models/Qwen2.5-VL-32B-Instruct/
#
# Usage:  bash setup_qwen_env.sh

set -e

VENV=/scratch-shared/lvinkestijn/PVIM-Thesis/.venv-qwen

module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "=== Moving any stale venv aside (mv is instant; rm -rf is slow on scratch) ==="
if [ -e "$VENV" ]; then
    mv "$VENV" "${VENV}.OLD-$(date +%s)" || {
        echo "mv failed; falling back to rm (this can take many minutes)"
        rm -rf "$VENV"
    }
fi

echo "=== Creating fresh venv (inherits modules, ignores ~/.local) ==="
python -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"
echo "venv python: $(which python)"

# Stop pip from also reading ~/.local — we want this venv airtight.
export PYTHONNOUSERSITE=1

pip install --upgrade pip

echo "=== Installing torchvision matched to torch 2.1.2 (from PyTorch wheel index) ==="
# +cu121 build, no deps so torch from the module isn't replaced
pip install --no-deps --no-cache-dir "torchvision==0.16.2+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing transformers + Qwen helpers + LoRA (peft) ==="
# transformers 4.49-4.54 supports Qwen2.5-VL and works with torch 2.1.2.
# peft is required for LoRA fine-tuning (imported lazily inside attach_lora).
# Versions are pinned to the environment that actually produced the thesis results
# (see future_work/saved_models/requirements-lock.txt). Ranges are deliberately avoided:
# every time pip was left free to resolve, it pulled something incompatible (a PyAV that
# broke torchvision, a safetensors that landed without its RECORD, etc).
# NOTE: qwen-vl-utils MUST be pinned here - its installed dist-info metadata is corrupt,
# so pip freeze cannot see it and it is absent from requirements-lock.txt.
# NOTE: the env that produced the results had peft from a git build reporting 0.13.2;
# the PyPI pin below is the reproducible equivalent.
pip install \
    "transformers==4.54.1" \
    "huggingface_hub==0.34.4" \
    "safetensors==0.4.5" \
    "tokenizers==0.21.4" \
    "qwen-vl-utils==0.0.14" \
    "accelerate==1.1.1" \
    "peft==0.13.2"

# Remove PyAV. qwen-vl-utils pulls the latest PyAV (12+), which removed the
# `av.logging` attribute that torchvision 0.16.2 calls at import time -> importing
# torchvision (and therefore transformers' image utils / the Qwen VL class) crashes
# with "module 'av' has no attribute 'logging'". We only use image inputs (contact
# sheets), not video, and torchvision imports fine without av (video backend off),
# so the cleanest fix is to drop av entirely rather than pin a version.
pip uninstall -y av 2>/dev/null || true

# NOTE: llama-cpp-python (the old LLaVA path) is intentionally NOT installed.
# It builds from source, routinely hangs/fails on the login node, and nothing in
# the Qwen/LoRA pipeline imports it (only stratified_run.py / llava_pvim_cue.py do).
# If you ever need the LLaVA path again, install it separately.

echo "=== Installing PVIM data-pipeline deps (action_predict.py / pie_data.py) ==="
# These are imported at module level by the PVIM data loader; pandas + pyyaml are
# imported by action_predict.py, so the import fails at load time without them.
# opencv-python-headless (not opencv-python): the regular build imports on the cluster but
# has no working imread. scikit-learn 1.9.0 is what the results were produced with - an
# earlier failure blamed on this version was actually an incomplete install, not the version.
# pandas and pyyaml are left unpinned: the modules already supply them via
# --system-site-packages, so pip installs nothing for them.
pip install \
    "wget==3.2" \
    "opencv-python-headless==5.0.0.93" \
    "scikit-learn==1.9.0" \
    "matplotlib==3.11.0" \
    "einops==0.8.2" \
    pandas \
    pyyaml

echo "=== Guard: keep numpy at the module version (opencv/others may pull numpy 2.x, ==="
echo "===        which shadows the module's 1.25.1 and breaks torch 2.1.2) ==="
# Remove any venv-local numpy so the module's numpy (via --system-site-packages) is used.
pip uninstall -y numpy 2>/dev/null || true
python -c "import numpy; print('numpy in use:', numpy.__version__, numpy.__file__)"

echo ""
echo "=== Verify ==="
python -c "
import sys, torch, torchvision, transformers, qwen_vl_utils, tensorflow as tf, numpy, sklearn, cv2, peft, pandas, yaml
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
print(f'venv python : {sys.executable}')
print(f'tensorflow  : {tf.__version__}')
print(f'torch       : {torch.__version__}  (CUDA build: {torch.version.cuda})')
print(f'torchvision : {torchvision.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'peft        : {peft.__version__}')
print(f'numpy       : {numpy.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print('Qwen2.5-VL  : OK')
print('qwen_vl_utils: OK  |  peft LoRA: OK')
"

echo "=== Stamping all venv files with a fresh timestamp ==="
# Snellius' scratch cleanup is timestamp-based, and pip-installed files keep the
# old timestamps from their wheel archives - so a fresh venv can be cleaned up
# file-by-file within days. Stamping everything as new prevents that.
find "$VENV" -exec touch -h {} + 2>/dev/null || true

echo ""
echo "Venv ready at $VENV"
echo "To use in a job:"
echo "  source $VENV/bin/activate"
echo "  export PYTHONNOUSERSITE=1   # ignore ~/.local"

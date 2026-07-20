"""Smoke test: load the saved opt+no-PVIM adapter and score a few val samples.

Proves the full inference path on a compute node - imports, 68 GB model load,
peft adapter attach, processor, val data pipeline (PIE + PVIM + cv2 + sheets),
and soft-score inference - without paying for a training run. If this prints
SMOKE TEST OK, the ablation jobs are safe to launch (training loop excepted,
which is unchanged code that has run clean in three prior jobs).

Run via run_smoke_infer.job (GPU, ~20 min).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import argparse
import numpy as np

import torch
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda *a, **k: False

from prompt_ablation import load_raw, soft_scores
from prompt_ablation_cues import assemble
from lora_sweep import yes_token_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    ap.add_argument("--adapter", default="data/models/lora_2-4s_opt_nopvim")
    ap.add_argument("--model-dir", default="data/models/pie/PVIM/ckpt_tte60_120")
    ap.add_argument("--val-config", default="config_files/PVIM_eval_tte90.yaml")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    print("=== [1/4] Loading VAL set05/06 (full data pipeline) ===", flush=True)
    raw = load_raw(args.val_config, args.model_dir, "val", None)
    samples = assemble(raw, frozenset({"pvim"}))[: args.n]
    print(f"  val={len(raw['gt'])} samples, scoring first {len(samples)}", flush=True)

    print("=== [2/4] Loading base model (68 GB) ===", flush=True)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model, device_map="auto", torch_dtype=torch.bfloat16)

    print("=== [3/4] Attaching saved adapter ===", flush=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    processor = AutoProcessor.from_pretrained(args.qwen_model, use_fast=False)
    yes_ids, no_ids = yes_token_ids(processor)

    print("=== [4/4] Scoring ===", flush=True)
    scores = soft_scores(model, processor, samples, yes_ids, no_ids)
    for i, (s, sc) in enumerate(zip(samples, scores)):
        print(f"  sample {i}: gt={s['label_int']}  P(yes)={sc:.4f}", flush=True)

    scores = np.asarray(scores)
    assert len(scores) == len(samples), "missing scores"
    assert np.all((scores >= 0) & (scores <= 1)), "scores out of [0,1]"
    assert scores.std() > 1e-4, "all scores identical - inference degenerate?"
    print("\nSMOKE TEST OK - full inference path works; safe to launch the ablation.", flush=True)


if __name__ == "__main__":
    main()

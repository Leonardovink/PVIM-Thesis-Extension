"""Evaluate the saved 2-4s-optimized LoRA adapter at extra horizons (TTE 60/120).

No training - loads the base Qwen + the saved adapter once, then evaluates on the
set03 test split at each requested TTE, saving per-sample scores and printing full
stats (F1/F2/Acc/Prec/Rec, both operating points, VLM-only + fusion, +/-std under
the 20-seed protocol). The adapter comes from lora_confirm_winner.py.

Run via run_eval_lora_horizons.job (GPU; inference only, ~1 h for two horizons).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import csv
import argparse
import numpy as np

import torch
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda *a, **k: False

from lora_finetune_qwen import build_training_samples
from lora_sweep import yes_token_ids
from prompt_ablation import soft_scores
from lora_validated import run_protocol


def report(tag, res):
    for key in ("f1", "f2"):
        a = res[key]
        b = f"  beta={a['beta']:.2f}" if a["beta"] is not None else ""
        print(f"  {tag:9s}[{key.upper()}-opt]  F1={a['f1'][0]:.3f}+/-{a['f1'][1]:.3f}  "
              f"F2={a['f2'][0]:.3f}+/-{a['f2'][1]:.3f}  Acc={a['acc'][0]:.3f}  "
              f"Prec={a['prec'][0]:.3f}  Rec={a['rec'][0]:.3f}{b}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    ap.add_argument("--adapter", default="data/models/lora_2-4s_optimized")
    ap.add_argument("--model-dir", default="data/models/pie/PVIM/ckpt_tte60_120")
    ap.add_argument("--configs", default="config_files/PVIM_eval_tte60.yaml,"
                                         "config_files/PVIM_eval_tte120.yaml")
    ap.add_argument("--out-prefix", default="results/lora_opt_2-4s_eval")
    args = ap.parse_args()

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    print(f"=== Loading base + adapter {args.adapter} ===", flush=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    processor = AutoProcessor.from_pretrained(args.qwen_model, use_fast=False)
    yes_ids, no_ids = yes_token_ids(processor)

    for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
        tte = cfg.split("tte")[-1].split(".")[0]
        print(f"\n=== Eval @ TTE {tte}  ({cfg}) ===", flush=True)
        test = build_training_samples(config_path=cfg, model_dir=args.model_dir,
                                      split="test", set_ids=("set03",))
        gt = np.array([s["label_int"] for s in test])
        pvim = np.array([s["pvim_prob"] for s in test], dtype=float)
        print(f"  {len(test)} samples ({int(gt.sum())} cross, {len(gt)-int(gt.sum())} no-cross)", flush=True)

        scores = soft_scores(model, processor, test, yes_ids, no_ids)
        out = f"{args.out_prefix}{tte}_set03.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "pvim_prob", "qwen_score",
                                              "ground_truth", "vlm_answer"])
            w.writeheader()
            for s, sc in zip(test, scores):
                w.writerow({"sample_id": s["sample_id"], "pvim_prob": f"{s['pvim_prob']:.4f}",
                            "qwen_score": f"{sc:.4f}", "ground_truth": s["label_int"],
                            "vlm_answer": "yes" if sc >= 0.5 else "no"})
        print(f"  saved {out}", flush=True)
        report("VLM-only", run_protocol(gt, scores))
        report("VLM+PVIM", run_protocol(gt, scores, pvim=pvim, fuse=True))

    print("\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()

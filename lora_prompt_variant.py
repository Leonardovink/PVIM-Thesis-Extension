"""Train one prompt variant (optimised config) and SELECT on the native PIE
validation set (set05/06), keeping set03 untouched.

Used for the in-regime prompt study:
  * Ablation (A): run once per cue, dropping {pvim, <cue>}, ranked on val.
  * Trimming  (B): run with the chosen worst-1/2/3 cue set, compared on val.
Only the final chosen prompt is later evaluated on set03 (--eval-test), so the
prompt selection never sees the test set.

Config = the optimised winner (cosine+warmup, class-balanced, rank 8, 1 epoch),
trained on the 2-4s range. PVIM is removed from the prompt by design (no double
counting with the fusion), so every variant drops at least 'pvim'.

Run via run_ablation_6cue.job (A) or directly with --drop for B / the final.
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

from prompt_ablation import load_raw, soft_scores
from prompt_ablation_cues import assemble, CUE_ORDER
from lora_sweep import load_base, yes_token_ids, ALL
from lora_validated import run_protocol
from lora_confirm_winner import attach_lora, train_winner, report

WINNER = dict(name="opt", lr=1e-4, schedule="cosine", pos_weight="balanced",
              r=8, alpha=16, dropout=0.1, modules=ALL, accum=1, weight_decay=0.01,
              warmup_frac=0.05, max_grad_norm=1.0, epochs=1)


def score_report(model, processor, raw, dropped, yes_ids, no_ids, out_csv, tag, split):
    samples = assemble(raw, dropped)
    gt = raw["gt"]; pvim = raw["pvim"]
    scores = soft_scores(model, processor, samples, yes_ids, no_ids)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "pvim_prob", "qwen_score", "ground_truth"])
        w.writeheader()
        for i, sc in enumerate(scores):
            w.writerow({"sample_id": i, "pvim_prob": f"{pvim[i]:.4f}",
                        "qwen_score": f"{sc:.4f}", "ground_truth": int(gt[i])})
    print(f"  saved {out_csv}", flush=True)
    print(f"  [{tag}] {split}:", flush=True)
    report("VLM-only", run_protocol(gt, scores))
    report("VLM+PVIM", run_protocol(gt, scores, pvim=pvim, fuse=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    ap.add_argument("--model-dir", default="data/models/pie/PVIM/ckpt_tte60_120")
    ap.add_argument("--train-config", default="config_files/PVIM_tte60_120.yaml")   # 2-4s
    ap.add_argument("--val-config", default="config_files/PVIM_eval_tte90.yaml")     # 3s val slice
    ap.add_argument("--test-configs", default="config_files/PVIM_eval_tte60.yaml,"
                                              "config_files/PVIM_eval_tte90.yaml,"
                                              "config_files/PVIM_eval_tte120.yaml")
    ap.add_argument("--drop", default="pvim",
                    help="comma-separated cues to drop from the prompt (always includes pvim)")
    ap.add_argument("--tag", default="nopvim")
    ap.add_argument("--eval-test", action="store_true",
                    help="also evaluate on set03 (only for the final chosen prompt)")
    ap.add_argument("--out-prefix", default="results/promptstudy_")
    ap.add_argument("--save-dir", default="data/models/promptstudy",
                    help="trained adapters are saved to <save-dir>/<tag>/ so they can be reused")
    ap.add_argument("--from-adapter", default=None,
                    help="load this saved adapter and evaluate WITHOUT training (cheap rerun)")
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed seed for LoRA init + data shuffle so variants differ ONLY by prompt")
    args = ap.parse_args()

    dropped = frozenset(c.strip() for c in args.drop.split(",") if c.strip()) | {"pvim"}
    kept = [c for c in CUE_ORDER if c not in dropped]
    print(f"=== Variant '{args.tag}': drop={sorted(dropped)}  keep={kept} ===", flush=True)

    # Pre-flight: load VAL (set05/06) FIRST so a missing val set fails in minutes,
    # not after hours of training. All eval data is loaded before training starts.
    print(f"=== Pre-flight: loading VAL set05/06 ({args.val_config}) ===", flush=True)
    val_raw = load_raw(args.val_config, args.model_dir, "val", None)
    print(f"  val={len(val_raw['gt'])} samples", flush=True)
    test_raws = []
    if args.eval_test:
        for cfg in [c.strip() for c in args.test_configs.split(",") if c.strip()]:
            tte = cfg.split("tte")[-1].split(".")[0]
            print(f"=== Pre-loading TEST set03 @ TTE {tte} ===", flush=True)
            test_raws.append((tte, load_raw(cfg, args.model_dir, "test", ("set03",))))

    base = load_base(args.qwen_model)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.qwen_model, use_fast=False)
    yes_ids, no_ids = yes_token_ids(processor)

    adapter_dir = args.from_adapter or os.path.join(args.save_dir, args.tag)
    if args.from_adapter:
        # Cheap rerun: load a previously trained adapter, skip training entirely.
        from peft import PeftModel
        print(f"\n=== Loading saved adapter (NO training): {args.from_adapter} ===", flush=True)
        model = PeftModel.from_pretrained(base, args.from_adapter).eval()
    else:
        print(f"=== Building TRAIN ({args.train_config}, 2-4s) ===", flush=True)
        train_raw = load_raw(args.train_config, args.model_dir, "train", None)
        train_samples = assemble(train_raw, dropped)
        print(f"  train={len(train_samples)}", flush=True)
        # Fixed seed AFTER data load, right before LoRA init + training: same adapter
        # init, same shuffle order, same dropout draws for every variant, so the only
        # thing that differs across the ablation is the prompt content.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"  seed={args.seed}", flush=True)
        model = attach_lora(base, WINNER)
        print(f"\n=== Fine-tuning (cosine, 1 ep, 2-4s) ===", flush=True)
        model = train_winner(model, processor, list(train_samples), WINNER)
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        print(f"=== Saved adapter -> {adapter_dir} (reuse with --from-adapter {adapter_dir}) ===", flush=True)

    # SELECTION signal: native validation set (set05/06)
    print(f"\n=== VAL eval on set05/06 (selection) ===", flush=True)
    score_report(model, processor, val_raw, dropped, yes_ids, no_ids,
                 f"{args.out_prefix}val_{args.tag}.csv", args.tag, "val")

    # Final reporting only: set03 (never used to choose the prompt)
    for tte, raw in test_raws:
        print(f"\n=== TEST eval on set03 @ TTE {tte} (reporting only) ===", flush=True)
        score_report(model, processor, raw, dropped, yes_ids, no_ids,
                     f"{args.out_prefix}test_{args.tag}_tte{tte}.csv", f"{args.tag}@{tte}", "test")

    print("\n=== Done ===", flush=True)


if __name__ == "__main__":
    main()

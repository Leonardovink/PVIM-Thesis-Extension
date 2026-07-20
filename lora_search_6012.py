"""Training optimization for the 2-4s LoRA: find the best config for training on
the [60,120] range and evaluating at TTE 90.

Fixes the two flaws of lora_sweep.py:
  * epochs are searched *for free* via per-epoch evaluation (train once to
    max_epochs, evaluate after every epoch), so lr/rank/loss verdicts are no
    longer contingent on a single fixed training length;
  * a validation split is carved from the TRAIN sets (set01/02/04) - config and
    epoch are selected on val AUROC (threshold-free), and set03 is touched only
    to *report* the final F1/F2, never to select.

Adds cosine+warmup scheduling, gradient clipping, and (optional) DoRA. The grid
is a focused joint sweep of the interacting knobs; edit GRID below to resize.

Shardable/resumable like run_prompt_ablation_combo: --start/--end run a slice of
the config list, and any config already fully in --out is skipped.

Run via run_lora_search_6012.job (GPU).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import csv
import gc
import argparse
import itertools
import numpy as np

import torch
import torch.nn as nn
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda *a, **k: False

from sklearn.metrics import roc_auc_score

from lora_finetune_qwen import build_training_samples
from lora_sweep import load_base, yes_token_ids, _encode, ATTN, ALL
from prompt_ablation import soft_scores
from lora_validated import run_protocol, metrics, THRESHOLDS

MAX_EPOCHS = 2   # 2-4s LoRA overfits the training horizon fast; epoch 1-2 is the useful range

# --- Focused joint grid (edit to resize). epochs handled free via per-epoch eval.
GRID_AXES = dict(
    lr=[1e-4, 2e-4],
    schedule=["constant", "cosine"],
    pos_weight=[1.0, "balanced"],
    r=[8],                          # rank fixed at 8 (inert axis dropped) -> 8 configs
)
FIXED = dict(dropout=0.1, modules=ALL, accum=1, weight_decay=0.01,
             warmup_frac=0.05, max_grad_norm=1.0, use_dora=False)


def build_grid():
    keys = list(GRID_AXES)
    out = []
    for combo in itertools.product(*[GRID_AXES[k] for k in keys]):
        c = dict(FIXED); c.update(dict(zip(keys, combo)))
        c["alpha"] = c["r"] * 2
        pw = "bal" if c["pos_weight"] == "balanced" else "1"
        c["name"] = f"lr{c['lr']:.0e}_{c['schedule']}_pw{pw}_r{c['r']}"
        out.append(c)
    return out


def attach_lora(base, c):
    from peft import LoraConfig, get_peft_model, TaskType
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    kw = dict(task_type=TaskType.CAUSAL_LM, r=c["r"], lora_alpha=c["alpha"],
              lora_dropout=c["dropout"], target_modules=c["modules"])
    if c.get("use_dora"):
        kw["use_dora"] = True
    return get_peft_model(base, LoraConfig(**kw))


def stratified_holdout(samples, frac=0.15, seed=0):
    """Carve a stratified val split out of the training samples."""
    y = np.array([s["label_int"] for s in samples])
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.where(y == 1)[0]); neg = rng.permutation(np.where(y == 0)[0])
    nvp, nvn = int(len(pos) * frac), int(len(neg) * frac)
    val_idx = set(pos[:nvp]).union(neg[:nvn])
    tr = [s for i, s in enumerate(samples) if i not in val_idx]
    va = [s for i, s in enumerate(samples) if i in val_idx]
    return tr, va


def train_and_track(model, processor, train, val, test, c, yes_ids, no_ids, emit):
    """Train to MAX_EPOCHS, evaluating val + test after each epoch; emit one row/epoch."""
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup
    opt = AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    steps = (len(train) // max(c["accum"], 1)) * MAX_EPOCHS
    sched = (get_cosine_schedule_with_warmup(opt, int(c["warmup_frac"] * steps), steps)
             if c["schedule"] == "cosine" else None)

    pos_w = c["pos_weight"]
    if pos_w == "balanced":
        npos = sum(s["label_int"] for s in train)
        pos_w = (len(train) - npos) / max(npos, 1)
    accum = c["accum"]

    gt_val = np.array([s["label_int"] for s in val])
    gt_te = np.array([s["label_int"] for s in test])
    pv_te = np.array([s["pvim_prob"] for s in test], dtype=float)

    for epoch in range(MAX_EPOCHS):
        model.train()
        np.random.shuffle(train)
        opt.zero_grad()
        for i, s in enumerate(train):
            full, imgs = _encode(processor, s, with_answer=True)
            prompt, _ = _encode(processor, s, with_answer=False)
            inp = processor(text=[full], images=imgs, return_tensors="pt", padding=True).to(model.device)
            plen = processor(text=[prompt], images=imgs, return_tensors="pt", padding=True)["input_ids"].shape[1]
            labels = inp["input_ids"].clone(); labels[:, :plen] = -100; inp["labels"] = labels
            loss = model(**inp).loss
            if pos_w != 1.0 and s["label_int"] == 1:
                loss = loss * pos_w
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["max_grad_norm"])
                opt.step()
                if sched: sched.step()
                opt.zero_grad()

        # --- per-epoch evaluation ---
        val_sc = soft_scores(model, processor, val, yes_ids, no_ids)
        te_sc = soft_scores(model, processor, test, yes_ids, no_ids)
        val_auc = float(roc_auc_score(gt_val, val_sc)) if len(set(gt_val)) > 1 else float("nan")
        # val-tuned F1 threshold, applied to val (selection sanity)
        vf1 = max(metrics(gt_val, (val_sc >= t).astype(int))["f1"] for t in THRESHOLDS)
        # report test under the 20-seed protocol (VLM-only + fusion)
        vlm = run_protocol(gt_te, te_sc)
        fus = run_protocol(gt_te, te_sc, pvim=pv_te, fuse=True)
        sat0 = float((te_sc <= 0.001).mean())
        emit(c, epoch + 1, val_auc, vf1, vlm, fus, sat0)
        print(f"    [{c['name']}] epoch {epoch+1}: valAUC={val_auc:.3f} valF1={vf1:.3f} "
              f"| test VLM F1={vlm['f1']['f1'][0]:.3f} F2={vlm['f2']['f2'][0]:.3f} "
              f"fusF1={fus['f1']['f1'][0]:.3f} (sat {sat0*100:.0f}%)", flush=True)


FIELDS = ["name", "lr", "schedule", "pos_weight", "r", "epoch",
          "val_auroc", "val_f1", "test_vlm_F1", "test_vlm_F2", "test_vlm_Acc",
          "test_fus_F1", "test_fus_F2", "fus_beta", "test_sat0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    ap.add_argument("--model-dir", default="data/models/pie/PVIM/ckpt_tte60_120")
    ap.add_argument("--train-config", default="config_files/PVIM_tte60_120.yaml")   # 2-4s
    ap.add_argument("--test-config",  default="config_files/PVIM_eval_tte90.yaml")  # 3s
    ap.add_argument("--out", default="results/lora_search_6012_tte90.csv")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=999)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    grid = build_grid()
    shard = grid[args.start:min(args.end, len(grid))]

    # Resume: a config counts as done only if it has all MAX_EPOCHS rows. Partial
    # configs (killed by the wall) are redone cleanly - rewrite the file keeping
    # only complete configs so re-running doesn't duplicate epoch rows.
    from collections import Counter
    complete = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            existing = list(csv.DictReader(f))
        counts = Counter(r["name"] for r in existing)
        complete = {n for n, k in counts.items() if k >= MAX_EPOCHS}
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
            w.writerows([r for r in existing if r["name"] in complete])
    todo = [c for c in shard if c["name"] not in complete]
    print(f"=== LoRA 2-4s search: {len(grid)} configs total, shard[{args.start},{args.end}) "
          f"-> {len(shard)}, {len(todo)} to run ===", flush=True)
    if not todo:
        print("=== Nothing to do. ==="); return

    print(f"=== Building TRAIN ({args.train_config}) + TEST ({args.test_config}) samples ===", flush=True)
    train_all = build_training_samples(config_path=args.train_config, model_dir=args.model_dir,
                                       split="train", set_ids=None)
    test = build_training_samples(config_path=args.test_config, model_dir=args.model_dir,
                                  split="test", set_ids=("set03",))
    train, val = stratified_holdout(train_all, frac=args.val_frac, seed=0)
    print(f"  train={len(train)} val={len(val)} test={len(test)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def emit(c, epoch, val_auc, vf1, vlm, fus, sat0):
        r = dict(name=c["name"], lr=c["lr"], schedule=c["schedule"],
                 pos_weight=c["pos_weight"], r=c["r"], epoch=epoch,
                 val_auroc=f"{val_auc:.4f}", val_f1=f"{vf1:.4f}",
                 test_vlm_F1=f"{vlm['f1']['f1'][0]:.4f}", test_vlm_F2=f"{vlm['f2']['f2'][0]:.4f}",
                 test_vlm_Acc=f"{vlm['f1']['acc'][0]:.4f}",
                 test_fus_F1=f"{fus['f1']['f1'][0]:.4f}", test_fus_F2=f"{fus['f2']['f2'][0]:.4f}",
                 fus_beta=f"{fus['f1']['beta']:.2f}", test_sat0=f"{sat0:.3f}")
        with open(args.out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(r)

    for c in todo:
        print(f"\n########## CONFIG {c['name']}  {c} ##########", flush=True)
        base = load_base(args.qwen_model)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.qwen_model, use_fast=False)
        yes_ids, no_ids = yes_token_ids(processor)
        model = attach_lora(base, c)
        train_and_track(model, processor, list(train), val, test, c, yes_ids, no_ids, emit)
        del model, base, processor
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n=== Search shard done. Summary: {args.out} ===", flush=True)


if __name__ == "__main__":
    main()

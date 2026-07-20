"""Confirmation run of the 2-4s LoRA search winner: retrain, SAVE the adapter,
then evaluate fully so every metric is available (F1/F2/Acc/Prec/Rec, both
operating points, VLM-only + fusion, with +/-std under the 20-seed protocol).

Winner config (from results/lora_search_6012_tte90.csv):
    lr 1e-4 . cosine+warmup . class-balanced . rank 8 . 1 epoch
trained on the full [60,120] (2-4s) range, evaluated at TTE 90.

Outputs:
  * saved adapter    -> --adapter-out   (the reusable model)
  * per-sample scores-> --out           (sample_id, pvim_prob, qwen_score, gt)
  * full stats printed to the log (this is what fills the 'pending' table cells)

Run via run_lora_confirm_winner.job (GPU).
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
from lora_sweep import load_base, yes_token_ids, _encode, ALL
from prompt_ablation import soft_scores
from lora_validated import run_protocol

# The search winner.
WINNER = dict(name="opt_2-4s", lr=1e-4, schedule="cosine", pos_weight="balanced",
              r=8, alpha=16, dropout=0.1, modules=ALL, accum=1, weight_decay=0.01,
              warmup_frac=0.05, max_grad_norm=1.0, epochs=1)


def attach_lora(base, c):
    from peft import LoraConfig, get_peft_model, TaskType
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    lc = LoraConfig(task_type=TaskType.CAUSAL_LM, r=c["r"], lora_alpha=c["alpha"],
                    lora_dropout=c["dropout"], target_modules=c["modules"])
    return get_peft_model(base, lc)


def train_winner(model, processor, train, c):
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup
    opt = AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    steps = (len(train) // max(c["accum"], 1)) * c["epochs"]
    sched = (get_cosine_schedule_with_warmup(opt, int(c["warmup_frac"] * steps), steps)
             if c["schedule"] == "cosine" else None)
    pos_w = c["pos_weight"]
    if pos_w == "balanced":
        npos = sum(s["label_int"] for s in train)
        pos_w = (len(train) - npos) / max(npos, 1)
    accum = c["accum"]
    model.train()
    for epoch in range(c["epochs"]):
        np.random.shuffle(train)
        opt.zero_grad(); total = 0.0
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
            total += float(loss)
            if (i + 1) % 200 == 0:
                print(f"    epoch {epoch+1} step {i+1}/{len(train)} loss {total/(i+1):.4f}", flush=True)
        print(f"  epoch {epoch+1}: avg loss {total/len(train):.4f}", flush=True)
    return model


def report(tag, res):
    for opt in ("f1", "f2"):
        a = res[opt]
        b = f"  beta={a['beta']:.2f}" if a["beta"] is not None else ""
        print(f"  {tag:16s} [{opt.upper()}-opt]  F1={a['f1'][0]:.3f}+/-{a['f1'][1]:.3f}  "
              f"F2={a['f2'][0]:.3f}+/-{a['f2'][1]:.3f}  Acc={a['acc'][0]:.3f}  "
              f"Prec={a['prec'][0]:.3f}  Rec={a['rec'][0]:.3f}{b}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    ap.add_argument("--model-dir", default="data/models/pie/PVIM/ckpt_tte60_120")
    ap.add_argument("--train-config", default="config_files/PVIM_tte60_120.yaml")   # 2-4s
    ap.add_argument("--test-config",  default="config_files/PVIM_eval_tte90.yaml")  # 3s
    ap.add_argument("--adapter-out", default="data/models/lora_2-4s_optimized")
    ap.add_argument("--out", default="results/lora_opt_2-4s_eval90_set03.csv")
    args = ap.parse_args()

    print(f"=== Building TRAIN ({args.train_config}, full 2-4s) + TEST ({args.test_config}, set03) ===", flush=True)
    train = build_training_samples(config_path=args.train_config, model_dir=args.model_dir,
                                   split="train", set_ids=None)
    test = build_training_samples(config_path=args.test_config, model_dir=args.model_dir,
                                  split="test", set_ids=("set03",))
    print(f"  train={len(train)}  test={len(test)}", flush=True)
    gt = np.array([s["label_int"] for s in test])
    pvim = np.array([s["pvim_prob"] for s in test], dtype=float)

    print(f"\n=== Fine-tuning winner: {WINNER['name']} ({WINNER['schedule']}, lr {WINNER['lr']}, "
          f"balanced, r{WINNER['r']}, {WINNER['epochs']} ep) ===", flush=True)
    base = load_base(args.qwen_model)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.qwen_model, use_fast=False)
    yes_ids, no_ids = yes_token_ids(processor)
    model = attach_lora(base, WINNER)
    model = train_winner(model, processor, list(train), WINNER)

    os.makedirs(args.adapter_out, exist_ok=True)
    model.save_pretrained(args.adapter_out)
    print(f"\nSaved adapter -> {args.adapter_out}", flush=True)

    print("\n=== Scoring set03 @ TTE 90 ===", flush=True)
    scores = soft_scores(model, processor, test, yes_ids, no_ids)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "pvim_prob", "qwen_score",
                                          "ground_truth", "vlm_answer"])
        w.writeheader()
        for s, sc in zip(test, scores):
            w.writerow({"sample_id": s["sample_id"], "pvim_prob": f"{s['pvim_prob']:.4f}",
                        "qwen_score": f"{sc:.4f}", "ground_truth": s["label_int"],
                        "vlm_answer": "yes" if sc >= 0.5 else "no"})
    print(f"Saved per-sample scores -> {args.out}", flush=True)

    print("\n=== Full stats (20-seed val/test protocol, TTE 90) ===", flush=True)
    report("VLM-only", run_protocol(gt, scores))
    report("VLM+PVIM", run_protocol(gt, scores, pvim=pvim, fuse=True))
    print(f"\n=== Done. Model: {args.adapter_out}  Scores: {args.out} ===", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate the LoRA fine-tuned Qwen2.5-VL on the PIE test split (set03).

Loads base Qwen + the trained LoRA adapter, runs the same contact-sheet +
7-cue prompt used in training, parses yes/no, and reports F1/F2/accuracy/recall.
Also writes a thesis-schema CSV so the existing fusion / validation code can
read it like any other VLM variant.

Run (on a GPU node, same env as training):
    python future_work/eval_lora_qwen.py \
        --qwen-model "$QWEN_MODEL" \
        --adapter data/models/qwen_lora_pie \
        --out results/qwen25_32b_pvim_lora_tte90_set03.csv
"""
import os
import csv
import argparse

# Same offline / TF-on-CPU / torch shims as the training script (import order matters)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import torch
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda *a, **k: False

import numpy as np

from lora_finetune_qwen import build_training_samples


def parse_yes_no(text):
    """Parse a yes/no decision from the model output, scanning bottom-up."""
    t = text.strip().lower()
    for line in reversed(t.splitlines()):
        line = line.strip().strip(".:*- ")
        if line in ("yes", "no"):
            return 1 if line == "yes" else 0
        if line.endswith("yes") or "answer: yes" in line:
            return 1
        if line.endswith("no") or "answer: no" in line:
            return 0
    if "yes" in t and "no" not in t:
        return 1
    if "no" in t and "yes" not in t:
        return 0
    return 0  # default to no-cross if unparseable


def metrics(gt, pred):
    gt = np.asarray(gt); pred = np.asarray(pred)
    tp = int(((pred == 1) & (gt == 1)).sum()); fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum()); tn = int(((pred == 0) & (gt == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    f2 = 5 * prec * rec / max(4 * prec + rec, 1e-9)
    acc = (tp + tn) / max(len(gt), 1)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, f2=f2, tp=tp, fp=fp, fn=fn, tn=tn)


def best_fusion(gt, pvim, vlm):
    """Grid-search beta/threshold for fused = beta*pvim + (1-beta)*vlm (F1 and F2)."""
    betas = np.round(np.arange(0.0, 1.01, 0.05), 3)
    thrs = np.round(np.arange(0.05, 0.96, 0.025), 4)
    out = {}
    for key in ("f1", "f2"):
        best = {"score": -1.0}
        for b in betas:
            fused = b * pvim + (1 - b) * vlm
            for thr in thrs:
                m = metrics(gt, (fused >= thr).astype(int))
                if m[key] > best["score"]:
                    best = dict(score=m[key], beta=float(b), thr=float(thr), **m)
        out[key] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    ap.add_argument("--adapter", default="data/models/qwen_lora_pie")
    ap.add_argument("--model-dir", default="data/models/pie/PVIM/ckpt_tte60_120")
    ap.add_argument("--config", default="config_files/PVIM_eval_tte90.yaml")
    ap.add_argument("--out", default="results/qwen25_32b_pvim_lora_tte90_set03.csv")
    ap.add_argument("--no-adapter", action="store_true",
                    help="Evaluate the base model (zero-shot) for comparison.")
    args = ap.parse_args()

    print("=== Building test samples (set03) ===")
    samples = build_training_samples(
        config_path=args.config, model_dir=args.model_dir,
        split="test", set_ids=("set03",),
    )
    gt = [s["label_int"] for s in samples]
    print(f"  {len(samples)} samples ({sum(gt)} cross, {len(gt)-sum(gt)} no-cross)")

    print(f"\n=== Loading {'base (zero-shot)' if args.no_adapter else 'LoRA'} model ===")
    print(f"Loading VLM from: {args.qwen_model}")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model, device_map="auto", torch_dtype=torch.bfloat16,
    )
    if not args.no_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  attached adapter: {args.adapter}")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.qwen_model, use_fast=False)

    # soft yes/no token ids (continuous score = P(yes) / (P(yes)+P(no)))
    tok = processor.tokenizer

    def _first(w):
        ids = tok(w, add_special_tokens=False).input_ids
        return ids[0] if ids else None
    yes_ids = list({_first(w) for w in ["yes", "Yes", " yes", " Yes", "YES"]} - {None})
    no_ids = list({_first(w) for w in ["no", "No", " no", " No", "NO"]} - {None})

    print("\n=== Running inference (soft score) ===")
    rows, scores = [], []
    for i, s in enumerate(samples):
        messages = [
            {"role": "system", "content": s["system"]},
            {"role": "user", "content": [
                {"type": "image", "image": s["image"]},
                {"type": "text", "text": s["prompt"]},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        py = float(probs[yes_ids].sum()); pn = float(probs[no_ids].sum())
        p_yes = py / max(py + pn, 1e-9)
        scores.append(p_yes)
        rows.append({
            "sample_id": s["sample_id"],
            "pvim_prob": f"{s['pvim_prob']:.4f}",
            "qwen_score": f"{p_yes:.4f}",       # continuous P(yes) for threshold tuning
            "ground_truth": s["label_int"],
            "vlm_answer": "yes" if p_yes >= 0.5 else "no",
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)}")

    preds = (np.asarray(scores) >= 0.5).astype(int)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {args.out}")

    # --- metrics ---
    gt = np.asarray(gt); preds = np.asarray(preds)
    pvim = np.asarray([s["pvim_prob"] for s in samples])
    m = metrics(gt, preds)
    print("\n================ VLM-only (LoRA) ================")
    print(f"  F1={m['f1']:.3f}  F2={m['f2']:.3f}  Acc={m['acc']:.3f}  "
          f"Prec={m['prec']:.3f}  Rec={m['rec']:.3f}")
    print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

    fus = best_fusion(gt, pvim, np.asarray(scores))
    print("\n================ Fused with PVIM (grid-searched) ================")
    for key in ("f1", "f2"):
        b = fus[key]
        print(f"  {key.upper()}-opt: {key.upper()}={b['score']:.3f}  Acc={b['acc']:.3f}  "
              f"Prec={b['prec']:.3f}  Rec={b['rec']:.3f}  (beta={b['beta']:.2f}, thr={b['thr']:.3f})")


if __name__ == "__main__":
    main()

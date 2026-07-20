"""Overnight LoRA hyperparameter sweep for Qwen2.5-VL on PIE.

Trains + evaluates a grid of LoRA configs and writes one summary CSV. Builds the
train (set01/02/04) and test (set03) samples ONCE and reuses them across configs;
each config reloads the base model fresh, attaches a LoRA adapter, trains, and is
evaluated on the held-out test split.

For each config it reports, on set03:
  - binary yes/no at the model's own decision (argmax),
  - a soft P(yes) score with a threshold sweep (best standalone F1 and F2),
  - the grid-searched fusion-with-PVIM (best F1 and F2).

Run via run_lora_sweep.job (needs a GPU; ~45-75 min per config).
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
import numpy as np

import torch
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda *a, **k: False

from lora_finetune_qwen import build_training_samples
from eval_lora_qwen import metrics, best_fusion

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL = ATTN + ["gate_proj", "up_proj", "down_proj"]

# Grid: each varies one thing from the baseline, plus a couple of combos.
# pos_weight="balanced" -> n_neg/n_pos (upweights the crossing class).
BASE = dict(lr=2e-4, r=16, alpha=32, dropout=0.05, modules=ALL,
            epochs=3, pos_weight=1.0, accum=1)


def cfg(name, **kw):
    d = dict(BASE); d.update(kw); d["name"] = name
    return d


CONFIGS = [
    cfg("baseline"),
    cfg("lr1e-4", lr=1e-4),
    cfg("lr3e-4", lr=3e-4),
    cfg("r8",  r=8,  alpha=16),
    cfg("r32", r=32, alpha=64),
    cfg("dropout0.1", dropout=0.1),
    cfg("attn_only", modules=ATTN),
    cfg("classweight", pos_weight="balanced"),
    cfg("accum4", accum=4),
    cfg("reg_balanced", lr=1e-4, r=8, alpha=16, dropout=0.1, pos_weight="balanced"),
]


def load_base(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16,
    )


def attach_lora(base, c):
    from peft import LoraConfig, get_peft_model, TaskType
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    lc = LoraConfig(task_type=TaskType.CAUSAL_LM, r=c["r"], lora_alpha=c["alpha"],
                    lora_dropout=c["dropout"], target_modules=c["modules"])
    return get_peft_model(base, lc)


def _encode(processor, sample, with_answer):
    from qwen_vl_utils import process_vision_info
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": sample["prompt"]},
        ]},
    ]
    if with_answer:
        messages.append({"role": "assistant", "content": sample["label"]})
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=not with_answer)
    image_inputs, _ = process_vision_info(messages)
    return text, image_inputs


def train_cfg(model, processor, samples, c):
    from torch.optim import AdamW
    opt = AdamW(model.parameters(), lr=c["lr"], weight_decay=0.01)
    model.train()
    pos_w = c["pos_weight"]
    if pos_w == "balanced":
        npos = sum(s["label_int"] for s in samples)
        pos_w = (len(samples) - npos) / max(npos, 1)
    accum = c.get("accum", 1)

    for epoch in range(c["epochs"]):
        np.random.shuffle(samples)
        total = 0.0
        opt.zero_grad()
        for i, s in enumerate(samples):
            full_text, imgs = _encode(processor, s, with_answer=True)
            prompt_text, _ = _encode(processor, s, with_answer=False)
            inputs = processor(text=[full_text], images=imgs, return_tensors="pt",
                               padding=True).to(model.device)
            prompt_inputs = processor(text=[prompt_text], images=imgs,
                                      return_tensors="pt", padding=True)
            plen = prompt_inputs["input_ids"].shape[1]
            labels = inputs["input_ids"].clone()
            labels[:, :plen] = -100
            inputs["labels"] = labels

            loss = model(**inputs).loss
            if pos_w != 1.0 and s["label_int"] == 1:
                loss = loss * pos_w
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                opt.step(); opt.zero_grad()
            total += float(loss)
        print(f"    [{c['name']}] epoch {epoch+1}: avg loss {total/len(samples):.4f}", flush=True)
    return model


def yes_token_ids(processor):
    tok = processor.tokenizer

    def first(w):
        ids = tok(w, add_special_tokens=False).input_ids
        return ids[0] if ids else None
    yes = {first(w) for w in ["yes", "Yes", " yes", " Yes", "YES"]} - {None}
    no = {first(w) for w in ["no", "No", " no", " No", "NO"]} - {None}
    return list(yes), list(no)


@torch.no_grad()
def eval_cfg(model, processor, test_samples, yes_ids, no_ids):
    model.eval()
    gt = np.array([s["label_int"] for s in test_samples])
    pvim = np.array([s["pvim_prob"] for s in test_samples])
    p_yes = np.zeros(len(test_samples))
    for i, s in enumerate(test_samples):
        text, imgs = _encode(processor, s, with_answer=False)
        inputs = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        py = float(probs[yes_ids].sum())
        pn = float(probs[no_ids].sum())
        p_yes[i] = py / max(py + pn, 1e-9)

    hard = (p_yes >= 0.5).astype(int)
    m = metrics(gt, hard)

    thrs = np.round(np.arange(0.05, 0.96, 0.025), 4)
    bestF1 = max((dict(metrics(gt, (p_yes >= t).astype(int)), thr=float(t)) for t in thrs),
                 key=lambda d: d["f1"])
    bestF2 = max((dict(metrics(gt, (p_yes >= t).astype(int)), thr=float(t)) for t in thrs),
                 key=lambda d: d["f2"])
    fus = best_fusion(gt, pvim, p_yes)
    return m, bestF1, bestF2, fus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    ap.add_argument("--model-dir", default="data/models/pie/PVIM/ckpt_tte60_120")
    ap.add_argument("--config", default="config_files/PVIM_eval_tte90.yaml")
    ap.add_argument("--train-sets", default="")  # empty = native train set01,02,04
    ap.add_argument("--out", default="results/lora_sweep_tte90.csv")
    ap.add_argument("--save-adapters", action="store_true",
                    help="Save each adapter under data/models/sweep/<name>.")
    args = ap.parse_args()

    set_ids = tuple(s.strip() for s in args.train_sets.split(",") if s.strip()) or None

    print("=== Building TRAIN samples (once) ===", flush=True)
    train_samples = build_training_samples(config_path=args.config, model_dir=args.model_dir,
                                           split="train", set_ids=set_ids)
    ntr = len(train_samples); ntrpos = sum(s["label_int"] for s in train_samples)
    print(f"  train: {ntr} ({ntrpos} cross, {ntr-ntrpos} no-cross)", flush=True)

    print("=== Building TEST samples (once, set03) ===", flush=True)
    test_samples = build_training_samples(config_path=args.config, model_dir=args.model_dir,
                                          split="test", set_ids=("set03",))
    nte = len(test_samples); ntepos = sum(s["label_int"] for s in test_samples)
    print(f"  test: {nte} ({ntepos} cross, {nte-ntepos} no-cross)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["name", "lr", "r", "alpha", "dropout", "modules", "epochs", "pos_weight", "accum",
              "F1@0.5", "Acc@0.5", "Prec@0.5", "Rec@0.5",
              "bestF1", "bestF1_thr", "bestF2", "bestF2_thr",
              "fusF1", "fusF1_beta", "fusF2", "fusF2_beta"]
    with open(args.out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for c in CONFIGS:
        print(f"\n########## CONFIG: {c['name']} ##########", flush=True)
        print(f"  {c}", flush=True)
        base = load_base(args.qwen_model)
        processor = __import__("transformers").AutoProcessor.from_pretrained(
            args.qwen_model, use_fast=False)
        yes_ids, no_ids = yes_token_ids(processor)
        model = attach_lora(base, c)
        model = train_cfg(model, processor, list(train_samples), c)
        if args.save_adapters:
            d = f"data/models/sweep/{c['name']}"
            os.makedirs(d, exist_ok=True); model.save_pretrained(d)
        m, b1, b2, fus = eval_cfg(model, processor, test_samples, yes_ids, no_ids)

        row = dict(name=c["name"], lr=c["lr"], r=c["r"], alpha=c["alpha"], dropout=c["dropout"],
                   modules="attn" if c["modules"] == ATTN else "all", epochs=c["epochs"],
                   pos_weight=c["pos_weight"], accum=c["accum"],
                   **{"F1@0.5": f"{m['f1']:.3f}", "Acc@0.5": f"{m['acc']:.3f}",
                      "Prec@0.5": f"{m['prec']:.3f}", "Rec@0.5": f"{m['rec']:.3f}",
                      "bestF1": f"{b1['f1']:.3f}", "bestF1_thr": f"{b1['thr']:.3f}",
                      "bestF2": f"{b2['f2']:.3f}", "bestF2_thr": f"{b2['thr']:.3f}",
                      "fusF1": f"{fus['f1']['score']:.3f}", "fusF1_beta": f"{fus['f1']['beta']:.2f}",
                      "fusF2": f"{fus['f2']['score']:.3f}", "fusF2_beta": f"{fus['f2']['beta']:.2f}"})
        with open(args.out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"  RESULT {c['name']}: F1@0.5={row['F1@0.5']} bestF1={row['bestF1']} "
              f"bestF2={row['bestF2']} fusF1={row['fusF1']} fusF2={row['fusF2']}", flush=True)

        del model, base, processor
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n=== Sweep done. Summary: {args.out} ===", flush=True)


if __name__ == "__main__":
    main()

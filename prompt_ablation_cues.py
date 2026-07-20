"""Per-cue leave-one-out ablation for the LoRA fine-tuned Qwen2.5-VL.

The Standard prompt's checklist has 7 numbered reasoning cues:
  1 orientation  2 gaze  3 movement  4 posture  5 context  6 speed  7 pvim
The coarse ablation (prompt_ablation.py) only removes the whole checklist at once
(plus speed and pvim). This script removes EACH cue individually, so you can read
off which of the 7 actually carries signal.

Variants (8): the full 7-cue prompt, then drop one cue at a time. For each it
LoRA fine-tunes with that prompt, runs soft-score inference on the held-out test
split (set03), and reports BOTH VLM-only and Fusion-with-PVIM under the same
20-seed val/test protocol as the thesis (F1- and F2-optimal thresholds).

The bbox image marker is held ON for every variant (it is the visual marker, not
a checklist line). Sheets + speed text are cached once and reused. LoRA
hyperparameters = the sweep winner (reg_balanced), identical to prompt_ablation.py.

Run via run_prompt_ablation_cues.job (GPU; ~45-60 min per variant, ~7 h total).
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

# Reuse the heavy lifting from the coarse ablation.
from prompt_ablation import load_raw, soft_scores, LORA_CFG
from lora_sweep import load_base, attach_lora, train_cfg, yes_token_ids
from lora_validated import run_protocol
from stratified_run_qwen_speed import ROLE_TEMPLATE

CUE_ORDER = ["orientation", "gaze", "movement", "posture", "context", "speed", "pvim"]

BEHAVIOR_TEXT = {
    "orientation": "BODY ORIENTATION: Is the pedestrian facing towards the road, away from it, or sideways?",
    "gaze":        "GAZE: Are they looking towards the road or traffic?",
    "movement":    "MOVEMENT: Are they stepping towards the curb, standing still, or moving parallel to the road?",
    "posture":     "POSTURE: Do they appear ready to step off the curb (leaning forward, weight shifted)?",
    "context":     "CONTEXT: Are they near a crosswalk or intersection? Is there a gap in traffic?",
}

# (variant name, set of cues dropped).  empty set = full 7-cue baseline.
VARIANTS = ([("full7", frozenset())] +
            [(f"no_{c}", frozenset({c})) for c in CUE_ORDER] +
            [("no_pvim_no_gaze", frozenset({"pvim", "gaze"}))])   # combined-drop follow-up


def build_prompt(dropped, pvim_prob, speed_text, with_bbox=True):
    """Full 7-cue prompt with the `dropped` cues removed, renumbered 1..k."""
    marker = "marked with the red bounding box " if with_bbox else ""
    items = []
    for key in ["orientation", "gaze", "movement", "posture", "context"]:
        if key not in dropped:
            items.append(BEHAVIOR_TEXT[key])
    if "speed" not in dropped:
        items.append(f"EGO-VEHICLE DYNAMICS: {speed_text}")
    if "pvim" not in dropped:
        interp = "This is a high probability." if pvim_prob >= 0.5 else "This is a low probability."
        items.append("PVIM MODEL: A trained pedestrian-vehicle interaction model estimates "
                     f"the crossing probability as {pvim_prob:.2f}/1.00. {interp}")
    L = [f"Analyse the pedestrian {marker}in the image sequence carefully. "
         "Before answering, reason through each of the following cues:"]
    for n, text in enumerate(items, 1):
        L.append(f"{n}. {text}")
    L.append("Consider ALL of the above before deciding. A pedestrian can intend to cross even if "
             "currently walking parallel to the road. Answer with exactly one word on the final line: yes or no.")
    return "\n".join(L)


def assemble(raw, dropped):
    """Build train/test samples for the dropped cue set (bbox marker always on)."""
    out = []
    for i in range(len(raw["gt"])):
        prompt = build_prompt(dropped, raw["pvim"][i], raw["speed"][i], with_bbox=True)
        out.append(dict(image=raw["bbox"][i], system=ROLE_TEMPLATE, prompt=prompt,
                        label="yes" if raw["gt"][i] == 1 else "no", label_int=int(raw["gt"][i])))
    return out


def completed_variants(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            done.add(row.get("variant"))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    ap.add_argument("--model-dir", default="data/models/pie/PVIM/ckpt_tte60_120")
    ap.add_argument("--config", default="config_files/PVIM_eval_tte90.yaml")
    ap.add_argument("--train-sets", default="")   # empty = native train set01/02/04
    ap.add_argument("--out", default="results/prompt_ablation_cues_tte90.csv")
    args = ap.parse_args()
    set_ids = tuple(s.strip() for s in args.train_sets.split(",") if s.strip()) or None

    done = completed_variants(args.out)
    todo = [(n, d) for (n, d) in VARIANTS if n not in done]
    print(f"=== Per-cue ablation: {len(VARIANTS)} variants, {len(done)} done, {len(todo)} to run ===", flush=True)
    if not todo:
        print("=== Nothing to do. ===", flush=True)
        return

    print("=== Loading + caching TRAIN sheets (set01/02/04) ===", flush=True)
    train_raw = load_raw(args.config, args.model_dir, "train", set_ids)
    print("=== Loading + caching TEST sheets (set03) ===", flush=True)
    test_raw = load_raw(args.config, args.model_dir, "test", ("set03",))
    gt = test_raw["gt"]; pvim = test_raw["pvim"]

    fields = ["variant", "dropped", "method", "opt", "F1", "F2", "Accuracy", "Precision", "Recall", "beta"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def emit(variant, dropped, method, res):
        dropped_str = "+".join(c for c in CUE_ORDER if c in dropped) or "--"
        rows = []
        for opt in ("f1", "f2"):
            a = res[opt]
            rows.append(dict(variant=variant, dropped=dropped_str, method=method, opt=opt.upper(),
                             F1=f"{a['f1'][0]:.3f}+/-{a['f1'][1]:.3f}", F2=f"{a['f2'][0]:.3f}+/-{a['f2'][1]:.3f}",
                             Accuracy=f"{a['acc'][0]:.3f}", Precision=f"{a['prec'][0]:.3f}",
                             Recall=f"{a['rec'][0]:.3f}", beta=f"{a['beta']:.2f}" if a['beta'] is not None else "--"))
        with open(args.out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(rows)

    for (name, dropped) in todo:
        shown = "+".join(c for c in CUE_ORDER if c in dropped) or "nothing"
        print(f"\n########## VARIANT: {name}  (dropped: {shown}) ##########", flush=True)
        tr = assemble(train_raw, dropped)
        te = assemble(test_raw, dropped)

        base = load_base(args.qwen_model)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.qwen_model, use_fast=False)
        yes_ids, no_ids = yes_token_ids(processor)
        model = attach_lora(base, LORA_CFG)
        model = train_cfg(model, processor, list(tr), LORA_CFG)
        scores = soft_scores(model, processor, te, yes_ids, no_ids)

        vlm = run_protocol(gt, scores)
        fus = run_protocol(gt, scores, pvim=pvim, fuse=True)
        emit(name, dropped, "VLM-only", vlm)
        emit(name, dropped, "Fusion", fus)
        print(f"  {name}: VLM-only F1={vlm['f1']['f1'][0]:.3f} F2={vlm['f2']['f2'][0]:.3f} | "
              f"Fusion F1={fus['f1']['f1'][0]:.3f} F2={fus['f2']['f2'][0]:.3f} (beta {fus['f1']['beta']:.2f})", flush=True)

        del model, base, processor
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n=== Per-cue ablation done. Summary: {args.out} ===", flush=True)


if __name__ == "__main__":
    main()

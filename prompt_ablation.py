"""Prompt-input ablation for the LoRA fine-tuned Qwen2.5-VL.

For each prompt variant (dropping one component of the Standard prompt) it:
  1. LoRA fine-tunes on the PIE train split with that prompt,
  2. runs soft-score inference on the held-out test split (set03),
  3. reports BOTH VLM-only and Fusion-with-PVIM, under the same 20-seed
     val/test protocol as the thesis, for F1- and F2-optimal thresholds.

Frames are loaded once (bbox + no-bbox contact sheets cached); only the prompt
text / chosen sheet change per variant, so the expensive IO is not repeated.

Variants: full, -pvim (semantic-only), -speed, -bbox, -checklist, minimal.
LoRA hyperparameters = the sweep winner (reg_balanced).

Run via run_prompt_ablation.job (GPU; ~45-60 min per variant).
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

from lora_finetune_qwen import load_pvim_data_split
from lora_sweep import load_base, attach_lora, train_cfg, yes_token_ids, ALL
from lora_validated import run_protocol
from stratified_run_qwen_speed import (
    make_contact_sheet_with_bbox, build_speed_text, load_pie_lookups, extract_ped_id,
    ROLE_TEMPLATE,
)
from stratified_run_qwen import make_contact_sheet   # no-bbox version

LORA_CFG = dict(name="regbal", lr=1e-4, r=8, alpha=16, dropout=0.1,
                modules=ALL, epochs=3, pos_weight="balanced", accum=1)

# (name, with_pvim, with_speed, with_bbox, with_checklist)
VARIANTS = [
    ("full",         True,  True,  True,  True),
    ("no_pvim",      False, True,  True,  True),   # semantic-only (the key one)
    ("no_speed",     True,  False, True,  True),
    ("no_bbox",      True,  True,  False, True),
    ("no_checklist", True,  True,  True,  False),
    ("minimal",      False, False, False, False),
]


def build_prompt(with_pvim, with_speed, with_bbox, with_checklist, pvim_prob, speed_text):
    marker = "marked with the red bounding box " if with_bbox else ""
    if with_checklist:
        L = [f"Analyse the pedestrian {marker}in the image sequence carefully. "
             "Before answering, reason through each of the following cues:",
             "1. BODY ORIENTATION: Is the pedestrian facing towards the road, away from it, or sideways?",
             "2. GAZE: Are they looking towards the road or traffic?",
             "3. MOVEMENT: Are they stepping towards the curb, standing still, or moving parallel to the road?",
             "4. POSTURE: Do they appear ready to step off the curb (leaning forward, weight shifted)?",
             "5. CONTEXT: Are they near a crosswalk or intersection? Is there a gap in traffic?"]
        n = 6
        if with_speed:
            L.append(f"{n}. EGO-VEHICLE DYNAMICS: {speed_text}"); n += 1
        if with_pvim:
            interp = "This is a high probability." if pvim_prob >= 0.5 else "This is a low probability."
            L.append(f"{n}. PVIM MODEL: A trained pedestrian-vehicle interaction model estimates "
                     f"the crossing probability as {pvim_prob:.2f}/1.00. {interp}")
        L.append("Consider ALL of the above before deciding. A pedestrian can intend to cross even if "
                 "currently walking parallel to the road. Answer with exactly one word on the final line: yes or no.")
        return "\n".join(L)
    extra = ""
    if with_speed:
        extra += f" Ego-vehicle dynamics: {speed_text}"
    if with_pvim:
        interp = "high" if pvim_prob >= 0.5 else "low"
        extra += (f" A pedestrian-vehicle interaction model estimates the crossing probability "
                  f"as {pvim_prob:.2f}/1.00 ({interp}).")
    return (f"Analyse the pedestrian {marker}in the image sequence. Will this pedestrian cross the road "
            f"in the near future?{extra} Answer with exactly one word on the final line: yes or no.")


def load_raw(config, model_dir, split, set_ids):
    """Return gt, pvim, and cached (bbox / no-bbox) sheets + speed text per sample."""
    from lora_finetune_qwen import build_training_samples  # noqa (ensures shim order)
    from stratified_run_qwen_speed import load_pvim_model, run_pvim_predictions
    td, _ = load_pvim_data_split(config, split=split, set_ids=set_ids)
    pvim = run_pvim_predictions(load_pvim_model(model_dir), td)
    gt = np.asarray(td["data"][1]).flatten().astype(int)
    lookups = load_pie_lookups()
    bbox_sheets, plain_sheets, speeds, frame_paths = [], [], [], []
    for i in range(len(gt)):
        paths = td["image"][i]; ped = extract_ped_id(td["ped_id"][i])
        bbox_sheets.append(make_contact_sheet_with_bbox(paths, ped, lookups, n_frames=8, size=336)[0])
        plain_sheets.append(make_contact_sheet(paths, n_frames=8, size=336))
        speeds.append(build_speed_text(paths, lookups)[0])
        frame_paths.append(list(paths))
        if (i + 1) % 100 == 0:
            print(f"    sheets {split} {i+1}/{len(gt)}", flush=True)
    return dict(gt=gt, pvim=np.asarray(pvim, dtype=float),
                bbox=bbox_sheets, plain=plain_sheets, speed=speeds, paths=frame_paths)


def assemble(raw, wp, ws, wb, wc):
    out = []
    for i in range(len(raw["gt"])):
        sheet = raw["bbox"][i] if wb else raw["plain"][i]
        prompt = build_prompt(wp, ws, wb, wc, raw["pvim"][i], raw["speed"][i])
        out.append(dict(image=sheet, system=ROLE_TEMPLATE, prompt=prompt,
                        label="yes" if raw["gt"][i] == 1 else "no", label_int=int(raw["gt"][i])))
    return out


@torch.no_grad()
def soft_scores(model, processor, samples, yes_ids, no_ids):
    from lora_sweep import _encode
    model.eval()
    ps = np.zeros(len(samples))
    for i, s in enumerate(samples):
        text, imgs = _encode(processor, s, with_answer=False)
        inp = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
        probs = torch.softmax(model(**inp).logits[0, -1].float(), dim=-1)
        py = float(probs[yes_ids].sum()); pn = float(probs[no_ids].sum())
        ps[i] = py / max(py + pn, 1e-9)
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))
    ap.add_argument("--model-dir", default="data/models/pie/PVIM/ckpt_tte60_120")
    ap.add_argument("--config", default="config_files/PVIM_eval_tte90.yaml")
    ap.add_argument("--train-sets", default="")   # empty = native train set01/02/04
    ap.add_argument("--out", default="results/prompt_ablation_tte90.csv")
    args = ap.parse_args()
    set_ids = tuple(s.strip() for s in args.train_sets.split(",") if s.strip()) or None

    print("=== Loading + caching TRAIN sheets (set01/02/04) ===", flush=True)
    train_raw = load_raw(args.config, args.model_dir, "train", set_ids)
    print("=== Loading + caching TEST sheets (set03) ===", flush=True)
    test_raw = load_raw(args.config, args.model_dir, "test", ("set03",))
    gt = test_raw["gt"]; pvim = test_raw["pvim"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["variant", "method", "opt", "F1", "F2", "Accuracy", "Precision", "Recall", "beta"]
    with open(args.out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def emit(variant, method, res):
        rows = []
        for opt in ("f1", "f2"):
            a = res[opt]
            rows.append(dict(variant=variant, method=method, opt=opt.upper(),
                             F1=f"{a['f1'][0]:.3f}+/-{a['f1'][1]:.3f}", F2=f"{a['f2'][0]:.3f}+/-{a['f2'][1]:.3f}",
                             Accuracy=f"{a['acc'][0]:.3f}", Precision=f"{a['prec'][0]:.3f}",
                             Recall=f"{a['rec'][0]:.3f}", beta=f"{a['beta']:.2f}" if a['beta'] is not None else "--"))
        with open(args.out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(rows)

    for (name, wp, ws, wb, wc) in VARIANTS:
        print(f"\n########## VARIANT: {name}  (pvim={wp} speed={ws} bbox={wb} checklist={wc}) ##########", flush=True)
        tr = assemble(train_raw, wp, ws, wb, wc)
        te = assemble(test_raw, wp, ws, wb, wc)

        base = load_base(args.qwen_model)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.qwen_model, use_fast=False)
        yes_ids, no_ids = yes_token_ids(processor)
        model = attach_lora(base, LORA_CFG)
        model = train_cfg(model, processor, list(tr), LORA_CFG)
        scores = soft_scores(model, processor, te, yes_ids, no_ids)

        vlm = run_protocol(gt, scores)
        fus = run_protocol(gt, scores, pvim=pvim, fuse=True)
        emit(name, "VLM-only", vlm)
        emit(name, "Fusion", fus)
        print(f"  {name}: VLM-only F1={vlm['f1']['f1'][0]:.3f} F2={vlm['f2']['f2'][0]:.3f} | "
              f"Fusion F1={fus['f1']['f1'][0]:.3f} F2={fus['f2']['f2'][0]:.3f} (beta {fus['f1']['beta']:.2f})", flush=True)

        del model, base, processor
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n=== Prompt ablation done. Summary: {args.out} ===", flush=True)


if __name__ == "__main__":
    main()

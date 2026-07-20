"""stratified_run_qwen.py — Run Qwen3-VL+PVIM on a balanced subset of the test set.

Picks `n_per_class` samples from each class (crossing / not-crossing) so the
evaluation is meaningful rather than just predicting 'no' for everything.

Usage:
    python stratified_run_qwen.py --model-dir data/models/pie/PVIM/21Apr2026-00h53m48s
    python stratified_run_qwen.py --model-dir data/models/pie/PVIM/21Apr2026-00h53m48s --n-per-class 50
"""

import argparse
import csv
import gc
import os
import re
import sys
import yaml

import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PATH_PIE           = "PIE_master"
DEFAULT_CONFIG     = os.path.join("config_files", "PVIM.yaml")
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"


# ── Prompt templates ──────────────────────────────────────────────────────────

ROLE_TEMPLATE = (
    "You are an autonomous vehicle safety system. "
    "Your task is to predict whether a pedestrian intends to cross the road "
    "in the near future, based on visual evidence from a front-facing camera."
)

PHYSICAL_CUES_TEMPLATE = (
    "Analyse the pedestrian marked in the image sequence carefully. "
    "Before answering, reason through each of the following cues:\n"
    "1. BODY ORIENTATION: Is the pedestrian facing towards the road, away from it, or sideways?\n"
    "2. GAZE: Are they looking towards the road or traffic?\n"
    "3. MOVEMENT: Are they stepping towards the curb, standing still, or moving parallel to the road?\n"
    "4. POSTURE: Do they appear ready to step off the curb (leaning forward, weight shifted)?\n"
    "5. CONTEXT: Are they near a crosswalk or intersection? Is there a gap in traffic?\n"
    "6. PVIM MODEL: A trained pedestrian-vehicle interaction model estimates "
    "the crossing probability as {pvim_prob:.2f}/1.00. {pvim_interp}\n\n"
    "Consider ALL of the above before deciding. "
    "A pedestrian can intend to cross even if currently walking parallel to the road. "
    "Answer with exactly one word on the final line: yes or no."
)


# ── PVIM helpers ──────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def configure_tensorflow_memory_growth():
    """Ask TensorFlow not to reserve the whole GPU before Qwen loads."""
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"WARNING: could not set TensorFlow memory growth for {gpu.name}: {e}")


def load_all_pvim_data(config_path):
    from action_predict import action_prediction
    from pie_data import PIE

    configs = load_yaml(os.path.join("config_files", "configs_default.yaml"))
    model_configs = load_yaml(config_path)
    for s in ["model_opts", "net_opts", "data_opts"]:
        if s in model_configs:
            configs[s].update(model_configs[s])

    obs = configs["model_opts"].get("obs_length", 16)
    tte = configs["model_opts"].get("time_to_event", 30)
    configs["data_opts"]["min_track_size"] = obs + (tte if isinstance(tte, int) else tte[1])
    configs["model_opts"]["generator"] = False

    imdb = PIE(data_path=PATH_PIE, images_path=configs["data_opts"].get("images_path"))
    raw  = imdb.generate_data_trajectory_sequence("test", **configs["data_opts"])
    mc   = action_prediction(configs["model_opts"]["model"])(**configs["net_opts"])
    td   = mc.get_data("test", raw, {**configs["model_opts"], "batch_size": 1})
    return td, configs


def load_pvim_model(model_dir):
    model_path = os.path.join(model_dir, "model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.h5 not found in {model_dir}")
    configure_tensorflow_memory_growth()
    from tensorflow.keras.models import load_model
    return load_model(model_path)


def run_pvim_predictions(pvim_model, test_data, batch_size=248):
    """Predict in small batches to avoid OOM on large test sets."""
    model_data = test_data["data"]
    data_items = model_data[0] if isinstance(model_data, tuple) else model_data

    if not isinstance(data_items, list):
        preds = pvim_model.predict(data_items, batch_size=batch_size, verbose=1)
        return np.asarray(preds).flatten()

    n_samples = len(data_items[0])
    all_preds = []
    for start in range(0, n_samples, batch_size):
        end   = min(start + batch_size, n_samples)
        batch = [x[start:end] for x in data_items]
        pred  = pvim_model.predict(batch, verbose=0)
        all_preds.append(np.asarray(pred).flatten())
        print("  PVIM batch {}-{}/{}".format(start, end, n_samples), end="\r")
    print()
    return np.concatenate(all_preds)


def extract_ped_id(ped_id_entry):
    try:
        return str(np.asarray(ped_id_entry).flat[0])
    except Exception:
        return "unknown"


# ── Image helpers ─────────────────────────────────────────────────────────────

def make_contact_sheet(image_paths, n_frames=8, size=336):
    """Create a contact sheet of n_frames frames in temporal order.

    Panels are filled left-to-right, then top-to-bottom.
    """
    n_frames = min(n_frames, len(image_paths))
    indices  = np.linspace(0, len(image_paths) - 1, n_frames, dtype=int)

    frames = [
        Image.open(image_paths[i]).convert("RGB").resize((size, size), Image.LANCZOS)
        for i in indices
    ]

    cols  = min(n_frames, 4)
    rows  = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (size * cols, size * rows), (0, 0, 0))

    for idx, frame in enumerate(frames):
        x = (idx % cols) * size
        y = (idx // cols) * size
        sheet.paste(frame, (x, y))

    return sheet


# ── Prompt helpers ────────────────────────────────────────────────────────────

def interpret_pvim_probability(pvim_prob: float) -> str:
    """Turn a raw PVIM probability into a short natural-language cue."""
    if pvim_prob >= 0.80:
        level = "a very high"
    elif pvim_prob >= 0.60:
        level = "a high"
    elif pvim_prob >= 0.40:
        level = "a moderate"
    elif pvim_prob >= 0.20:
        level = "a low"
    else:
        level = "a very low"

    return (
        f"This indicates {level} model-estimated probability that the pedestrian "
        "will cross the road soon."
    )


def build_prompt(pvim_prob: float) -> str:
    """
    Build the multimodal prompt that:
      - Sets the role (autonomous vehicle safety system)
      - Describes the temporal grid of frames
      - Asks the model to reason through specific physical cues
      - Injects the PVIM probability + a textual interpretation
      - Requires a final one-word answer on the last line: yes or no
    """
    pvim_interp   = interpret_pvim_probability(pvim_prob)
    image_context = (
        "The image is a grid of frames from the same front-facing camera video. "
        "Panels are ordered left-to-right, top-to-bottom in time, from earliest to latest.\n\n"
        "Focus only on the pedestrian indicated by the bounding box (if visible) "
        "or the most clearly relevant pedestrian in the scene.\n\n"
    )

    return (
        ROLE_TEMPLATE
        + "\n\n"
        + image_context
        + PHYSICAL_CUES_TEMPLATE.format(pvim_prob=pvim_prob, pvim_interp=pvim_interp)
    )


# ── Qwen3-VL ──────────────────────────────────────────────────────────────────

def load_qwen(model_name=DEFAULT_QWEN_MODEL):
    # Inject qwen_packages AFTER TF is already in sys.modules, so TF's numpy
    # is never replaced by a potentially incompatible version from qwen_packages.
    pkgdir = os.environ.get("QWEN_PACKAGES", "")
    if pkgdir and pkgdir not in sys.path:
        sys.path.insert(0, pkgdir)

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        import torch
    except ImportError as e:
        sys.exit(
            "transformers/torch import failed: {}\n"
            "Need: transformers>=4.49 (for Qwen2.5-VL) + torch + qwen_vl_utils".format(e)
        )

    print(f"Loading Qwen2.5-VL: {model_name} ...")
    sys.stdout.flush()

    # use_fast=False: the fast image processor in transformers 4.54+ calls
    # torch.compiler.is_compiling(), which only exists in torch 2.3+. The
    # Snellius PyTorch module is 2.1.2, so we use the slow path.
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Qwen2.5-VL loaded on device(s): {set(str(p.device) for p in model.parameters())}")
    return processor, model


def query_qwen(processor, model, sheet, prompt_text):
    """Ask Qwen2.5-VL for an explanation + final yes/no decision."""
    from qwen_vl_utils import process_vision_info
    import torch

    messages = [
        # System message carries the high-level role
        {"role": "system", "content": ROLE_TEMPLATE},
        # User message: image + detailed cue-based instructions (incl. PVIM)
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sheet},
                {"type": "text",  "text": prompt_text},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,   # 6-cue chain needs ~200-400 tokens before final answer
            do_sample=False,      # deterministic (greedy)
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # Free CUDA memory between samples — prevents KV-cache fragmentation slowdown
    del generated_ids, inputs, trimmed
    torch.cuda.empty_cache()
    return output[0].strip()


def parse_decision(raw):
    """Parse yes/no from Qwen's response, scanning bottom-up.

    Returns ("yes"|"no", parsed_ok). parsed_ok is False when the response
    contained no clear yes/no token and we had to fall back to the default.
    """
    lines = [l.strip().lower().rstrip(".") for l in raw.splitlines() if l.strip()]
    for line in reversed(lines):
        if re.search(r"\byes\b", line) and not re.search(r"\bno\b", line):
            return "yes", True
        if re.search(r"\bno\b", line) and not re.search(r"\byes\b", line):
            return "no", True

    text = raw.lower()
    yes_hit = bool(re.search(r"\byes\b", text))
    no_hit  = bool(re.search(r"\bno\b", text))
    if yes_hit and not no_hit:
        return "yes", True
    if no_hit and not yes_hit:
        return "no", True

    return "no", False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",        default="saved_models/pvim")
    parser.add_argument("--config",           default=DEFAULT_CONFIG)
    parser.add_argument("--n-per-class",      type=int,   default=100,
                        help="Samples per class (crossing / not-crossing). Default 100+100=200 total.")
    parser.add_argument("--n-frames",         type=int,   default=4)
    parser.add_argument("--output",           default="results/qwen_pvim_stratified.csv")
    parser.add_argument("--qwen-model",       default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--fusion-beta",      type=float, default=0.5,
                        help="Weight for PVIM in fused_score = beta*pvim + (1-beta)*qwen_binary.")
    parser.add_argument("--fusion-threshold", type=float, default=0.5,
                        help="Threshold on fused_score for final yes/no decision.")
    args = parser.parse_args()

    configure_tensorflow_memory_growth()

    # ── 1. Load all PVIM data ─────────────────────────────────────────────────
    print("\n=== Loading PVIM data ===")
    test_data, _ = load_all_pvim_data(args.config)

    labels = np.asarray(test_data["data"][1]).flatten()
    print(f"  Total   : {len(labels)}")
    print(f"  Cross   : {int((labels==1).sum())}")
    print(f"  No-cross: {int((labels==0).sum())}")

    # ── 2. Stratified sampling ────────────────────────────────────────────────
    rng    = np.random.default_rng(args.seed)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    n_pos   = min(args.n_per_class, len(idx_pos))
    n_neg   = min(args.n_per_class, len(idx_neg))
    chosen  = np.concatenate([
        rng.choice(idx_pos, n_pos, replace=False),
        rng.choice(idx_neg, n_neg, replace=False),
    ])
    rng.shuffle(chosen)
    if n_pos != n_neg:
        print(f"  WARNING: unequal class sizes (cross={n_pos}, no-cross={n_neg}) — results will be imbalanced")
    print(f"\n  Stratified subset: {n_pos} crossing + {n_neg} not-crossing = {len(chosen)} total")

    # ── 3. Run PVIM on full set ───────────────────────────────────────────────
    print("\n=== Running PVIM inference (full set) ===")
    pvim_model  = load_pvim_model(args.model_dir)
    predictions = run_pvim_predictions(pvim_model, test_data)
    del pvim_model
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"WARNING: could not clear TensorFlow session before Qwen load: {e}")
    gc.collect()

    # ── 4. Load Qwen3-VL ──────────────────────────────────────────────────────
    print("\n=== Loading Qwen3-VL ===")
    processor, model = load_qwen(args.qwen_model)

    # ── 5. Inference loop ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows = []
    print(f"\n=== Running Qwen3-VL on {len(chosen)} samples ===\n")

    beta      = args.fusion_beta
    threshold = args.fusion_threshold

    for run_idx, sample_idx in enumerate(chosen):
        sample_idx  = int(sample_idx)
        pvim_prob   = float(predictions[sample_idx])
        gt          = int(labels[sample_idx])
        ped_id      = extract_ped_id(test_data["ped_id"][sample_idx])
        image_paths = test_data["image"][sample_idx]

        sheet      = make_contact_sheet(image_paths, n_frames=args.n_frames)
        prompt     = build_prompt(pvim_prob)
        raw        = query_qwen(processor, model, sheet, prompt)
        qwen_dec, qwen_ok = parse_decision(raw)
        qwen_bin   = 1.0 if qwen_dec == "yes" else 0.0

        pvim_decision  = "yes" if pvim_prob >= 0.5 else "no"
        fused_score    = beta * pvim_prob + (1.0 - beta) * qwen_bin
        fused_decision = "yes" if fused_score >= threshold else "no"
        correct        = (fused_decision == ("yes" if gt == 1 else "no"))

        rows.append({
            "sample_id":      sample_idx,
            "ped_id":         ped_id,
            "pvim_prob":      pvim_prob,
            "pvim_decision":  pvim_decision,
            "qwen_raw":       raw,
            "qwen_binary":    qwen_dec,
            "qwen_score":     qwen_bin,
            "qwen_parsed_ok": qwen_ok,
            "fused_score":    fused_score,
            "final_decision": fused_decision,
            "ground_truth":   gt,
            "correct":        correct,
            "image_paths":    ";".join(image_paths),
        })

        status = "✓" if correct else "✗"
        trunc_warn = "  [TRUNC?]" if not qwen_ok else ""
        print(f"[{run_idx+1:>4}/{len(chosen)}] {ped_id:12s} pvim={pvim_prob:.3f}  "
              f"qwen={qwen_dec:3s}  fused={fused_score:.3f}  gt={gt}  {status}{trunc_warn}")

    # ── 6. Save + summary ─────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    acc  = sum(r["correct"] for r in rows) / len(rows)
    n_yes = sum(1 for r in rows if r["final_decision"] == "yes")
    n_no  = sum(1 for r in rows if r["final_decision"] == "no")
    tp   = sum(1 for r in rows if r["final_decision"] == "yes" and r["ground_truth"] == 1)
    fp   = sum(1 for r in rows if r["final_decision"] == "yes" and r["ground_truth"] == 0)
    tn   = sum(1 for r in rows if r["final_decision"] == "no"  and r["ground_truth"] == 0)
    fn   = sum(1 for r in rows if r["final_decision"] == "no"  and r["ground_truth"] == 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n{'='*52}")
    print(f"  Samples   : {len(rows)}  ({n_pos} cross + {n_neg} no-cross)")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Predicted yes={n_yes}  no={n_no}")

    n_unparsed = sum(1 for r in rows if not r["qwen_parsed_ok"])
    if n_unparsed:
        print(f"  WARNING: {n_unparsed}/{len(rows)} Qwen responses had no clear yes/no — defaulted to 'no'")

    pvim_acc = sum(1 for r in rows if r["pvim_decision"] == ("yes" if r["ground_truth"] == 1 else "no")) / len(rows)
    qwen_acc = sum(1 for r in rows if r["qwen_binary"]   == ("yes" if r["ground_truth"] == 1 else "no")) / len(rows)
    print(f"  PVIM-only acc : {pvim_acc:.4f}")
    print(f"  Qwen-only acc : {qwen_acc:.4f}")
    print(f"  Fused acc     : {acc:.4f}")

    print(f"  Saved to  : {args.output}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()

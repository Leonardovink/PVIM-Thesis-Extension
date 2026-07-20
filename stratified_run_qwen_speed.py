"""stratified_run_qwen_speed.py — Standard (non-temporal) Qwen+PVIM prompt
with red bounding boxes and ego-vehicle speed (time-conscious).

This is the standard 6-cue checklist variant of stratified_run_qwen.py with
the bbox + speed additions from Azarmi et al. (IV 2025), letting you compare
prompt design (standard vs. temporal) independently of the bbox/speed cues.

Output: yes/no binary (same as the original standard script), parsed into a
0/1 qwen_score for the fusion grid search downstream.
"""

import argparse
import csv
import gc
import os
import pickle
import re
import sys
import yaml

import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PATH_PIE           = "PIE_master"
DEFAULT_CONFIG     = os.path.join("config_files", "PVIM.yaml")
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
PIE_DB_PATH        = os.path.join("PIE_master", "data_cache", "pie_database.pkl")
FPS                = 30


# ── Prompt templates ──────────────────────────────────────────────────────────

ROLE_TEMPLATE = (
    "You are an autonomous vehicle safety system. "
    "Your task is to predict whether a pedestrian intends to cross the road "
    "in the near future, based on visual evidence from a front-facing camera."
)

PHYSICAL_CUES_TEMPLATE = (
    "Analyse the pedestrian marked with the red bounding box in the image sequence carefully. "
    "Before answering, reason through each of the following cues:\n"
    "1. BODY ORIENTATION: Is the pedestrian facing towards the road, away from it, or sideways?\n"
    "2. GAZE: Are they looking towards the road or traffic?\n"
    "3. MOVEMENT: Are they stepping towards the curb, standing still, or moving parallel to the road?\n"
    "4. POSTURE: Do they appear ready to step off the curb (leaning forward, weight shifted)?\n"
    "5. CONTEXT: Are they near a crosswalk or intersection? Is there a gap in traffic?\n"
    "6. EGO-VEHICLE DYNAMICS: {speed_text}\n"
    "7. PVIM MODEL: A trained pedestrian-vehicle interaction model estimates "
    "the crossing probability as {pvim_prob:.2f}/1.00. {pvim_interp}\n\n"
    "Consider ALL of the above before deciding. "
    "A pedestrian can intend to cross even if currently walking parallel to the road. "
    "Answer with exactly one word on the final line: yes or no."
)


# ── PIE annotation lookup ─────────────────────────────────────────────────────

PATH_RE = re.compile(r"set(\d+)[/\\]video_(\d+)[/\\](\d+)\.png", re.IGNORECASE)


def parse_image_path(path: str) -> Optional[Tuple[str, str, int]]:
    m = PATH_RE.search(path.replace("\\", "/"))
    if not m:
        return None
    return f"set{m.group(1)}", f"video_{m.group(2)}", int(m.group(3))


def load_pie_lookups(db_path: str = PIE_DB_PATH) -> Dict:
    if not os.path.exists(db_path):
        sys.exit(f"PIE annotation cache not found at {db_path}.")
    print(f"Loading PIE annotations from {db_path} ...")
    with open(db_path, "rb") as f:
        db = pickle.load(f)

    bbox_by_pid: Dict[Tuple[str, str, str, int], List[float]] = {}
    speed_by_vid: Dict[Tuple[str, str, int], float] = {}
    for sid, sdata in db.items():
        for vid, vdata in sdata.items():
            for pid, ped_ann in (vdata.get("ped_annotations") or {}).items():
                frames = ped_ann.get("frames", [])
                bboxes = ped_ann.get("bbox", [])
                for fr, bb in zip(frames, bboxes):
                    bbox_by_pid[(sid, vid, pid, int(fr))] = list(bb)
            vann = vdata.get("vehicle_annotations") or {}
            for fr, frame_data in vann.items():
                if not isinstance(frame_data, dict):
                    continue
                sp = frame_data.get("OBD_speed")
                if sp is None:
                    continue
                try:
                    speed_by_vid[(sid, vid, int(fr))] = float(sp)
                except (TypeError, ValueError):
                    continue
    print(f"  Loaded {len(bbox_by_pid)} ped bboxes, {len(speed_by_vid)} speed entries")
    return {"bbox_by_pid": bbox_by_pid, "speed_by_vid": speed_by_vid}


def lookup_bbox(lookups, set_id, video_id, ped_id, frame):
    return lookups["bbox_by_pid"].get((set_id, video_id, ped_id, int(frame)))


def lookup_speed(lookups, set_id, video_id, frame):
    return lookups["speed_by_vid"].get((set_id, video_id, int(frame)))


# ── PVIM helpers ──────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def configure_tensorflow_memory_growth():
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
        print(f"  PVIM batch {start}-{end}/{n_samples}", end="\r")
    print()
    return np.concatenate(all_preds)


def extract_ped_id(ped_id_entry):
    try:
        return str(np.asarray(ped_id_entry).flat[0])
    except Exception:
        return "unknown"


# ── Image helpers ─────────────────────────────────────────────────────────────

def make_contact_sheet_with_bbox(image_paths, ped_id, lookups, n_frames=8, size=336):
    n_frames = min(n_frames, len(image_paths))
    indices  = np.linspace(0, len(image_paths) - 1, n_frames, dtype=int)
    frames = []
    n_drawn = 0
    for i in indices:
        path = image_paths[i]
        parsed = parse_image_path(path)
        img = Image.open(path).convert("RGB")
        orig_w, _ = img.size
        if parsed is not None:
            set_id, video_id, frame_idx = parsed
            bb = lookup_bbox(lookups, set_id, video_id, ped_id, frame_idx)
            if bb is not None:
                draw = ImageDraw.Draw(img)
                lw = max(4, orig_w // 200)
                draw.rectangle(bb, outline=(255, 0, 0), width=lw)
                n_drawn += 1
        frames.append(img.resize((size, size), Image.LANCZOS))
    cols  = min(n_frames, 4)
    rows  = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (size * cols, size * rows), (0, 0, 0))
    for idx, frame in enumerate(frames):
        x = (idx % cols) * size
        y = (idx // cols) * size
        sheet.paste(frame, (x, y))
    return sheet, n_drawn


def build_speed_text(image_paths, lookups) -> Tuple[str, Optional[float], Optional[float], Optional[float]]:
    """Build the speed cue text. Returns (text, sp_start, sp_end, dt_s)."""
    first = parse_image_path(image_paths[0])
    last  = parse_image_path(image_paths[-1])
    if first is None or last is None:
        return "Ego-vehicle speed information is not available for this clip.", None, None, None
    sp_start = lookup_speed(lookups, first[0],  first[1],  first[2])
    sp_end   = lookup_speed(lookups, last[0],   last[1],   last[2])
    if sp_start is None or sp_end is None:
        return "Ego-vehicle speed information is not available for this clip.", None, None, None
    dt_s = (last[2] - first[2]) / FPS if last[2] >= first[2] else None
    if dt_s is None or dt_s <= 0:
        return f"The ego-vehicle is currently moving at {sp_start:.0f} km/h.", sp_start, sp_end, None
    delta = sp_end - sp_start
    if abs(delta) < 1.0:
        body = f"Over the past {dt_s:.2f}s the ego-vehicle maintained its speed at {sp_start:.0f} km/h."
    elif delta < 0:
        body = f"Over the past {dt_s:.2f}s the ego-vehicle decelerated from {sp_start:.0f} km/h to {sp_end:.0f} km/h."
    else:
        body = f"Over the past {dt_s:.2f}s the ego-vehicle accelerated from {sp_start:.0f} km/h to {sp_end:.0f} km/h."
    return body, sp_start, sp_end, dt_s


# ── Prompt helpers ────────────────────────────────────────────────────────────

def interpret_pvim_probability(pvim_prob: float) -> str:
    if pvim_prob >= 0.80:   level = "a very high"
    elif pvim_prob >= 0.60: level = "a high"
    elif pvim_prob >= 0.40: level = "a moderate"
    elif pvim_prob >= 0.20: level = "a low"
    else:                   level = "a very low"
    return (f"This indicates {level} model-estimated probability that the pedestrian "
            "will cross the road soon.")


def build_prompt(pvim_prob: float, speed_text: str) -> str:
    pvim_interp = interpret_pvim_probability(pvim_prob)
    image_context = (
        "The image is a grid of frames from the same front-facing camera video. "
        "Panels are ordered left-to-right, top-to-bottom in time, from earliest to latest.\n\n"
        "The target pedestrian is marked with a red bounding box in every frame.\n\n"
    )
    return (
        ROLE_TEMPLATE
        + "\n\n"
        + image_context
        + PHYSICAL_CUES_TEMPLATE.format(
            pvim_prob=pvim_prob, pvim_interp=pvim_interp, speed_text=speed_text,
        )
    )


# ── Qwen ──────────────────────────────────────────────────────────────────────

def load_qwen(model_name=DEFAULT_QWEN_MODEL):
    pkgdir = os.environ.get("QWEN_PACKAGES", "")
    if pkgdir and pkgdir not in sys.path:
        sys.path.insert(0, pkgdir)
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        import torch
    except ImportError as e:
        sys.exit(f"transformers/torch import failed: {e}")
    print(f"Loading Qwen2.5-VL: {model_name} ...")
    sys.stdout.flush()
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    print(f"Qwen2.5-VL loaded on device(s): {set(str(p.device) for p in model.parameters())}")
    return processor, model


def query_qwen(processor, model, sheet, prompt_text):
    from qwen_vl_utils import process_vision_info
    import torch
    messages = [
        {"role": "system", "content": ROLE_TEMPLATE},
        {"role": "user", "content": [
            {"type": "image", "image": sheet},
            {"type": "text",  "text": prompt_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del generated_ids, inputs, trimmed
    torch.cuda.empty_cache()
    return output[0].strip()


def parse_decision(raw):
    """Parse yes/no from the bottom of Qwen's response.

    Returns (qwen_score in {0.0, 1.0}, parsed_ok).
    Defaults to 0.5 + parsed_ok=False if nothing matches.
    """
    lines = [l.strip().lower().rstrip(".") for l in raw.splitlines() if l.strip()]
    for line in reversed(lines):
        if re.search(r"\byes\b", line) and not re.search(r"\bno\b", line):
            return 1.0, True
        if re.search(r"\bno\b", line) and not re.search(r"\byes\b", line):
            return 0.0, True
    return 0.5, False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",        default="saved_models/pvim")
    parser.add_argument("--config",           default=DEFAULT_CONFIG)
    parser.add_argument("--n-per-class",      type=int,   default=100)
    parser.add_argument("--n-frames",         type=int,   default=8)
    parser.add_argument("--output",           default="results/qwen_pvim_speed_stratified.csv")
    parser.add_argument("--qwen-model",       default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--fusion-beta",      type=float, default=0.5)
    parser.add_argument("--fusion-threshold", type=float, default=0.5)
    parser.add_argument("--pie-db",           default=PIE_DB_PATH)
    args = parser.parse_args()

    configure_tensorflow_memory_growth()

    print("\n=== Loading PVIM data ===")
    test_data, _ = load_all_pvim_data(args.config)
    labels = np.asarray(test_data["data"][1]).flatten()
    print(f"  Total: {len(labels)}  Cross: {int((labels==1).sum())}  No-cross: {int((labels==0).sum())}")

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
    print(f"\n  Stratified subset: {n_pos} + {n_neg} = {len(chosen)} samples")

    print("\n=== Loading PIE annotation lookups ===")
    lookups = load_pie_lookups(args.pie_db)

    print("\n=== Running PVIM inference ===")
    pvim_model  = load_pvim_model(args.model_dir)
    predictions = run_pvim_predictions(pvim_model, test_data)
    del pvim_model
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"WARNING: could not clear TF session: {e}")
    gc.collect()

    print("\n=== Loading Qwen ===")
    processor, model = load_qwen(args.qwen_model)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows = []
    print(f"\n=== Running Qwen on {len(chosen)} samples ===\n")
    beta      = args.fusion_beta
    threshold = args.fusion_threshold
    n_speed_missing = 0
    n_bbox_missing  = 0

    for run_idx, sample_idx in enumerate(chosen):
        sample_idx  = int(sample_idx)
        pvim_prob   = float(predictions[sample_idx])
        gt          = int(labels[sample_idx])
        ped_id      = extract_ped_id(test_data["ped_id"][sample_idx])
        image_paths = test_data["image"][sample_idx]

        sheet, n_drawn = make_contact_sheet_with_bbox(
            image_paths, ped_id, lookups, n_frames=args.n_frames,
        )
        if n_drawn == 0:
            n_bbox_missing += 1

        speed_text, sp_start, sp_end, dt = build_speed_text(image_paths, lookups)
        speed_ok = sp_start is not None and sp_end is not None
        if not speed_ok:
            n_speed_missing += 1

        prompt = build_prompt(pvim_prob, speed_text)
        raw    = query_qwen(processor, model, sheet, prompt)
        qwen_score, qwen_ok = parse_decision(raw)

        pvim_decision  = "yes" if pvim_prob >= 0.5 else "no"
        fused_score    = beta * pvim_prob + (1.0 - beta) * qwen_score
        fused_decision = "yes" if fused_score >= threshold else "no"
        correct        = (fused_decision == ("yes" if gt == 1 else "no"))

        rows.append({
            "sample_id":      sample_idx,
            "ped_id":         ped_id,
            "pvim_prob":      pvim_prob,
            "pvim_decision":  pvim_decision,
            "qwen_raw":       raw,
            "qwen_score":     qwen_score,
            "qwen_parsed_ok": qwen_ok,
            "fused_score":    fused_score,
            "final_decision": fused_decision,
            "ground_truth":   gt,
            "correct":        correct,
            "n_bbox_drawn":   n_drawn,
            "speed_start":    "" if sp_start is None else f"{sp_start:.1f}",
            "speed_end":      "" if sp_end   is None else f"{sp_end:.1f}",
            "speed_dt_s":     "" if dt       is None else f"{dt:.2f}",
            "image_paths":    ";".join(image_paths),
        })

        status = "OK" if correct else "WRONG"
        trunc = "  [NO YES/NO]" if not qwen_ok else ""
        print(f"[{run_idx+1:>4}/{len(chosen)}] {ped_id:12s} pvim={pvim_prob:.3f}  "
              f"qwen={qwen_score:.2f}  fused={fused_score:.3f}  gt={gt}  {status}{trunc}  "
              f"(bbox={n_drawn}/{args.n_frames}, speed={'yes' if speed_ok else 'no'})")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    acc  = sum(r["correct"] for r in rows) / len(rows)
    tp   = sum(1 for r in rows if r["final_decision"] == "yes" and r["ground_truth"] == 1)
    fp   = sum(1 for r in rows if r["final_decision"] == "yes" and r["ground_truth"] == 0)
    tn   = sum(1 for r in rows if r["final_decision"] == "no"  and r["ground_truth"] == 0)
    fn   = sum(1 for r in rows if r["final_decision"] == "no"  and r["ground_truth"] == 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Samples            : {len(rows)} (cross={n_pos} + no-cross={n_neg})")
    print(f"  Bbox missing       : {n_bbox_missing}")
    print(f"  Speed missing      : {n_speed_missing}")
    print(f"  Accuracy / F1      : {acc:.4f} / {f1:.4f}")
    print(f"  Precision / Recall : {prec:.4f} / {rec:.4f}")
    print(f"  TP/FP/TN/FN        : {tp}/{fp}/{tn}/{fn}")
    n_unparsed = sum(1 for r in rows if not r["qwen_parsed_ok"])
    if n_unparsed:
        print(f"  WARNING: {n_unparsed}/{len(rows)} responses missing yes/no")
    print(f"  Saved to           : {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""Draft: LoRA fine-tuning Qwen2.5-VL on PIE crossing labels.

Instead of zero-shot prompting, fine-tune the VLM to predict crossing
directly from the same contact-sheet + prompt input. QLoRA keeps memory
manageable on a single H100 (80 GB).

Method:
    1. Build a supervised dataset from PIE: (contact_sheet, prompt, label)
       reusing the same 8-frame 336x336 tiled images and 7-cue prompt
       from stratified_run_qwen_speed.py.
    2. Attach LoRA adapters to the language model's attention layers.
    3. Fine-tune on binary cross-entropy (crossing vs not-crossing)
       using the VLM's next-token prediction on "yes"/"no".
    4. Evaluate with the same 20-seed stratified validation protocol.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# PVIM runs on TensorFlow and TF grabs the whole GPU by default. Pin TF to CPU
# so all 93 GB stays free for the Qwen2.5-VL load (TF's CPU pin does not affect
# PyTorch's CUDA access). PVIM is tiny, so CPU inference is fine.
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import csv
import yaml
import numpy as np
from PIL import Image
from typing import List, Tuple

import torch
if not hasattr(torch.compiler, "is_compiling"):
    # torch 2.1.2 lacks torch.compiler.is_compiling, which the Qwen2.5-VL fast
    # image processor calls. We never use torch.compile, so stub it to False.
    torch.compiler.is_compiling = lambda *a, **k: False

# ---------------------------------------------------------------------------
# 1. Dataset: reuse the existing contact-sheet builder
# ---------------------------------------------------------------------------

def load_pvim_data_split(config_path, split="train", set_ids=None):
    """Load a PIE split, optionally restricted to specific set folders.

    Mirrors stratified_run_qwen_speed.load_all_pvim_data but lets us pick the
    split (train vs test) and limit it to the sets we actually have extracted
    (e.g. set01+set02 for training, since set04 frames are not on disk).
    """
    from action_predict import action_prediction
    from pie_data import PIE
    from stratified_run_qwen_speed import load_yaml, PATH_PIE

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
    if set_ids is not None:
        _orig = imdb._get_image_set_ids
        imdb._get_image_set_ids = (
            lambda image_set: list(set_ids) if image_set == split else _orig(image_set)
        )
    raw = imdb.generate_data_trajectory_sequence(split, **configs["data_opts"])
    mc = action_prediction(configs["model_opts"]["model"])(**configs["net_opts"])
    td = mc.get_data(split, raw, {**configs["model_opts"], "batch_size": 1})
    return td, configs


def build_training_samples(
    config_path: str = "config_files/PVIM_eval_tte90.yaml",
    model_dir: str = "data/models/pie/PVIM/ckpt_tte60_120",
    split: str = "train",
    set_ids=None,
) -> List[dict]:
    """Build (image, prompt, label) tuples from the given PIE split/sets.

    set_ids=None uses PIE's native split (train = set01/set02/set04, test = set03).
    Pass an explicit tuple to restrict/extend (e.g. all non-test sets).
    """
    from stratified_run_qwen_speed import (
        load_pvim_model,
        run_pvim_predictions,
        make_contact_sheet_with_bbox,
        build_speed_text,
        load_pie_lookups,
        parse_image_path,
        extract_ped_id,
        PHYSICAL_CUES_TEMPLATE,
        ROLE_TEMPLATE,
    )

    test_data, configs = load_pvim_data_split(config_path, split=split, set_ids=set_ids)
    pvim_model = load_pvim_model(model_dir)
    pvim_probs = run_pvim_predictions(pvim_model, test_data)

    gt = np.asarray(test_data["data"][1]).flatten().astype(int)
    image_lists = test_data["image"]
    ped_ids = test_data["ped_id"]
    lookups = load_pie_lookups()

    samples = []
    for i in range(len(gt)):
        paths = image_lists[i]
        ped_id = extract_ped_id(ped_ids[i])

        sheet, _ = make_contact_sheet_with_bbox(
            paths, ped_id, lookups, n_frames=8, size=336
        )

        pvim_prob = float(pvim_probs[i])
        pvim_interp = (
            "This is a high probability." if pvim_prob >= 0.5
            else "This is a low probability."
        )
        speed_text, _, _, _ = build_speed_text(paths, lookups)

        prompt = PHYSICAL_CUES_TEMPLATE.format(
            pvim_prob=pvim_prob,
            pvim_interp=pvim_interp,
            speed_text=speed_text,
        )

        samples.append({
            "image": sheet,                # PIL Image
            "system": ROLE_TEMPLATE,
            "prompt": prompt,
            "label": "yes" if gt[i] == 1 else "no",
            "label_int": int(gt[i]),
            "pvim_prob": pvim_prob,        # for fusion at eval time
            "sample_id": ped_id,
        })

    return samples


# ---------------------------------------------------------------------------
# 2. LoRA setup
# ---------------------------------------------------------------------------

def setup_model_with_lora(
    model_name: str = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Load Qwen2.5-VL in bf16 and attach LoRA adapters.

    Plain LoRA (not 4-bit QLoRA): the 32B model in bf16 (~64 GB) fits on a
    93 GB H100, and this avoids the bitsandbytes/transformers version clash on
    the offline cluster. model_name defaults to the QWEN_MODEL env var.
    """
    if model_name is None:
        model_name = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
    print(f"Loading VLM from: {model_name}")
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    # Pin the whole model to GPU 0. device_map="auto" reserves headroom and
    # offloads layers to CPU, which streams weights every step (~100x slower).
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for grad-checkpointing + LoRA

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    return model, processor


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------

def train(
    model,
    processor,
    samples: List[dict],
    epochs: int = 3,
    batch_size: int = 1,
    lr: float = 2e-4,
    output_dir: str = "data/models/qwen_lora_pie",
):
    """Fine-tune on binary yes/no prediction."""
    import torch
    from torch.optim import AdamW
    from qwen_vl_utils import process_vision_info

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    for epoch in range(epochs):
        np.random.shuffle(samples)
        total_loss = 0.0

        for i, sample in enumerate(samples):
            messages = [
                {"role": "system", "content": sample["system"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image"]},
                        {"type": "text", "text": sample["prompt"]},
                    ],
                },
                {"role": "assistant", "content": sample["label"]},
            ]

            # full sequence (prompt + assistant answer)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # prompt only (no answer) - used to measure how many tokens to mask
            prompt_text = processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, return_tensors="pt", padding=True,
            ).to(model.device)
            prompt_inputs = processor(
                text=[prompt_text], images=image_inputs, return_tensors="pt", padding=True,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # loss only on the answer tokens: mask the prompt with -100 so the
            # long 7-cue prompt does not drown out the yes/no signal
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100
            inputs["labels"] = labels

            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch+1}, sample {i+1}/{len(samples)}, "
                      f"loss={total_loss/(i+1):.4f}", flush=True)

            # periodic checkpoint so the walltime can't wipe all progress
            if (i + 1) % 200 == 0:
                model.save_pretrained(output_dir)

        print(f"Epoch {epoch+1}: avg loss = {total_loss/len(samples):.4f}", flush=True)
        model.save_pretrained(output_dir)   # checkpoint after every epoch

    model.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}")


# ---------------------------------------------------------------------------
# 4. Inference with fine-tuned model
# ---------------------------------------------------------------------------

def predict_finetuned(model, processor, image, system_prompt, user_prompt):
    """Run inference with the LoRA-adapted model."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(new_tokens, skip_special_tokens=True)
    return response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"),
                    help="Local path to the Qwen2.5-VL model (or HF repo id).")
    ap.add_argument("--model-dir", default="data/models/pie/PVIM/ckpt_tte60_120",
                    help="PVIM checkpoint dir for the PVIM-probability cue.")
    ap.add_argument("--epochs", type=int, default=2,
                    help="Number of fine-tuning epochs.")
    ap.add_argument("--train-sets", default="",
                    help="Comma-separated PIE sets to train on (empty = native "
                         "train split set01,set02,set04). E.g. set01,set02,set04,set05,set06")
    ap.add_argument("--output-dir", default="data/models/qwen_lora_pie",
                    help="Where to save the LoRA adapter (use a unique name per run "
                         "so runs don't overwrite each other).")
    args = ap.parse_args()

    set_ids = tuple(s.strip() for s in args.train_sets.split(",") if s.strip()) or None
    print("=== Building training samples ===")
    print(f"  train sets: {set_ids if set_ids else 'native PIE train (set01,set02,set04)'}")
    samples = build_training_samples(model_dir=args.model_dir, set_ids=set_ids)
    print(f"  {len(samples)} samples "
          f"({sum(s['label_int'] for s in samples)} cross, "
          f"{sum(1-s['label_int'] for s in samples)} no-cross)")

    print("\n=== Loading model with QLoRA ===")
    model, processor = setup_model_with_lora(model_name=args.qwen_model)

    print(f"\n=== Training ({args.epochs} epochs) -> {args.output_dir} ===")
    train(model, processor, samples, epochs=args.epochs, output_dir=args.output_dir)

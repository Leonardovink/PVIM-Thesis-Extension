# Fine-tuning a Vision-Language Model for long-term pedestrian crossing prediction

Honours extension to my bachelor thesis *Vision-Language Models for Long-Term
Pedestrian Crossing Prediction*. The thesis fused a geometric model (PVIM) with a
prompted, frozen Qwen2.5-VL-32B and found the two complementary. This extension
asks the next question: if the VLM is *adapted* to the crossing domain with LoRA
instead of only prompted, does its standalone signal improve, and does the
geometric model still add anything once it has?

The adapted VLM on its own reaches 0.644 F1 at three seconds, above the thesis's
best fusion of 0.573, using the images alone. Fusing with PVIM still helps at four
seconds, where the images run out.

Main thesis repository:
https://github.com/Leonardovink/Vision-Language-Models-for-Long-Term-Pedestrian-Crossing-Prediction

## Layout

```
*.py                  the experiment scripts and shared modules (flat, as run)
config_files/         PVIM training and evaluation configs (yaml)
environment/          setup_qwen_env.sh + pinned package versions
jobs/                 the Snellius (SLURM) job scripts, in run order
results/              raw per-sample scores from every run (CSV)
models/               the trained LoRA adapters (via Git LFS, see below)
figures/              figures used in the write-up
```

The three experiments, in the order they run (see `jobs/`):

- `lora_finetune_qwen.py` (`jobs/01_lora_finetune.job`) — LoRA fine-tuning of the
  VLM; establishes that adapting it helps at all, and produces the model the other
  two refine.
- `lora_search_6012.py` (`jobs/02_lora_search.job`) — search over learning rate,
  schedule, passes, and class balancing for the training setup.
- `lora_prompt_variant.py` (`jobs/03_prompt_ablation.job`, `04_prompt_trim.job`) —
  leave-one-out ablation over the six prompt cues, one retrain per dropped cue.

`jobs/05_eval_adapter.job` re-scores any saved adapter without retraining. The rest
of the top level is shared machinery: `lora_sweep.py` (model loading, prompt
encoding, the training loop), `lora_confirm_winner.py` (the selected training
config), `lora_validated.py` (the 20-split validation/test protocol),
`prompt_ablation.py` and `prompt_ablation_cues.py` (data loading and prompt
assembly), the geometric-side code carried over from the main thesis
(`pie_data.py`, `action_predict.py`, `base_models.py`, `stratified_run_qwen_speed.py`),
and the cluster sanity checks `preflight.py` / `smoke_infer.py`.

## Environment

Training was run on the Snellius H100 cluster. PyTorch 2.1.2 (CUDA 12.1),
TensorFlow, and NumPy 1.25 are provided by the cluster modules; the rest are pinned
in `environment/requirements.txt`. `environment/setup_qwen_env.sh` reproduces the
exact environment, and `environment/requirements-lock.txt` is the full frozen list.
On other hardware, install a matching torch build first, then the pinned packages.

## Running

The scripts import each other as top-level modules and read `config_files/`
relative to the working directory, so run them from the repository root, as the
jobs in `jobs/` do:

```
# fine-tune the baseline adapted model
python lora_finetune_qwen.py --help

# search over training settings
python lora_search_6012.py --help

# prompt-cue ablation: retrain with one cue removed
python lora_prompt_variant.py --drop gaze --tag drop_gaze

# re-score a saved adapter without retraining
python lora_prompt_variant.py --from-adapter models/lora_2-4s_opt_nopvim \
    --drop none --tag base_nopvim --eval-test

# rebuild the ablation figure from the shipped result CSVs
python make_cue_ablation_fig.py
```

Every run writes per-sample scores (PVIM probability, VLM score, ground truth) to
`results/`, and metrics are computed under the 20-seed protocol in
`lora_validated.py`: each split is a stratified 50/50 partition, the threshold (and
fusion weight, when fusing) is tuned on one half and reported on the other, and
results are the mean over 20 splits.

## Trained models

`models/` holds the trained LoRA adapters (base weights frozen, only the adapters
are trained):

- `models/promptstudy/drop_<cue>/` — the six ablation variants, one cue removed each
- `models/lora_2-4s_opt_nopvim/` — the optimised model with all six cues (the
  headline model; PVIM removed from the prompt)
- `models/lora_2-4s_optimized/` — optimised, with PVIM kept in the prompt
- `models/lora_2-4s_beforeopt/` — the pre-search model, for comparison

Each adapter is an `adapter_config.json` + `adapter_model.safetensors` (~270 MB) and
needs the base Qwen2.5-VL-32B-Instruct weights to run. Reload one with
`--from-adapter` as shown above.

The `.safetensors` weight files exceed GitHub's 100 MB per-file limit, so only the
`adapter_config.json` files are kept in the repository; the weights are hosted on
Google Drive. See `models/README.md` for the download link and where to place each
file.

## Results

VLM-only F1 at the F1-optimal threshold, mean over 20 splits, on the PIE test set
(set03):

| horizon | thesis best fusion | adapted VLM alone | adapted VLM + PVIM |
| ------- | ------------------ | ----------------- | ------------------ |
| 2 s     | 0.604              | 0.721             | 0.723              |
| 3 s     | 0.573              | 0.644             | 0.637              |
| 4 s     | 0.553              | 0.573             | 0.637              |

At two and three seconds the adapted VLM carries the decision on its own; at four
seconds only the fusion clears the baseline, so the geometric model earns its place
exactly where the images are weakest. The cue ablation
(`figures/abl_cue_ranking_val.png`) shows all six prompt cues contribute, with the
ego-vehicle speed by far the most important. Numbers are from a single training run
per model; the seed-stability of the margins is discussed in the thesis.

## Data

The PIE dataset is not redistributed here and must be obtained from its authors
(Rasouli et al., 2019). The base Qwen2.5-VL-32B-Instruct weights are downloaded from
Hugging Face. Neither is included in this repository.

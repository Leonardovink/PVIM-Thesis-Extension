# Trained LoRA adapters

Each folder is a LoRA adapter for Qwen2.5-VL-32B-Instruct (base weights frozen,
only the adapter trained). The `adapter_config.json` is kept here; the weights
(`adapter_model.safetensors`, ~270 MB each) are over GitHub's 100 MB file limit and
are hosted on Google Drive:

**Weights: <ADD GOOGLE DRIVE LINK>**

Download the `adapter_model.safetensors` for the adapter you need and drop it into
the matching folder next to its `adapter_config.json`.

| folder | what it is |
| ------ | ---------- |
| `lora_2-4s_opt_nopvim/` | optimised model, all six cues, PVIM removed from the prompt (the headline model) |
| `lora_2-4s_optimized/` | optimised, with PVIM kept in the prompt |
| `lora_2-4s_beforeopt/` | pre-search model, for comparison |
| `promptstudy/drop_<cue>/` | the six ablation variants, one prompt cue removed each |

All were trained on the 2–4 second range with the selected configuration: learning
rate 1e-4, cosine schedule with warm-up, class-balanced loss, rank 8, alpha 16, one
pass, fixed seed. Reload one with:

```
python lora_prompt_variant.py --from-adapter models/lora_2-4s_opt_nopvim \
    --drop none --tag base_nopvim --eval-test
```

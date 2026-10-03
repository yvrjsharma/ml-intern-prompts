Train a LoRA so a small text-to-image model can draw new illustrations of **Huggy**, the Hugging Face mascot, in the official brand style shown in the Space `Chunte/HFBA` (flat 2D vector look, round yellow body, thick orange outline, floating hands, simple face, plain white background, a prop or costume per image).

**Dataset.** Use `Chunte/huggy_for_training` (public). It is the captioned version of the images in that Space: a `train/` folder with 84 pairs `huggy_NNN.png` + `huggy_NNN.txt`. All images are 1024x1024 on a white background. 13 are RGBA with no real transparency, so convert everything to RGB. Every caption is comma-separated tags and starts with the trigger word and fixed style tags, then what varies, for example:
`hggycrcl, circular body, floating hands, orange outlined, flat 2D, black dot eyes, red open mouth, tilted, pointing left, blue headphones, winking open mouth`
`load_dataset` will NOT pair the .txt captions with the images automatically. Download the repo with `snapshot_download`, then build an imagefolder with a `metadata.jsonl` (`file_name`, `text`) and train from that with `--caption_column text`. Keep the captions exactly as written. Use all 84 images for training; there is no test split for a style LoRA.

**Base model.** `black-forest-labs/FLUX.2-klein-base-4B` (Apache 2.0, ungated). Train on the undistilled *base* checkpoint, not the distilled `FLUX.2-klein-4B`, because distilled models take LoRAs poorly. At the end, also test the trained LoRA on the distilled `FLUX.2-klein-4B` for fast 4-step inference and report whether it transfers.

**Training script.** Use the official diffusers script `examples/dreambooth/train_dreambooth_lora_flux2_klein.py` from the `main` branch, with diffusers installed from source. Read `examples/dreambooth/README_flux2.md` first; note that `--remote_text_encoder` is not supported for klein. Starting recipe:
- LoRA rank 32, alpha 32, default attention target layers
- resolution 1024, no center crop, no random flip (several captions say "pointing left", so flipping would break them)
- batch size 1, gradient accumulation 4, 600 optimizer steps (about 28 epochs), checkpoint every 100 steps
- AdamW 8-bit, learning rate 1e-4, constant schedule with 50 warmup steps
- bf16, gradient checkpointing, `--cache_latents`, `--guidance_scale=1` as in the README, `--offload` only if memory requires it
- `--instance_prompt "hggycrcl"` as the fallback prompt
- Only add `--do_fp8_training` on a GPU with compute capability 8.9 or higher (L40S, H200). Do not use it on A100.
- Report loss to Trackio. This script only logs validation *images* to wandb or tensorboard, so save validation images to disk yourself and upload them.

**Validation and checkpoint selection.** Fix one seed and generate these 6 prompts, none of which are in the training set, with the base model alone and with every checkpoint (100 to 600). Each uses the training caption structure:
1. `hggycrcl, circular body, floating hands, orange outlined, flat 2D, black dot eyes, red open mouth, wearing white astronaut helmet, holding small red rocket, happy open mouth`
2. `hggycrcl, circular body, floating hands, orange outlined, flat 2D, black dot eyes, red open mouth, tilted, holding green watering can, straw farmer hat, happy open mouth`
3. `hggycrcl, circular body, floating hands, orange outlined, flat 2D, black dot eyes, red open mouth, wearing pirate hat and eye patch, holding treasure map, winking open mouth`
4. `hggycrcl, circular body, floating hands, orange outlined, flat 2D, curved closed eyes, peaceful, sitting in meditation pose, purple yoga mat`
5. `hggycrcl, circular body, floating hands, orange outlined, flat 2D, black dot eyes, red open mouth, riding blue bicycle, red scarf, rushing with speed lines, happy open mouth`
6. `hggycrcl, circular body, floating hands, orange outlined, flat 2D, starry eyes, holding orange citrus fruit, green leaf on head, happy open mouth`
Also generate two leak checks WITHOUT the trigger word (`a photo of a golden retriever in a park`, `a watercolor painting of a lighthouse`) to see whether the style bleeds into unrelated prompts. Build one comparison grid image (rows = prompts, columns = base, step 100 ... step 600). Pick the earliest checkpoint where the character is on-model (round body, orange outline, floating hands, white background, correct prop) and the leak checks still look normal. Explain your choice. If 600 steps underfits, continue to 1000; if step 200 already overfits, say so.

**Deliverables.**
1. Push the chosen LoRA to `ysharma/huggy-flux2-klein-lora` with a model card containing: the trigger word `hggycrcl`, the caption recipe (fixed style tags, then pose, props, expression), the comparison grid, recommended inference settings for both the base and the distilled klein model, LoRA scale guidance, training config, and attribution (artwork by Chunte Lee / Hugging Face brand assets, dataset `Chunte/huggy_for_training`, Space `Chunte/HFBA`).
2. Deploy a Gradio Space `ysharma/huggy-generator` on ZeroGPU: a text box where the user describes only the pose, props and expression, a checkbox (on by default) that auto-prepends the trigger word and fixed style tags, sliders for LoRA scale, steps and seed, and at least 8 clickable examples. Use `@spaces.GPU`, load the model at module scope, and follow the ZeroGPU rules (no `torch.compile`).

**Budget.** First run a 20-step smoke test at 512 resolution on a cheap GPU to confirm the data pipeline and script work end to end. Then do the full run on one A100-large or L40S. Cap total spend at 12 USD and ask me before exceeding it. Tell me the estimated cost and GPU choice before launching the full run.

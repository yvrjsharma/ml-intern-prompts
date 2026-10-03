Train an image-editing LoRA for `Qwen/Qwen-Image-2.1` that takes ONE photo with a magenta scribble drawn on it plus a short text naming an object. It returns the same photo with the scribble replaced by that object, following the scribble's shape and pose, lit consistently with the scene, and with everything outside the scribble's neighbourhood unchanged. Prove it with numbers against the base model, at 40 steps and with the Viggle 6-step turbo LoRA.

**Verified facts, do not re-derive.**
- Qwen-Image 2.1 is one model for generation and editing; it edits when given `image=`. Inference: `QwenImage21Pipeline` from diffusers main, 40 steps, no CFG (`true_cfg_scale=1.0`), `transformers>=5.17`. The model card says local edits can be specified "via circles, painted annotations, or separate masks", so the base model may already partly do this.
- **Trainer: ostris/ai-toolkit, `arch: qwen_image_2`.** A config that has already worked for an edit LoRA on 2.1 (our Orbit Alpha run: 2,000 steps at 768 px, about 2.2 s/step on one A100 80 GB):
  - `sd_trainer`, LoRA `linear: 32`, `linear_alpha: 32`
  - dataset `folder_path` (targets) + `control_path` (inputs) + `caption_ext: txt`, `resolution: [768]`, `cache_latents_to_disk: true`
  - `optimizer: adamw8bit`, `lr: 1e-4`, `batch_size: 1`, `timestep_type: weighted`, `noise_scheduler: flowmatch`, `gradient_checkpointing: true`, `dtype: bf16`
  - `cache_text_embeddings: true`, `skip_first_sample: true`, `disable_sampling: true`
  - `model.name_or_path: Qwen/Qwen-Image-2.1`, `quantize: false`, `quantize_te: false`, `model_kwargs.match_target_res: true`, `save.dtype: float16`, `save_every: 500`

  Captions go next to the target images. Leave RGBA off: this is a photo task.
- **Known ai-toolkit bugs for 2.1 (open as of 26 Sep).**
  - Issue 1059: sampling during training crashes when text embeddings are cached and control images are used ("prompt embeddings hold N reference image slots…"). Keep `disable_sampling: true` and run all evaluation separately.
  - PR 1067: loading from a *local* diffusers directory can crash with "Cannot copy out of meta tensor". Load from the Hub id. If it still happens, apply the PR's one-line fix (`low_cpu_mem_usage=False` in `aitk_from_pretrained`).
  - Caching text embeddings for edit pairs runs Qwen3-VL over every control image. One user reported about 46 minutes for 2,364 items. Count this in the training budget.
- **Training through the turbo LoRA is not possible here.** ai-toolkit's `assistant_lora_path` is only loaded by the legacy model class. The `qwen_image_2` arch never loads it. Train on the base model and *measure* the turbo stack.
- **Viggle turbo recipe (runs in a live Space).** Repo `Viggle/Qwen-Image-2.1-viggle-turbo`, file `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors`. Settings: 6 steps, `sigmas=[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]`, `true_cfg_scale=1.0`, scheduler `FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift_terminal=None)`. Stack both adapters with `pipe.set_adapters([ours, "turbo"], [1.0, 1.0])`.
- **diffusers loading lesson from Orbit Alpha.** ai-toolkit saves ComfyUI-style keys. diffusers loads them but **silently drops the fused `img_mlp.gate_up` LoRA** (diffusers stores it as two linears). Ship a converted `_gate_up_split` file too, and verify it loads with zero dropped keys and gives the same outputs as the original. Pin the diffusers commit you verified with (Orbit used `0121a91f9d419ff7234c8a5923f82c244e6f1914`).
- **Data: Open Images V7 segmentation masks** (annotations CC BY 4.0; images listed as CC BY 2.0, with per-image author and URL in the image metadata CSV).
  - Test split: `https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv` (2.9 MB) and `validation-masks/validation-masks-{0-9,a-f}.zip`
  - Train split: `.../v5/train-annotations-object-segmentation.csv` (807 MB) and `train-masks/train-masks-{0-9,a-f}.zip` (about 385 MB each)
  - CSV columns: `MaskPath, ImageID, LabelName, BoxID, BoxXMin, BoxXMax, BoxYMin, BoxYMax, PredictedIoU, Clicks`. Class names come from `.../v7/oidv7-class-descriptions.csv`.
  - Images download without auth from `https://s3.amazonaws.com/open-images-dataset/{train|validation}/<ImageID>.jpg`.
  - One or two train mask shards are enough. Do not download the whole set.
- **Hole filling for inputs: LaMa**, either `pip install simple-lama-inpainting` or the ONNX `opencv/inpainting_lama` (Apache 2.0). It only makes inputs, so its artifacts are acceptable.
- License: Qwen Research License, non-commercial. Derivatives may be shared with the license text, a change notice and "Built with Qwen".

**Marker convention (fixed, so the Space can reproduce it exactly).** The scribble is a single colour, pure magenta RGB (255, 0, 255). This is chosen to differ from the object remover's red (239, 68, 68) so both LoRAs can share one Space. Stroke width is 0.4% to 1.0% of the image's short side, drawn opaque with round caps directly on the image. The prompt is `<doodle> Turn the magenta scribble into {caption}.` A 50/50 mix of captions:
- the Open Images class name ("a dog")
- a short description of the object crop from a small VLM through Inference Providers, e.g. "a brown leather armchair"; check which VLMs are live, and use class names only if none are

**Step 1. Build pairs (CPU job, or a small GPU if LaMa on CPU is too slow).**
- **Mask filters:** object area 3% to 35% of the image; mask does not touch the image border (untruncated); one connected main component.
  - Inspect `PredictedIoU`; if it is populated, keep masks at 0.6 or above.
  - Exclude classes that are parts of or worn by other objects: human body parts and faces, clothing, footwear, wheels, tyres, and similar. Removing them leaves unnatural holes.
  - Cap any single class at 6% of pairs.
  - Hold out 20 whole classes spread across animals, furniture, vehicles, food and household objects, to test unseen categories.
- **Input:** dilate the mask by 3% of the object's bounding-box diagonal, extended further downwards so contact shadows below the object are mostly removed. Fill with LaMa. Then draw the scribble. The **target** is the original photo, resized to match.
- **Three scribble styles:**
  - 50% rough outline: the mask contour simplified with Douglas–Peucker at a random tolerance, then wobbled with smooth noise and given occasional small gaps and overshoots
  - 30% outline plus a few interior strokes taken from a Canny map of the object inside the mask, sparsified
  - 20% lazy blob: a jittered ellipse or convex hull. These users care about placement, not shape.
- **Size:** about 5,000 train pairs from Open Images train. The test set is 160 pairs from Open Images **validation**, including about 40 from the held-out classes. Store the scribble polyline, the style, the dilated mask and the LaMa region with every pair, because the metrics need them.
- **Save the scribble drawer** as one standalone `scribble.py` in the dataset repo. The Space will import it.

**Step 2. Baselines first, on the 160 test pairs.**
- (a) Base 2.1 with the same marked input and a plain instruction: "Replace the magenta scribble with {caption}, following its shape and matching the photo's lighting. Remove all magenta lines."
- (b) Base 2.1 with our `<doodle>` prompt.
- (c) Both of the above with the Viggle turbo LoRA at 6 steps.
- (d) *Optional, only if it sets up in under an hour and costs under 1.5 USD:* `alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union` through VideoX-Fun (`config/qwenimage21/qwenimage21_control.yaml`, `control_context_scale` 1.0, 40 steps). Inputs: the scribble as a Scribble control image and the filled scribble region, dilated, as the inpaint mask. Skip it and say so if not.

Record seconds per edit and peak memory for each.

**Step 3. Train.** Use the config above with the magenta-marked inputs as `control_path` and the real photos as targets.
- Smoke test first: 50 steps. Load the saved adapter, **assert that the `lora_B` tensors are non-zero**, and check that one training pair's output visibly changed compared with the base model. Report seconds per step and the text-embedding caching time.
- Then 2,000 to 2,500 steps at 768 px, checkpoints every 500, loss logged to Trackio. If the measured speed allows 2,000 steps at 1024 px inside 7 USD including caching, use 1024 instead.

**Step 4. Evaluate, inside a 5 USD cap.** Checkpoint selection uses a fixed 48-pair subset at 20 steps. The final comparison runs all 160 test pairs at 40 steps and at turbo 6 steps, for the chosen checkpoint and baselines (a) to (c). Fixed seeds. Metrics per edit:
- **Object present:** OWLv2 (`google/owlv2-base-patch16-ensemble`) queried with the class name. A hit means the score is above threshold and the box IoU with the scribble's bounding box is at least 0.3. Report the hit rate.
- **Shape fidelity:** SAM 2.1 prompted with the scribble's bounding box on the output. Report mask IoU against the filled scribble region, and against the ground-truth object mask.
- **Ink removed:** the fraction of the scribble's stroke pixels still close to magenta in the output. It should be near zero; leftover ink is the most visible failure.
- **Background preserved:** PSNR and LPIPS against the input, outside the union of the dilated scribble region and the LaMa hole.
- **Realism:** a blind, randomised-order pairwise judgement with a VLM through Inference Providers, LoRA against base on the same input, asking which edit looks more naturally part of the photo (lighting, shadow, scale). Run it on 100 pairs.
- **Speed:** seconds per edit at 40 and 6 steps.

Break results down by scribble style (outline, detailed, blob), object size, and **seen versus held-out classes**. Pick the checkpoint with the best hit rate that does not lose background PSNR.

**Step 5. Out-of-domain check (about 1 USD).** Training inputs always contain a LaMa-filled patch under the scribble; real users draw on clean backgrounds. Generate 12 clean scene photos with the base model's text-to-image mode (a kitchen counter, a beach, a living room, a park bench, a desk and so on). Draw scribbles on empty areas with `scribble.py`, using outlines taken from masks of other images, and add a few absurd requests ("a small dinosaur", "a vintage robot"). Run the LoRA and base at 40 and 6 steps and publish contact sheets. Say honestly where it breaks, especially:
- missing or wrong shadows
- objects drawn at the wrong scale for the scene
- the model redrawing background it should have kept

**Deliverables.**
1. LoRA repo `ML-Intern-lab/Qwen-Image-2.1-doodle-in-LoRA` with a real README covering:
   - the exact marker convention (colour, stroke-width range, prompt template), so a Space can reproduce the training inputs
   - the evaluation table against all baselines at 40 and 6 steps, with the breakdowns
   - contact sheets, limitations, license text, change notice and "Built with Qwen"
   - attribution to Open Images (CC BY 4.0 annotations; CC BY 2.0 images with per-image credits in the dataset), LaMa, and the object-remover LoRA for the marker idea

   Confirm at the end that README.md is actually in the repo.
2. **Verify loading in diffusers** at a pinned commit with `QwenImage21Pipeline.load_lora_weights`. Ship the `_gate_up_split` diffusers-keyed file and the original. Include a snippet you actually ran for both 40-step and turbo (stacked adapters, sigmas, scheduler).
3. Dataset repo `ML-Intern-lab/doodle-in-pairs` containing:
   - inputs, targets, captions, scribble polylines and styles, masks, and the train/test split with held-out classes
   - per-image Open Images credits (author, original URL, license)
   - the build script and `scribble.py`
4. No Space; I will build it.

**Budget.** Smoke test first: 200 pairs built, 50 training steps with the `lora_B` check and a visible-change check, 4 edits per baseline, seconds per step, per edit and for caching measured. Expected costs:

| Item | Expected cost |
|---|---|
| Pair building | under 1 USD |
| Smoke test | about 1.5 USD |
| Training including text-embedding caching | at most 7 USD |
| Evaluation including baselines | at most 5 USD |
| Out-of-domain check | about 1 USD |
| Optional ControlNet baseline | at most 1.5 USD |

Cap the total at 19 USD. Report the measured timings and the final plan before the full training job, and cut steps, resolution or evaluation counts rather than exceed the cap.

Fine-tune the 4-step student in `ysharma/agate-002-4step` (unofficial distillation of `Logolabs/agate-preview-002`,
MIT) so its 4-step images get closer to the teacher's and sharper. Keep the architecture, the sampler and the file
format unchanged so the model repo, both Spaces and the WebGPU export keep working. First measure precisely where we
are, with confidence intervals, because run 1's numbers were too noisy to separate the students.

**Verified facts from run 1, do not re-derive.**
- **Run 1's working scripts are in `hf://models/ysharma/agate-002-distill-artifacts/scripts/run1/`**: `cache.py`,
  `train.py`, `gen.py` (batched eval sampler, proven identical to the single-prompt pipeline path), `metrics.py`
  (FID-5k, FD-DINOv2, CLIPScore, LPIPS), `geneval_run.py` (official GenEval scorer), `export_agate.py` (WebGPU ONNX
  export + parity), `verify.py` (clean-env check). **Start from these rather than rewriting them.** Each has a
  `MODE`/`STAGE` constant edited per submission.
- **Models** (all `FCDMThinker2(**config["model_kw"])`, bf16 `generator.safetensors`, identical keys and shapes):
  teacher `Logolabs/agate-preview-002`; in `ysharma/agate-002-distill-artifacts`: `g/` (CFG baked in, run at 50
  steps), `s16/`, `s8/`, `s4/` (= the shipped 4-step model). Load the code with
  `sys.path.insert(0, snapshot_download("Logolabs/agate-preview-002"))`. Call signature `model(z, t, ctx, mask) ->
  velocity`; t = 0 is noise and t = 1 is data; uniform Euler `z <- z + v / steps` with `t_i = i / steps`; teacher CFG
  3.0 against the empty prompt as a batch of 2; students use no CFG. `torch.backends.cuda.enable_cudnn_sdp(False)`
  always. SD-VAE `stabilityai/sd-vae-ft-mse`, decode `vae.decode(z / 0.18215)`.
- **Noise convention (this is what makes pairs line up):** each image's starting noise is
  `torch.randn(1, 4, 32, 32, device="cuda", generator=torch.Generator("cuda").manual_seed(seed))`, drawn per image
  (not one batched draw). GenEval seeds = `prompt_id`; holdout seeds = row index.
- **Training traps run 1 hit, keep the fixes:**
  - keep **fp32 master weights** under bf16 autocast (at lr 1e-5 AdamW updates vanish in bf16 weights);
  - pad text to the longest caption in the batch, not the 512 bucket (memory);
  - use the model's own `grad_checkpoint` flag;
  - load `train/train.parquet` from `ysharma/flux-reason-agate-latents-150k` **directly with pyarrow** (the repo's
    two splits have mismatched schemas and `datasets` fails);
  - missing captions are the string `"None"`;
  - lpips 0.1.4 needs the torchvision `alexnet(pretrained=)` shim that is in `train.py`.
- **Measured throughput on a100-large (run 1):**
  - teacher at 50 steps with CFG: ~3.3 images/s batched, including VAE decode and upload;
  - teacher at 16 steps with CFG: ~8.5 images/s;
  - 4-step student: ~23 images/s;
  - PD training at batch 128: 2.1 s/step (1 student forward+backward plus 2 teacher forwards).
- **Run 1 results** (1 image per prompt, so each GenEval number is about ±0.02):

  | | GenEval | FID-5k | FD-DINOv2 | LPIPS vs teacher@50 | CLIP |
  |---|---|---|---|---|---|
  | teacher@50 | 0.562 | 11.40 | 130.8 | – | 27.03 |
  | teacher@16 | 0.569 | 11.90 | 130.0 | 0.196 | 26.74 |
  | teacher@4 | 0.466 | 20.07 | 218.2 | 0.416 | 26.62 |
  | G@50 | 0.507 | 11.89 | 131.9 | 0.218 | 26.39 |
  | s4 | 0.527 | 12.96 | 146.5 | 0.307 | 26.25 |

  The images, raw scorer output and metrics are in `ysharma/agate-002-eval-images` (`geneval/<config>.parquet`,
  `holdout5k/<config>.parquet`, `reference/holdout.parquet`, `results/*.json`).
- **GenEval:** prompts `load_dataset("Jialuo21/GenEval", split="test")` (553). The official scorer only runs in its
  own env: `geneval_run.py` builds it (Python 3.11 venv, torch 2.1.2, mmcv-full 1.7.2 cu121, mmdet 2.x editable,
  Mask2Former Swin-S). It worked on a10g-small, about 0.3 s per image. Never use an LLM-judged GenEval.

**Step 0. Measure where we are, properly (about 2 USD).**
- Score GenEval with **4 images per prompt** (2,212 images) for teacher@16, s4 and **G@4**.
  - G@4 is the CFG-baked model run at 4 steps with no step-halving training; run 1 never measured it.
  - Seeds `prompt_id * 4 + k` for k = 0..3; reuse run 1's teacher@50 images as k = 0 and add k = 1..3 for it too.
- Report every overall and per-task score with a **95% paired bootstrap interval over prompts**, and for each pair of
  configs the paired difference with its interval.
- Also report LPIPS to teacher@50 and FID-5k for G@4 (the others exist).
- Report this table before training. It decides the result's baseline, and it may show s4 is already level with G@4
  (which would mean progressive distillation added little at 4 steps).

**Step 1. Teacher pairs (about 2 USD).**
- Take 24,000 captions from the **train** split with run 1's caption mix: 80% one short caption weighted toward
  `caption_composition`, 20% `caption_detail`.
- **Exclude** every GenEval prompt, and every caption from the 5k holdout, by exact string match.
- Seeds 10,000,000 + row index, drawn with the per-image noise convention above.
- For each (caption, seed), run the **teacher at 16 steps with CFG 3**. It matches 50 steps on GenEval and is about
  2.6× cheaper. Store the final **latent** (fp16, 4×32×32), not a decoded image.
- Batch as large as memory allows.
- Push `ysharma/agate-002-4step-pairs-24k` (private): caption, seed, latent.
- **Check before generating the rest:** 8 held-out GenEval prompts regenerated this way must reproduce run 1's stored
  teacher16 images (max |pixel diff| ≤ 2 after decode). This proves seeds and noise line up.

**Step 2. Fine-tune (about 5 USD).**
- Initialize from `s4/` (fp32 master copy); the teacher is not needed during training.
- For a batch of (caption, seed, target latent): regenerate the same starting noise, run the student's full **4-step
  Euler sampler with gradients** through all four steps (gradient checkpointing), and get the predicted final latent
  x̂.
- Loss = **1.0 × latent MSE(x̂, x_target)** + **λ_p × LPIPS(dec(x̂), dec(x_target))** + **λ_d × (1 − cosine) of
  DINOv2-small CLS features** of the two decodes.
  - `dec` is the **TAESD** decoder (`madebyollin/taesd`, 1.2M params, differentiable and cheap), used for the loss
    only. Every metric uses the real SD-VAE.
  - Start at λ_p = 1.0, λ_d = 0.5, and let the smoke test adjust them so the three terms are within 10× of each other
    at step 0.
- AdamW lr 5e-6, betas (0.9, 0.95), grad clip 1.0, EMA 0.995 (much shorter than run 1's 0.999, whose stages ended
  with 20–30% of the weights still on the previous stage), bf16 autocast, batch 32–64 as memory allows.
- Every 250 steps, on 64 fixed held-out pairs (not in training) and 32 GenEval prompts at seed = `prompt_id`, log:
  - LPIPS to the target;
  - LPIPS to teacher@50;
  - a 16-prompt grid.
- Stop early when held-out LPIPS stops improving by at least **1% relative** over 2 checks.
- Plan for about 3 passes over the 24k pairs if the budget allows, and size it from the smoke test.
- Push `s4p/generator.safetensors` (bf16 EMA weights, same keys and shapes as the teacher; assert both) to the
  artifacts repo.

**Step 3. Evaluate (about 3 USD).**
- For **s4p**, same protocol as Step 0: GenEval with 4 images per prompt and paired intervals against s4, G@4,
  teacher@16 and teacher@50.
- Also FID-5k, FD-DINOv2, CLIPScore and LPIPS to teacher@50 on run 1's holdout set and seeds.
- Batch-1 latency must be unchanged (same network and step count).
- Look for over-sharpening or artifacts: put the 16-prompt grids for teacher@50 | s4 | s4p side by side, and
  describe what changed honestly.

**Ship gate.** Update the shipped model only if s4p is:
- no worse than s4 on paired GenEval (difference ≥ −0.01);
- better on LPIPS to teacher@50;
- no more than 0.5 worse on FID-5k.

If it passes:
- **in `ysharma/agate-002-4step`:**
  - move the current `generator.safetensors` to `v1/generator.safetensors`;
  - put s4p's weights in `generator.safetensors`;
  - regenerate `webgpu/generator.onnx` and its `manifest.json` entry (real bytes and sha256) with `export_agate.py`,
    and pass its parity check (≤ 5/255 per pixel after decode, 4 steps, same noise);
  - update the README with the new table, the grids and a changelog line;
- **then:**
  - run `verify.py` from a clean env against the repo;
  - confirm the model still loads with `AgatePipeline.from_pretrained("ysharma/agate-002-4step")(prompt, steps=4,
    cfg=1.0)`.

If it fails the gate, leave the shipped repo untouched, keep s4p in the artifacts repo, and explain why.

**Deliverables.**
1. The Step 0 table (with intervals) and the Step 3 table, as `results/run2.json` in `ysharma/agate-002-eval-images`.
2. `s4p/` weights, logs and grids in `ysharma/agate-002-distill-artifacts`, plus this run's scripts under
   `scripts/run2/`.
3. The pairs dataset `ysharma/agate-002-4step-pairs-24k` (private).
4. If the gate passes, the updated `ysharma/agate-002-4step` (weights, `v1/`, WebGPU export, README), verified from a
   clean env. No Spaces.

**Budget.** Smoke test first:
- 256 pairs;
- 50 fine-tuning steps with the weight-change assert (student weights differ from s4, one fixed-seed image changed);
- measured seconds per step and pair-generation images/s.

Rough plan: Step 0 2 USD, pairs 2 USD, smoke 1 USD, fine-tune 5 USD, evaluation 3 USD, export and verify 1 USD. **Cap
the total at 15 USD.** Report measured timings and the final plan before the long fine-tune; cut passes or pair count,
not the evaluation protocol, to stay inside the cap. Use a100-large for GPU work and a10g-small for the GenEval scorer
as in run 1.

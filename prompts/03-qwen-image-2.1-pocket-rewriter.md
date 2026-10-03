Distill the official Qwen-Image 2.1 text-to-image prompt rewriter into a tiny model that anyone can run on a CPU.

**Why.** Qwen-Image 2.1 gives its best results when short user requests are first expanded by `Qwen/Qwen-Image-2.1-PE-T2I`. That rewriter is a 9B thinking model: about 20 GB in bf16, a 1,700-word system prompt (`system_prompt.txt` in its repo), thinking mode required, and up to 16,256 new tokens, all to emit one JSON object `{"rewritten_prompt": "...", "wh_ratio": "3:2"}`. People running the image model on a 16 to 24 GB GPU cannot host it alongside, and the rewrite often takes longer than the image. The only derivatives on the Hub are quantized copies of the same 9B. Goal: a 0.8B and a 2B student that take ONLY the user's short request, with no system prompt and no thinking, and return the same JSON directly.

**Teacher facts (verified, do not re-derive).**
- Architecture `Qwen3_5ForConditionalGeneration`; load as in its model card; the T2I rewriter takes text only.
- Use the official code: `prompt_rewrite/run_vllm.py --task t2i` from `github.com/QwenLM/Qwen-Image-2.1`. It handles the system prompt, thinking, sampling (temperature 1.0, top_p 0.95, top_k 20) and JSON parsing, and writes `thinking`, `rewritten_prompt`, `wh_ratio`, `parse_ok`.
- Input is a short request in any language. Output is one long English paragraph describing the finished image, plus an aspect ratio. The rewriter must copy any text the user wants shown character for character, in the original script.
- Allowed ratios per the system prompt: 3:2 and 2:3 are the defaults, then 1:1, 16:9, 9:16, 1:2, and rarely 3:4, 4:3, 2:1, 21:9, 9:21, 4:5, 5:4, 3:1, 1:3.

**Step 1. Build 10,000 short requests.** Generate them with a cheap instruct model through Inference Providers, from a taxonomy so coverage is deliberate: photo, illustration, poster, logo or icon, infographic, product shot, portrait, UI or screen, sticker, scene. Mix lengths from 3 to 40 words. About 35% must ask for specific text in quotes (signs, titles, labels), 15% must state an aspect ratio or orientation, 10% must include a meta-instruction such as "make the text sharp", and 30% must be non-English, spread over Chinese, Hindi, Spanish, Arabic, Japanese, French and German. Deduplicate. Hold out 300 for evaluation, stratified the same way.

**Step 2. Label with the teacher.** Run the official vLLM script on one `a100-large` with prefix caching on, since every request shares the long system prompt. Keep the thinking text in the dataset but do not train on it.

**Step 3. Filter hard.** Keep a row only if: `parse_ok` is true; `wh_ratio` is in the allowed set; every quoted string from the request appears verbatim in `rewritten_prompt`; a ratio stated by the user is respected; the prompt is English and between 80 and 400 words; and no ratio, resolution or pixel count appears inside the prompt text. Report how many rows each rule removed.

**Step 4. Train two students with TRL SFT.** `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B`, both Apache 2.0. Full fine-tuning, text only, bf16, 2 epochs, learning rate 1e-5, cosine schedule, effective batch 32, max length 1024. Chat format: the user turn is the raw request, the assistant turn is the compact JSON. No system prompt and thinking disabled in the chat template, so the instructions are baked into the weights. Log to Trackio.

**Step 5. Evaluate on the 300 held-out requests. Report teacher, both students, and the untuned 2B given the full system prompt as a baseline.**
- Valid JSON rate and allowed-ratio rate.
- Text fidelity: share of requests whose quoted strings all survive verbatim, broken down by script.
- Ratio agreement with the teacher.
- Cost: mean generated tokens, latency per rewrite on GPU, and for the 0.8B also on a CPU-only machine.
- Image-level check on a 40-request subset: render with `QwenImage21Pipeline` (diffusers from main, 40 steps, no CFG, fixed seed, size from `wh_ratio` scaled to about 1 megapixel) for four arms: raw request, teacher rewrite, 0.8B rewrite, 2B rewrite. Score with OCR word accuracy on the text requests and a pairwise VLM judge for overall preference. Publish the contact sheets. The honest question is whether the students keep most of the teacher's gain over the raw request.

**Deliverables.**
1. Models `ysharma/image21-pocket-rewriter-0.8B` and `-2B`, each with a model card: usage snippet that plugs into `QwenImage21Pipeline`, the eval table, limitations, and a GGUF Q8 export of the 0.8B.
2. Dataset `ysharma/image21-rewriter-distill` with request, teacher thinking, teacher answer, filter flags and split.
3. A Gradio Space `ysharma/image21-pocket-rewriter` on CPU: type a request, get the rewritten prompt and ratio from the 0.8B student, with a copy button and 12 examples across languages. No image generation in the Space.
4. Licensing: the teacher is under the Qwen Research License, non-commercial. State that the students were trained on its outputs, mark all three repos non-commercial research use, include the license text and the words "Built with Qwen", and do not use "Qwen" as the primary repo name.

**Budget.** Smoke test first: 200 requests through the whole pipeline on a cheap GPU, including one tiny training run and 4 rendered images. Then the full run. Expected: teacher labelling about 2 to 2.5 hours on `a100-large`, two SFT runs under 1 hour total on `l40sx1` or `a10g-large`, image eval under 1 hour on `l40sx1`. Cap total spend at 18 USD, give me the estimate before the full labelling job, and if labelling runs long, cut to 7,000 requests instead of exceeding the cap.

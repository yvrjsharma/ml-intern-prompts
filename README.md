# ML-intern prompts

These are the briefs I pasted into [ML-intern](https://huggingface.co/chat/) in HuggingChat in September 2026. Each one became a public model on the Hugging Face Hub. Every file in `prompts/` is the opening message of one run, as I wrote it before pasting it in.

| # | Prompt | Result | Words | Compute |
|---|---|---|---|---|
| 1 | [Citrus Doctor](prompts/01-citrus-doctor.md): fine-tune a small VLM to diagnose citrus diseases | [model](https://huggingface.co/ML-Intern-lab/citrus-disease-vlm) · [Space](https://huggingface.co/spaces/ML-Intern-lab/citrus-doctor) | 356 | $1.90 |
| 2 | [Huggy LoRA](prompts/02-huggy-flux2-klein-lora.md): a character style LoRA on FLUX.2 klein 4B | [model](https://huggingface.co/ML-Intern-lab/huggy-flux2-klein-lora) · [Space](https://huggingface.co/spaces/ML-Intern-lab/huggy-generator) | 860 | $7.60 |
| 3 | [Pocket rewriter](prompts/03-qwen-image-2.1-pocket-rewriter.md): distil the 9B Qwen-Image 2.1 prompt rewriter into 0.8B and 2B models | [0.8B](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-0.8B) · [2B](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-2B) · [Space](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-pocket-studio) | 891 | $16.05 |
| 4 | [Viewpoint Orbit](prompts/04-qwen-image-2.1-viewpoint-orbit-lora.md): a camera-angle edit LoRA with transparent output | [model](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA) · [Space](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA) | 1,304 | $16 |
| 5 | [Doodle-in](prompts/05-qwen-image-2.1-doodle-in-lora.md): replace a scribble on a photo with a named object | [model](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-doodle-in-LoRA) · [Space](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-doodle-in-LoRA) | 1,873 | $24.30 |
| 6 | [Agate 4-step, run 1](prompts/06-agate-4step-distillation.md): distil a 260M text-to-image model from 50 steps with CFG to 4 steps | first 4-step student, refined in run 7 | 1,752 | $37 for runs 6 and 7 |
| 7 | [Agate 4-step, run 2](prompts/07-agate-4step-sharpen.md): re-measure run 1, then fine-tune the student against the teacher's images | [model](https://huggingface.co/ML-Intern-lab/agate-preview-002-4step) · [Space](https://huggingface.co/spaces/ML-Intern-lab/agate-preview-002-4step-webgpu) | 1,469 | see run 6 |

Compute is the GPU and CPU job charges each ML-intern session reported.

## How the briefs are built

Read them in order and you can see the format settle. Most of the later ones follow the same pattern:

1. **The idea in one line**, and what the finished thing should do.
2. **Verified facts, do not re-derive.** Anything I had already checked: exact model and dataset ids, API signatures, trainer configs that worked before, and known bugs in the tools. This keeps the agent from spending its budget rediscovering what I know.
3. **A baseline before training.** The base model is scored on the same metric first, so the result shows the gain and not just a number.
4. **A smoke test with a check.** For example, train 50 steps and confirm the saved weights changed before paying for the full run.
5. **Evaluation**, defined precisely: the split, the metric and how it is computed.
6. **Deliverables**: what goes on the Hub and what the model card must contain.
7. **Budget**: a hard cap, plus "ask me before exceeding it."

## Notes

- The prompts name repos under `ysharma/`, my personal account at the time. The finished models, datasets and Spaces were later moved to the [ML-Intern-lab](https://huggingface.co/ML-Intern-lab) organisation, so the links above point there.
- The facts in each brief were true when it was written, in September 2026. Libraries and open issues change, so check them again before you reuse a brief.

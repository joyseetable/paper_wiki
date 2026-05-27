---
type: task
---
# Fine-Grained Visual Recognition (FGVR)

A long-standing computer vision challenge: distinguishing subtle differences among visually similar categories, such as bird species (200+ kinds of warblers), car models (Audi A5 vs. S5 Coupe 2012), or aircraft variants (Boeing 737-200 vs. 737-300). Difficult even for humans — requires extensive domain knowledge to resolve minimal inter-class variance and large intra-class variation.

## Relationship to HVR
FGVR focuses on the leaf-node discrimination problem (telling apart fine-grained subcategories). [[HVR]] adds the requirement of predicting a correct hierarchical path. FGVR is a prerequisite for HVR — if you can't tell leaf nodes apart, you can't get the hierarchy right.

## Key Challenge for MLLMs
General-purpose MLLMs (GPT-4, Gemini, Qwen) underperform contrastive CLIP models on FGVR, despite CLIP being used as the vision encoder inside those same MLLMs. The gap stems from knowledge *deployment* — MLLMs have the visual features and factual knowledge but can't effectively use them for discriminative tasks.

## Key Benchmarks
- CUB-200 (Birds), Stanford Cars-196, Stanford Dogs-120, Flowers-102, Oxford Pets-37, FGVC-Aircraft

## Papers Using This
- [[Fine-R1-fine-grained-recognition]]: CoT SFT + TAPO, 4-shot SOTA surpassing CLIP models
- [[TARA-taxonomy-aware-alignment]]: BFM-guided representation alignment for HVR (FGVR + hierarchy)

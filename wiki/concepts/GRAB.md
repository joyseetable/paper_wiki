---
aliases: [Guided Representation with Attentive Bank]
related_papers: [[ITSELF-attention-guided-alignment]]
---
# GRAB (Guided Representation with Attentive Bank)

An attention-driven local alignment branch for vision-language models. Instead of using external supervision (parsing networks, MLLMs) to find discriminative image regions, GRAB harvests the model's own attention maps to select high-saliency tokens from both image and text modalities, then applies local objectives on this "Attentive Bank."

The core insight: CLIP's attention maps already surface spatially precise evidence from the earliest training epochs — GRAB simply extracts and optimizes over these signals.

## Key Papers
- [[ITSELF-attention-guided-alignment]]: Introduces GRAB with MARS and ATS for TBPS

## Relationships
- builds on: CLIP's multi-head self-attention, implicit local alignment approaches
- contrasts with: explicit local alignment (parsing networks, MLLM captioning), masked-modeling alignment, fully implicit local losses
- components: [[MARS]], [[ATS]]

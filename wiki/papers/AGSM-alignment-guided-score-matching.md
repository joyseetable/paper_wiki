---
paper_id: 2605.30038
title: "Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models"
authors: "Jaa-Yeon Lee, Yeobin Hong, Taesung Kwon, Jong Chul Ye"
year: 2026
venue: arxiv
status: skimmed
confidence: medium
tags: [diffusion-models, T2I, text-image-alignment, score-matching, soft-tokens, preference-learning]
---
# Summary
A reward-free, lightweight post-training method that improves text-image alignment in diffusion models by integrating contrastive alignment guidance directly into the score-matching objective via a Plackett-Luce preference model, using only 1.8M trainable soft tokens while keeping the backbone frozen.

## Key Takeaways
- **Explicit negative guidance prevents off-manifold drift**: SoftREPA's contrastive loss maximizes denoising error for negative pairs, causing unbounded divergence. AGSM assigns bounded preference directions to negative samples, preventing failure cases like overcounting (+35% counting accuracy on GenEval over SoftREPA).
- **Plackett-Luce generalizes pairwise Bradley-Terry**: Multi-candidate preference formulation naturally handles in-batch negatives; PL consistently outperforms BT across all alignment metrics.
- **Complementary to RL-based DPO methods**: Adding AGSM soft tokens to Diffusion-DPO/SPO/InPO consistently improves performance — representation alignment + preference alignment are orthogonal.
- **Training stability**: SoftREPA's validation ImageReward degrades after early peak despite decreasing loss; AGSM remains stable throughout.

## Method
(1) Derive alignment reward from diffusion denoising error as implicit reward; (2) Formulate text-image alignment as Plackett-Luce preference; (3) Derive target score for positive and negative pairs via classifier-free-guidance-style tilting; (4) Train separate soft tokens (ψ+, ψ−) to match the guided target score. At inference, use only ψ+ (drop ψ−). Backbone fully frozen. Works on SD1.5, SDXL, SD3 (UNet and transformer).

## Results
- SD3 COCO-val5K: ImageReward 94.27 → 103.3, CLIP 26.30 → 27.00
- GenEval counting: 0.29 (SoftREPA) → 0.64 (AGSM), +35% improvement
- Image editing: Pareto front across CLIP vs. LPIPS trade-off
- Training stable without early-stopping (vs. SoftREPA requiring heuristic early-stopping)

## Limitations & Open Questions
- Only tested on SD-family backbones
- [[gaps/questions]]

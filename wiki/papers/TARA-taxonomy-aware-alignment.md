---
paper_id: TARA (CVPR 2026)
title: "Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models"
authors: "Hulingxiao He, Zhi Tan, Yuxin Peng (Peking University)"
year: 2025
venue: CVPR 2026
status: read
confidence: high
tags: [HVR, LMM, taxonomy, representation-alignment, biology, fine-grained, novel-categories]
---
# Summary
Aligning LMM intermediate representations with Biology Foundation Model (BFM) embeddings — at both the visual feature level and the answer-token level — injects taxonomic structure into LMMs, substantially improving hierarchical consistency and enabling recognition of novel species never seen in training.

## Key Takeaways
- **Two-level alignment works better than one**: Visual alignment (LMM visual features → BFM vision encoder) + Label alignment (LMM first answer token → BFM text-encoded label). The combination improves HCA by 3.55pp on iNat-Plant over No-Thinking RFT alone.
- **"All visual tokens" > "last visual token" for L_V**; **"first answer token" > "averaged/last question token" for L_C**. Careful feature selection matters — using the wrong token drops performance by ~2pp HCA.
- **Generalizes to novel categories**: On TerraIncognita's novel species split (rare/undiscovered insects), TARA improves Order F1 by +10.15pp and Family F1 by +1.20pp — the taxonomic structure transfers beyond training classes.
- **TARA is a training-only mechanism**: BFMs and projectors are discarded at inference. Zero additional cost at deployment. Training overhead is minimal (lightweight 3-layer MLPs).

## Method
TARA consists of two alignment losses trained alternately with No-Thinking RFT:

1. **Taxonomic Visual Representation Alignment (L_V)**: Cosine similarity loss between projected LMM visual features (layer ℓ) and BFM vision encoder outputs. Uses all visual tokens (not just the last one).
2. **Free-grained Label Representation Alignment (L_C)**: Cosine similarity loss between projected first answer token embedding (layer m) and BFM text-encoded ground-truth label at the specified taxonomy level.

Projectors P_V and P_T are lightweight 3-layer MLPs with SiLU activations. BFM teacher: BioCLIP2. Base models: Qwen3-VL-2B and Qwen2.5-VL-3B.

Training alternates: update LMM+projectors with L_alignment, then update LMM alone with No-Thinking RFT accuracy reward.

## Results
- iNat21-Plant (Qwen3-VL-2B): HCA 6.46 → 12.78 (+6.32), Acc_leaf 30.16 → 32.66 (+2.50)
- iNat21-Animal (Qwen3-VL-2B): HCA 7.18 → 10.26 (+3.08), Acc_leaf 27.86 → 30.77 (+2.91)
- TerraIncognita known species: Order F1 23.30 → 41.56 (+18.26), Family F1 11.47 → 25.47 (+14.00)
- TerraIncognita novel species: Order F1 23.30 → 33.45 (+10.15), Family F1 11.47 → 12.67 (+1.20)
- ImageWikiQA (complex VQA): 48.70 → 51.40 (+2.70) — suggests HVR improvements transfer to broader visual reasoning
- Faster convergence: surpasses baseline early in training

## Limitations & Open Questions
- Only tested in biological domain — hierarchical taxonomies exist in many domains (products, documents, medical) [[gaps/questions]]
- Requires a pretrained BFM with hierarchical contrastive learning — what's the minimum viable BFM? Can a weaker teacher still help?
- The optimal alignment layers (ℓ=14 for visual, m=28 for label) are model-specific — how to determine these without exhaustive search? [[gaps/questions]]
- 1-shot VQA setting is strong but sparse — how does TARA scale with more data?
- No comparison against hyperbolic embedding methods (HiCLIP, HGCLIP) for hierarchical consistency

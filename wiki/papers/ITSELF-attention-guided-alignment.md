---
paper_id: ITSELF (WACV 2026)
title: "ITSELF: Attention Guided Fine-Grained Alignment for Vision–Language Retrieval"
authors: "Tien-Huy Nguyen, Huu-Loc Tran, Thanh Duc Ngo"
year: 2025
venue: WACV 2026
status: read
confidence: high
tags: [TBPS, attention, local-alignment, CLIP, person-reid, fine-grained]
---
# Summary
Encoder attention maps, which surface spatially precise evidence from the earliest training epochs, can be turned into an Attentive Bank for implicit local alignment in text-based person search — no external supervision needed.

## Key Takeaways
- Attention saliency appears early (by epoch 3 the R1 gap between unmasked and attention-retained inputs falls below 1pp), meaning CLIP's internal attention is already a reliable localization cue from the start of training.
- Multi-layer attention fusion (MARS) outperforms single-layer selection because shallow layers fire on low-entropy textures while middle+late layers capture semantic relations — fusing them balances contextual coverage with discriminative focus.
- Coarse-to-fine token scheduling (ATS) matters: retaining more tokens early prevents information loss during unstable optimization, then annealing to fewer tokens sharpens fine-grained discriminative learning.

## Method
Three components built on CLIP ViT-B/16:
1. **GRAB** (Guided Representation with Attentive Bank): Attention-driven local branch that selects high-saliency tokens from both modalities and applies local objectives — no extra supervision, no inference cost.
2. **MARS** (Multi-layer Attention for Robust Selection): Aggregates attention across layers, denoises by discarding lowest-δ fraction, combines via sequential matrix multiplication, then diversity-aware TopK selection across modalities.
3. **ATS** (Adaptive Token Scheduler): Anneals token retention budget from ρ_start=0.65 to ρ_end=0.5 over training steps via exponential decay — preserves context early, focuses on discriminative details late.

Training: Triplet Alignment Loss + Cross-Modal Identity Loss applied to both global [[CLS]] and local GRAB embeddings.
Inference: Similarity = λ_S × S_global + (1-λ_S) × S_local.

## Results
SOTA among CLIP-based methods on all 3 TBPS benchmarks:
- CUHK-PEDES: 76.95 R1 (+1.01), 69.38 mAP
- ICFG-PEDES: 69.23 R1 (+1.55), 43.80 mAP
- RSTPReid: 67.30 R1 (+1.95), 53.05 mAP (+2.17)

Strong cross-dataset generalization (6 transfer settings, all best). Works with both ViT-B/16 and ViT-B/32.

## Limitations & Open Questions
- The method relies on CLIP's attention quality — untested on weaker backbones. What happens when attention maps are noisy or uninformative?
- Only tested on person search. Does attention-guided token selection generalize to other fine-grained VL tasks (e.g., product retrieval, medical imaging)?
- The discard ratio δ=0.25 and layer selection (Middle+Late) are dataset-tuned — how sensitive are these to domain shift? [[gaps/questions]]
- No comparison against explicit methods using MLLM-generated captions (only CLIP-backbone methods compared)

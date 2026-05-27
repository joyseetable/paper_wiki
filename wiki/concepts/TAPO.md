---
aliases: [Triplet Augmented Policy Optimization]
related_papers: [[Fine-R1-fine-grained-recognition]]
---
# TAPO (Triplet Augmented Policy Optimization)

A policy gradient algorithm designed for fine-grained visual recognition (FGVR) that addresses the twin challenges of high intra-class variance and low inter-class variance. Built on top of DAPO (a GRPO successor).

## Two Components

### Intra-class Augmentation
For each anchor image x, sample a positive image x_pos from the same subcategory. Mix n1 anchor rollouts + n2 positive rollouts into one reward pool. Policy update stays conditioned on the anchor only — the positive rollouts just diversify the reward signal. Optimal ratio n1:n2 = 1:1.

This captures broader intra-class variation and provides discriminative guidance when anchor and positive images yield different predictions.

### Inter-class Augmentation
Sample a negative image x_neg from the most visually similar but different subcategory. Add a KL-divergence maximization term: max KL[π_θ(o|x_*) || π_θ(o|x_neg)]. This forces the model to produce *different* outputs for visually similar but distinct categories.

Double entropy regularization (on both π_θ and π_θ^neg) keeps distributions stable and low-entropy.

## Full Objective
Clipped policy gradient (DAPO style) + γ × KL_inter[π_θ || π_θ^neg] − η1 × H[π_θ] − η2 × H[π_θ^neg]

## Key Papers
- [[Fine-R1-fine-grained-recognition]]: Introduces TAPO, 4-shot SOTA on 6 FGVR datasets

## Relationships
- built on: DAPO, GRPO
- contrasts with: [[No-Thinking-RFT]] (suppresses reasoning), CLS-RL (classification RL without contrastive triplet signal)
- part of: Fine-R1 two-stage framework

---
aliases: [Representation Divergence Mutual Reinforcement]
related_papers: [[DIVA-representation-divergence-mutual-reinforcement]]
---
# DIVA (DIvergence-based Mutual Reinforcement)

A self-improved post-training framework for UMMs that transforms the representation divergence between understanding and generation branches into mutual reinforcement. The key insight: middle layers of a shared backbone spontaneously decouple into distinct subspaces for each task — DIVA explicitly factorizes this into shared and unique components and uses mutual information objectives to enable cross-task transfer while preventing interference.

## Two-Stage Pipeline

**Stage 1 — Encoder Warmup** (backbone frozen):
- Train shared encoder E_sh and unique encoder E_uni (3-layer gated MLPs) per information flow
- Cross-task logit injection: understanding logits receive generation's shared factors + own unique factors (and vice versa)
- Orthogonality constraint prevents shared/unique leakage
- Schedule: shared-only warmup → then shared+unique

**Stage 2 — Backbone Fine-Tuning** (encoders frozen):
- Fine-tune only middle layers of the backbone
- Shared alignment: asymmetric InfoNCE (with stop-gradient) maximizes MI between shared factors from two flows
- Unique disentanglement: NCE-CLUB minimizes upper bound on MI between unique factors
- Combined with native understanding and generation losses

## Key Design Choices
- **Asymmetric alignment with stop-gradient**: prevents one task from dominating optimization due to loss scale differences
- **Middle-layer targeting** (layers 8–18): only apply factorization where task-specific divergence is strongest
- **Gated-MLP encoders**: sufficient non-linearity for factorization without Transformer overhead
- **Low-rank logit readouts** (rank r=24): parameter-efficient conditioning

## Key Papers
- [[DIVA-representation-divergence-mutual-reinforcement]]: Introduces DIVA; consistent gains across 3 UMMs on understanding, generation, and editing

## Relationships
- builds on: InfoNCE (contrastive MI estimation), NCE-CLUB (MI upper bound), DAPO/GRPO-style architecture
- contrasts with: [[concepts/TARA]] (external BFM teacher vs. internal MI), RecA (reconstruction alignment vs. factorized MI), UAE (text bottleneck vs. explicit factorization)
- extends: the "don't retrain, redirect" pattern — only post-trains middle layers + lightweight encoders
- applies to: [[entities/UMM]]

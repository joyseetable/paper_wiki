---
type: model-architecture
---
# Unified Multimodal Models (UMMs)

A class of models that handle both visual understanding (image→text) and visual generation (text→image) within a single shared backbone architecture. The fundamental promise is cross-task synergy via parameter sharing, but in practice, the distinct inductive biases of understanding (low-frequency, semantic) and generation (high-frequency, detail-oriented) often lead to mutual impairment rather than reinforcement.

## Core Challenge
Understanding favors semantically discriminative, task-irrelevant-factor-invariant representations. Generation favors high-fidelity, fine-grained representations capable of reconstruction. These complementary but non-equivalent objectives create gradient conflicts in shared transformers — especially in shallow and deep layers.

## Architectures
- **Fully Shared (AR-based)**: Single transformer, discrete visual tokens. e.g., Emu3, Show-o, Liquid, Nexus-Gen
- **Hybrid (AR + Diffusion)**: Separate encoders/decoders for understanding vs generation. e.g., Janus-Pro, BLIP3-o, Bagel, OmniGen2
- **Mixture-of-Transformers**: Separate generation transformer + shared language backbone. e.g., Bagel

## Key Benchmarks
- Understanding: MMMU, MME, MMBench, MMVet, POPE
- Generation: GenEval, DPG-Bench, WISE
- Editing: ImgEdit, GEdit-Bench-EN

## Papers Using This
- [[DIVA-representation-divergence-mutual-reinforcement]]: Factorized MI-based post-training for mutual reinforcement across 3 UMMs

---
paper_id: DIVA (ICML 2026)
title: "DIVA: Harnessing the Representation Divergence in Unified Multimodal Models for Mutual Reinforcement"
authors: "Renjie Lu, Xulong Zhang, Xiaoyang Qu, Shangfei Wang, Jianzong Wang (Ping An Technology + USTC)"
year: 2026
venue: ICML 2026
status: read
confidence: high
tags: [UMM, representation-learning, mutual-information, multimodal, understanding, generation, post-training, factorization]
---
# Summary
The representation divergence between understanding and generation branches in UMMs is not a bug but a feature — by explicitly factorizing middle-layer representations into shared and unique components and applying asymmetric mutual-information alignment, the conflict between inductive biases transforms into mutual reinforcement, yielding consistent gains across both understanding (+7.82%) and generation (+8.46%).

## Key Takeaways
- **Middle layers spontaneously decouple**: Gradient, geometric, and spectral analyses reveal an inverted-U conflict pattern — understanding (low-frequency, semantic) and generation (high-frequency, detail) objectives clash in shallow and deep layers, but middle layers naturally separate into distinct subspaces. This decoupling is the foundation for DIVA.
- **Factorization + MI alignment is the key mechanism**: Shared encoders capture cross-task semantic consensus (aligned via InfoNCE with stop-gradient); unique encoders preserve task-specific inductive biases (disentangled via NCE-CLUB upper bound minimization); orthogonality constraint prevents leakage. Removing unique-information regularization causes consistent degradation on both tasks.
- **Post-training, not retraining**: DIVA is a two-stage post-training framework — Stage 1 trains lightweight gated-MLP encoders + low-rank logit readouts (backbone frozen), Stage 2 fine-tunes only middle layers of the backbone. Works across 3 diverse UMM architectures (Nexus-Gen, Show-o, Liquid).

## Method
**Core insight**: UMM understanding branch favors low-frequency, semantically invariant representations; generation branch favors high-frequency, fine-grained representations. Rather than forcing a compromise, DIVA explicitly factorizes representations in middle layers (where divergence peaks) into:
- **Shared components** (Π_sh): cross-task semantic consensus, aligned via InfoNCE maximization
- **Unique components** (Π_uni): task-specific inductive biases, kept separate via NCE-CLUB upper bound minimization

**Two-stage training**:
1. **Stage 1 — Encoder Warmup** (backbone frozen): Train shared/unique gated-MLP encoders via cross-task logit injection. Understanding logits get injected with generation's shared factors + understanding's unique factors (and vice versa). Orthogonality constraint L_⊥ keeps shared and unique subspaces separate. Schedule: shared-only first, then add unique-residual injection.
2. **Stage 2 — Backbone Fine-Tuning** (encoders frozen): Post-train middle layers (8–18) of backbone using asymmetric InfoNCE alignment (with stop-gradient for stability) + NCE-CLUB unique disentanglement + native task losses.

**Information flows**: Both constructed from the same image-text pair — understanding flow uses captioning supervision, generation flow uses masked inpainting (random mask ratio 0.2–0.6).

## Results
Applied to 3 UMMs (Nexus-Gen 7B, Show-o 1.5B, Liquid 7B):
- **Understanding**: MMMU +5.9, MME +76.2, POPE +3.8 (Nexus-Gen). Show-o: MME +108.4, POPE +6.0.
- **Generation**: GenEval +0.06–0.11, DPG-Bench +2.84–6.57.
- **Image Editing**: ImgEdit overall +0.37, GEdit-Bench-EN overall +0.31 (Nexus-Gen).
- **Ablations**: Base+SFT (same data, no DIVA) shows negligible gains → gains come from the method, not the data. Removing I_uni term causes consistent degradation. Stop-gradient removal causes milder decline.

## Limitations & Open Questions
- Only tested on 1.5B–8B models — scalability to larger models unverified
- Middle-layer range (8–18) empirically chosen based on diagnostic analysis — no principled method for layer selection [[gaps/questions]]
- Only tested on image understanding+generation — video, audio, interleaved modalities unexplored [[gaps/questions]]
- Requires paired data for both flows — performance dependency on data quality/alignment not fully characterized
- 74 hours on 8×4090 for 7B model — non-trivial compute for post-training

## Relationship to Wiki
- Contrasts with [[concepts/TARA]]: TARA uses external BFM teacher for representation alignment; DIVA uses internal mutual information between understanding and generation branches — both are "representation alignment" but with different signal sources
- Extends the "don't retrain, redirect" pattern: DIVA only tunes middle layers + lightweight encoders, backbone mostly frozen
- New domain for this wiki: first paper on UMMs (previous papers all on FGVR/HVR/TBPS with single-task models)

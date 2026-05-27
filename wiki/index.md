# Research Wiki Index

This is the entry point into the knowledge base. It tracks what areas are covered, how deeply, and where to find things.

## Current Coverage

### Active Research Areas
- **Fine-Grained Visual Recognition (FGVR) / Hierarchical Visual Recognition (HVR)**: MLLMs for fine-grained classification and taxonomic prediction
- **Text-based Person Search (TBPS)**: Fine-grained vision-language retrieval for person identification
- **Implicit Local Alignment**: Region–phrase correspondence without external supervision
- **Attention-guided Methods**: Using transformer attention maps as learning signals
- **Representation Alignment**: Using pretrained foundation models as teachers to inject structured knowledge
- **Reinforcement Fine-Tuning for VL**: CoT-based RL (TAPO) vs. No-Thinking RL for classification
- **Unified Multimodal Models (UMMs)**: Single-backbone models for both understanding and generation, and the challenge of turning inter-branch conflict into synergy
- **Explainable Image Quality Assessment (IQA)**: Multi-granularity quality understanding with LMMs — from global quality description to pixel-level distortion grounding

### Papers Ingested
- FGVR / HVR: 2 papers ([TARA](wiki/papers/TARA-taxonomy-aware-alignment.md) CVPR 2026, [Fine-R1](wiki/papers/Fine-R1-fine-grained-recognition.md) ICLR 2026)
- TBPS: 1 paper ([ITSELF](wiki/papers/ITSELF-attention-guided-alignment.md) WACV 2026)
- UMMs: 1 paper ([DIVA](wiki/papers/DIVA-representation-divergence-mutual-reinforcement.md) ICML 2026)
- IQA: 1 paper ([IQA-Spider](wiki/papers/IQA-Spider-multi-granularity-quality.md) ICML 2026)

### Concepts Indexed
- [[concepts/TAPO]] — Triplet Augmented Policy Optimization
- [[concepts/TARA]] — Taxonomy-Aware Representation Alignment
- [[concepts/HVR]] — Hierarchical Visual Recognition
- [[concepts/No-Thinking-RFT]] — No-Thinking Reinforcement Fine-Tuning
- [[concepts/GRAB]] — Guided Representation with Attentive Bank
- [[concepts/MARS]] — Multi-layer Attention for Robust Selection
- [[concepts/ATS]] — Adaptive Token Scheduler
- [[concepts/implicit-local-alignment]] — Implicit Local Alignment approaches
- [[concepts/DIVA]] — DIvergence-based mutual reinforcement for UMMs
- [[concepts/text-to-point-grounding]] — Training-free grounding via logit-to-coordinate mapping

### Entities Indexed
- [[entities/FGVR]] — Fine-Grained Visual Recognition task
- [[entities/BFM]] — Biology Foundation Models (BioCLIP, BioCLIP2, BioCAP)
- [[entities/TBPS]] — Text-based Person Search task
- [[entities/UMM]] — Unified Multimodal Models architecture
- [[entities/IQA]] — Image Quality Assessment task

### Cross-cutting Threads
- **Knowledge injection vs. knowledge deployment**: TARA injects external taxonomic knowledge from BFMs; Fine-R1 trains MLLMs to better deploy their *internal* knowledge via CoT+RL. Same lab, same base models, complementary approaches — can they be combined?
- **Representation alignment as a general recipe**: Both ITSELF (attention-guided) and TARA (BFM-guided) use an external representation signal to guide intermediate features — different teachers, same pattern.
- **"Don't retrain, redirect"**: All four papers take a strong pretrained model and add lightweight training mechanisms rather than full retraining. Pattern worth naming.
- **Internal vs. external alignment signals**: TARA uses external BFM teacher for representation alignment; DIVA uses internal cross-branch mutual information. Two ends of a spectrum — when is each appropriate?
- **Training-free capability unlocking**: IQA-Spider achieves pixel-level grounding by reusing native LMM logits (text-to-point) without any grounding-specific training. Same philosophical pattern as "don't retrain, redirect" — but pushed even further to "don't train at all for this capability."

## Quick Links

- [[gaps/confirmed-gaps]] — problems the field agrees are open
- [[gaps/hypotheses]] — your current ideas and their status
- [[gaps/questions]] — open questions from reading (15 questions)
- [[synthesis/field-map]] — how the pieces fit together
- [[synthesis/shared-assumptions]] — what everyone takes for granted
- [[overview]] — 1-page summary of your current understanding
- [[log]] — timestamped history of all actions

## Reading Queue
1.
2.
3.

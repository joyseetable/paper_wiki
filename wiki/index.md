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
- **Diffusion Model Alignment**: Reward-free score-matching alignment (AGSM) and critic-guided diffusion RL (CGPO)
- **AI Agent Safety**: Taxonomy-guided guardrail models for trajectory-level agent safety diagnosis
- **Cross-lingual Low-Resource Generation**: Source-grounded semantic RL for target-language generation
- **Robot Manipulation with Future Priors**: Video diffusion-based future conditioning for RL exploration

### Papers Ingested
- FGVR / HVR: 2 papers ([TARA](wiki/papers/TARA-taxonomy-aware-alignment.md) CVPR 2026, [Fine-R1](wiki/papers/Fine-R1-fine-grained-recognition.md) ICLR 2026)
- TBPS: 1 paper ([ITSELF](wiki/papers/ITSELF-attention-guided-alignment.md) WACV 2026)
- UMMs: 1 paper ([DIVA](wiki/papers/DIVA-representation-divergence-mutual-reinforcement.md) ICML 2026)
- IQA: 1 paper ([IQA-Spider](wiki/papers/IQA-Spider-multi-granularity-quality.md) ICML 2026)
- Diffusion / T2I: 1 paper ([AGSM](wiki/papers/AGSM-alignment-guided-score-matching.md) arxiv 2026)
- Diffusion RL: 1 paper ([CGPO](wiki/papers/CGPO-critic-guided-diffusion-policy.md) arxiv 2026)
- Agent Safety: 1 paper ([AgentDoG 1.5](wiki/papers/AgentDoG-1.5-agent-safety.md) arxiv 2026)
- NLP / Cross-lingual: 1 paper ([SG-SRL](wiki/papers/SG-SRL-crosslingual-semantic-rl.md) arxiv 2026)
- Robotics: 1 paper ([FEC](wiki/papers/FEC-future-experience-conditioning.md) arxiv 2026)

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
- [[concepts/reward-hacking]] — When RL optimizes the proxy reward but violates intended behavior
- [[concepts/influence-function-purification]] — Gradient-alignment-based data selection (AgentDoG 1.5)
- [[concepts/score-matching-alignment]] — Reward-free T2I alignment via Plackett-Luce score guidance (AGSM)
- [[concepts/diffusion-policy-rl]] — Diffusion models as policy representations in RL (CGPO)

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
- **Soft-token-only training (AGSM)**: 1.8M trainable params, frozen backbone, yet achieves substantial alignment gains and is complementary to full DPO. The purest "don't retrain, redirect" instance yet — only 8 tokens trained.
- **"Expensive guidance at training, cheap inference"**: TARA (BFM at training, discard at inference), CGPO (critic guidance at training, unguided deployment). Pattern: use costly signals to shape behavior, then deploy the distilled policy.
- **Decouple-then-recover**: SG-SRL separates semantic learning from form learning, accepts intermediate degradation, recovers at the end. Same logic as TARA's "inject knowledge via BFM, then fine-tune with RFT" — two-phase design where an intermediate messy state is valuable.

## Quick Links

- [[gaps/confirmed-gaps]] — 4 gaps confirmed by 3+ papers (G1–G4)
- [[gaps/hypotheses]] — 13 current hypotheses (H1–H13)
- [[gaps/questions]] — open questions from reading (30+ questions)
- [[synthesis/field-map]] — how the pieces fit together
- [[synthesis/shared-assumptions]] — what everyone takes for granted
- [[synthesis/discussion-2026-05-29]] — TARA-centric FGVR research roadmap (4-phase decision tree)
- [[overview]] — 1-page summary of your current understanding
- [[log]] — timestamped history of all actions

## Reading Queue
1.
2.
3.

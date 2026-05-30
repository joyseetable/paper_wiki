---
paper_id: 2605.29864
title: "LLM-Guided Future Hypotheses for Horizon-Aware Exploration in Multi-Step Robot Manipulation"
authors: "Mohammad Khoshnazar, Andrew Melnik, Michael Beetz"
year: 2026
venue: arxiv
status: skimmed
confidence: medium
tags: [robotics, manipulation, video-diffusion, future-prediction, RL, digital-twin]
---
# Summary
Short-horizon, task-consistent future videos — generated via LLM-guided task grounding + robot-free digital-twin rollout + mask-free diffusion inpainting — can serve as structured priors for multi-step robot manipulation BC and RL fine-tuning, improving policy execution and RL adaptation speed.

## Key Takeaways
- **Future conditioning as structured exploration**: GenFuture (imperfect generated futures) outperforms NoFuture in both BC and BC+RL settings across RoboCasa and CALVIN; WrongFuture (mismatched) degrades to zero — confirming signal specificity
- **RL fine-tuning benefits from future priors**: BC+RL learning curves averaged across 8 CALVIN tasks show GTFuture improves fastest, GenFuture > NoFuture in both speed and asymptote
- **Mask-free diffusion inpainting avoids privileged info**: 32-channel branch on CogVideoX generates robot-containing future clips from DT transparent video without needing robot masks at inference

## Method
FEC: (1) LLM task grounding: GPT-4o + task ontology → structured spec (object, part, state transition); (2) Robot-free DT articulation rollout → transparent video; (3) Mask-free video diffusion inpainting (CogVideoX + VideoPainter adaptation) → robot-containing future clip. Future latents: ResNet-18 per-frame → temporal pooling (B=4 bins) → learned projection (dg=256). Main controller: BC+RL (TD3-style fine-tuning from BC initialization). VDM trained on paired DT/robot clips.

## Results
- CALVIN BC+RL: GTFuture near-100% on easy tasks, GenFuture 5-15pp above NoFuture
- CALVIN BC-only: GenFuture 5-15pp above NoFuture on most tasks
- RoboCasa CloseDrawer BC+RL: NoFuture 61.7% → GenFuture 75.7% → GTFuture 82.3%
- WrongFuture = 0% across all settings, confirming signals are interpreted correctly

## Limitations & Open Questions
- Simulation only — no sim-to-real transfer
- Task grounding uses privileged simulator state
- No quantitative study of VDM or DT fidelity
- [[gaps/questions]]

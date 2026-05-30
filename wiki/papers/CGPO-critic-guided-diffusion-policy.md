---
paper_id: 2605.30056
title: "Sample-Efficient Diffusion-based Reinforcement Learning with Critic Guidance"
authors: "Shutong Ding, Zejia Zhong, Zhongyi Wang, Ke Hu, Bikang Pan, Jingya Wang, Ye Shi"
year: 2026
venue: arxiv
status: skimmed
confidence: medium
tags: [RL, diffusion-policy, critic-guidance, exploration-exploitation, real-world-robot]
---
# Summary
Training-free critic guidance integrated into the denoising process of diffusion policies addresses the exploration-exploitation trade-off in diffusion-based RL — steering action generation toward high-value regions while preserving action diversity — and enables the first successful real-world diffusion RL on a Franka robot arm.

## Key Takeaways
- **Sampling-based improvement plateaus as policy concentrates**: As training proceeds, the policy distribution narrows → sampled candidates become similar → critic contrast ∆Q shrinks → update signal weakens. CGPO replaces multi-candidate sampling with a single guided target.
- **DSG preserves manifold compatibility**: Spherical Gaussian constraint ensures guided reverse steps stay within the typical-radius shell of the diffusion reverse process, preventing out-of-distribution action generation.
- **Q-signal calibration is critical**: DDQN targets + truncated-quantile aggregation + base value network — all three needed for reliable critic-weighted training. The same Q signal is used for both guidance direction AND state reweighting.
- **First real-world diffusion RL**: 80% success on cylindrical peg-in-hole vs. 65% SAC baseline, training from scratch without demonstrations.

## Method
CGPO: (1) Critic-guided target synthesis — run guided reverse diffusion (DSG) only in last G steps using critic gradient ∇xt Qˆ(s, xˆ0(xt, s, t)); (2) Weighted denoising regression on guided targets with state reweighting via calibrated value network Vϕ; (3) Diffusion entropy regularization for diversity preservation. Q-signal: distributional critic with DDQN targets and truncated-quantile aggregation. Guidance only during training; deployment uses unguided sampler.

## Results
- 5 MuJoCo v3 tasks: CGPO best overall — Ant 7272, HalfCheetah 14369, Hopper 4171, Humanoid 7131, Walker2d 6483
- Real-world Franka: 80% peg-in-hole success vs. SAC 65%
- Ablation: removing DSG, DDQN, TQC, or Vϕ all degrade performance

## Limitations & Open Questions
- Only tested on MuJoCo locomotion + Franka manipulation
- [[gaps/questions]]

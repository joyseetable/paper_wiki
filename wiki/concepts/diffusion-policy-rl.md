---
aliases: [CGPO, diffusion-based RL, critic-guided diffusion]
related_papers: [[CGPO-critic-guided-diffusion-policy]]
---
# Diffusion Policy RL
Using diffusion models as policy representations in reinforcement learning, where the action distribution is modeled as a conditional denoising diffusion process. Two main paradigms:

1. **Sampling-based**: Draw K candidate actions from the diffusion policy, select best by critic value, train on reweighted candidates. Limitation: as policy concentrates, candidates become similar → critic contrast ∆Q shrinks → update signal weakens.

2. **Gradient-based (CGPO)**: Use critic gradients as training-free guidance during the diffusion reverse process to directly steer toward high-value actions. The guided action becomes the supervised target for policy regression.

## CGPO innovations
- DSG (Spherical Gaussian constraint): Keeps guided reverse steps on the diffusion manifold
- Late-step-only guidance: Apply critic guidance only in final G denoising steps when clean predictions are reliable
- Q-signal calibration: DDQN targets + truncated-quantile aggregation + base value network for stable weighting
- Guidance only during training; deployment uses unguided sampler (zero inference overhead)

## Key trade-off
Sampling-based = better exploration early (diverse candidates), worse exploitation late (weak signal).
Gradient-based = better exploitation (directed search), risk of premature convergence + out-of-distribution actions.
CGPO balances via DSG constraint + late-step guidance.

---
aliases: [Adaptive Token Scheduler]
related_papers: [[ITSELF-attention-guided-alignment]]
---
# ATS (Adaptive Token Scheduler)

A coarse-to-fine token budget scheduler used in [[GRAB]]. The number of retained tokens decays exponentially over training steps: k_t = ⌊N × ρ_start × (ρ_end/ρ_start)^(t/T)⌋, after which it stays at ⌊N × ρ_end⌋.

Rationale: In early training, the model needs broad context to avoid prematurely discarding useful cues (the R1 gap between full and attention-retained inputs narrows to <1pp by epoch 3). Once attention becomes reliable, the budget tightens to focus on the most discriminative tokens.

Typical settings: ρ_start=0.65, ρ_end=0.5, T = epoch when baseline achieves best result. Step-level scheduling outperforms epoch-level scheduling because it provides smoother annealing.

## Key Papers
- [[ITSELF-attention-guided-alignment]]: Introduces ATS as part of GRAB

## Relationships
- part of: [[GRAB]]
- contrasts with: fixed token budgets

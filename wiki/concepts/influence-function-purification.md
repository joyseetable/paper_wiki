---
aliases: [data purification, preference-aware data selection]
related_papers: [[AgentDoG-1.5-agent-safety]]
---
# Influence-Function Purification
A data selection method that scores training examples by how strongly their gradient aligns with a desired behavior direction in parameter space. Retains only the most instructive examples.

## How it works
1. Define a guardrail direction ĝ_guard: weighted average of gradient contrasts between correct vs. incorrect responses on safety target prompts
2. Compute gradient ĝ_z for each candidate training example
3. Score = ĝ_z^T · ĝ_guard — how well does training on this example push the model toward the desired behavior?
4. Retain top-scoring examples

## Key properties
- Reduces dataset from thousands to ~1k while improving quality
- Mitigates overfitting to spurious patterns
- The selection signal is behavior-specific (not generic "quality") — examples are scored by alignment with the target *behavior* direction

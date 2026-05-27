---
aliases: [No-Thinking Reinforcement Fine-Tuning]
related_papers: [[TARA-taxonomy-aware-alignment]]
---
# No-Thinking RFT

A variant of rule-based reinforcement fine-tuning that explicitly prohibits the model from generating reasoning traces. The prompt says "Please directly output the answer" and the reward is strict accuracy (1 if exact match, 0 otherwise).

Unlike Thinking-RFT (which incentivizes chain-of-thought and produces "aha moments"), No-Thinking RFT produces shorter, more concise outputs and has been shown to outperform thinking-based RFT on classification tasks, especially for smaller models.

## Key Papers
- Think or Not Think (Li et al., 2025): First to show No-Thinking RFT can outperform Thinking-RFT on classification
- [[TARA-taxonomy-aware-alignment]]: Uses No-Thinking RFT as the base training method, then adds TARA on top

## Relationships
- contrasts with: [[Thinking-RFT]], standard SFT
- used by: [[TARA]]

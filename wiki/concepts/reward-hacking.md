---
aliases: [reward exploitation, Goodhart's law in RL]
related_papers: [[SG-SRL-crosslingual-semantic-rl]]
---
# Reward Hacking
A failure mode in RL where the policy optimizes the proxy reward function in ways that satisfy the literal reward criteria but violate the intended behavior. Not the same as overfitting — the model achieves high reward but degrades on the true objective.

Two key insights from SG-SRL:
1. **Verbosity-based reward hacking is structural**: A relevance-oriented reranker reward inherently rewards longer outputs (more text = more source concepts covered). Changing the RL optimizer (GRPO → Dr.GRPO) doesn't fix it — the reward *structure* is the issue.
2. **The "hacked" intermediate policy may still have value**: SG-SRL treats semantic RL as mid-training — the intermediate checkpoint learns useful cross-lingual alignment even though it's verbose. Recovery SFT restores form while preserving semantic gains.

## Contrast with typical solutions
- Standard: add KL penalty, earlier stopping (treats hacking as pathology)
- SG-SRL approach: accept the semantic gains, fix the form later (treats hacking as a phase)

## Connections to the wiki
- Similar pattern to [[H7]]: the intermediate state (RL checkpoint with teacher) may have value even if the final state is standard
- This "accept the messy, recover later" pattern is a recurring theme in reward design

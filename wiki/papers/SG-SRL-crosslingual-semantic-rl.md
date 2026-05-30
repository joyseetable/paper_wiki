---
paper_id: 2605.29502
title: "Source-Grounded Semantic Reinforcement Learning for Low-Resource Target-Language Generation"
authors: "Zeli Su, Ziyin Zhang, Zewei Pan, Zhou Liu, Dingcheng Huang, Dehan Li, Zhankai Xu, Longfei Zheng, Xiaolu Zhang, Jun Zhou, Wentao Zhang"
year: 2026
venue: arxiv
status: skimmed
confidence: medium
tags: [NLP, cross-lingual, reinforcement-learning, low-resource, reward-hacking]
---
# Summary
Source-language monolingual data can be converted into cross-lingual semantic supervision for low-resource target-language generation via a train–reinforce–recover framework using a multilingual reranker as a reference-free reward model.

## Key Takeaways
- **Train–Reinforce–Recover decoupling**: Separate target-language form learning (small parallel corpus) from semantic grounding (large source-only corpus), then recover fluency. The RL stage is "semantic mid-training" — the intermediate checkpoint has better semantics but degraded form.
- **Verbosity reward hacking is structural, not optimization-specific**: The relevance-oriented reranker reward inherently encourages longer outputs. Changing RL optimizer (GRPO → Dr.GRPO) doesn't fix it — the reward structure itself is the issue.
- **Encoder-based embedding can substitute for LLM reranker**: In the Tibetan experiment, a CINO-based embedding reward works when an LLM reranker is unavailable for genuinely low-resource languages.

## Method
Three stages: (1) SFT on small parallel corpus → cold-start; (2) GRPO on source-only data with reranker semantic reward + language/length/repetition safeguards; (3) 1-epoch SFT recovery on parallel corpus. Reranker reward: normalized yes/no probability from a generative judge for cross-lingual query-candidate relevance.

## Results
- Chinese→Thai title generation: 72.3% pairwise win rate over cold-start SFT
- Long-form translation transfer: SG-SRL > cold-start SFT across entity/event/factual/fluency/completeness
- Tibetan setting: encoder-based reward achieves comparable BLEU with Chinese vs. Tibetan reference

## Limitations & Open Questions
- Not matched to the most ideal low-resource setting (strong cross-lingual reranker often unavailable)
- Recovery stage can't perfectly restore form — some verbosity artifacts may remain
- [[gaps/questions]]

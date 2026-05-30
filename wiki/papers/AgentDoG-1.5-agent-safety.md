---
paper_id: 2605.29801
title: "AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security"
authors: "Shanghai Artificial Intelligence Laboratory"
year: 2026
venue: arxiv
status: skimmed
confidence: medium
tags: [AI-safety, agents, guardrails, taxonomy, data-purification, RL]
---
# Summary
A lightweight agent safety alignment framework using (1) a 3D risk taxonomy with setting-specific leaf-category customization, (2) influence-function-based data purification to train guard models from ~1k samples, and (3) AgentDoG 1.5 as both a reward model for safety RL and an online Pre-Reply guardrail.

## Key Takeaways
- **Taxonomy-guided data engine with influence-function purification**: Retains only ~1k most informative samples — preference-weighted guardrail direction in parameter space selects examples whose gradients align with risk recognition behavior
- **GDPO for multi-dimensional safety RL**: Normalize advantages per dimension (failure mode, harm, risk source) rather than summing into scalar — preserves partial-satisfaction signal
- **Pre-Reply guardrail**: Intercept at final delivery stage (not every tool call) — balances comprehensive risk interception with sub-second latency

## Method
Three components: (1) 3D safety taxonomy (risk source × failure mode × real-world harm) with setting-specific leaf categories for OpenClaw and Codex; (2) Data engine: taxonomy-guided trajectory synthesis → CoT augmentation → influence-function purification → ~1k samples; (3) Two-stage training: SFT on purified data → RL with GDPO (per-dimension advantage normalization). Deployed as online Pre-Reply guardrail on OpenClaw.

## Results
- AgentDoG 1.5-4B: 92.2% Acc / 92.7% F1 on R-Judge; 72.4% Acc / 74.3% F1 on ATBench — best open-source
- Fine-grained diagnosis: 55.2% avg (vs. GPT-5.4 25.8%) — substantial improvement over frontier models
- Online guardrail: reduces ClawSafety ASR from 56.25% → 18.75%; AgentHazard ASR from 41.92% → 26.92%
- Scalable lightweight RL environment: <2.5GB peak memory, supports 10k concurrent environments

## Limitations & Open Questions
- Unified coarse-to-fine model not systematically optimized
- [[gaps/questions]]

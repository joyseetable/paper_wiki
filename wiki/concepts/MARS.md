---
aliases: [Multi-layer Attention for Robust Selection]
related_papers: [[ITSELF-attention-guided-alignment]]
---
# MARS (Multi-layer Attention for Robust Selection)

A token selection mechanism inside [[GRAB]] that aggregates attention maps across multiple transformer layers rather than relying on a single layer. 

Process: (1) Denoise each layer's attention by discarding the lowest δ fraction of weights; (2) Add identity matrix to preserve self-dependencies and normalize; (3) Compose across selected layers via sequential matrix multiplication; (4) Diversity-aware TopK selection of image patches and text tokens.

Rationale: Shallow layers have low-entropy attention (peaked on textures/edges), middle layers capture broader context, and deep layers re-focus on semantic regions. Fusing Middle+Late layers balances contextual coverage with discriminative focus. Including early layers degrades performance (noise injection). Optimal discard ratio δ=0.25.

## Key Papers
- [[ITSELF-attention-guided-alignment]]: Introduces MARS as part of GRAB

## Relationships
- part of: [[GRAB]]
- contrasts with: single-layer attention selection, mean-attention heuristics, std-attention heuristics

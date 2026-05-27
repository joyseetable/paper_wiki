---
aliases: [Taxonomy-Aware Representation Alignment]
related_papers: [[TARA-taxonomy-aware-alignment]]
---
# TARA (Taxonomy-Aware Representation Alignment)

A training-only strategy for injecting taxonomic knowledge into LMMs by aligning their intermediate representations with those of a pretrained Biology Foundation Model (BFM). Two alignment paths:

1. **Visual alignment**: LMM intermediate visual features → BFM vision encoder output (all visual tokens)
2. **Label alignment**: LMM first answer token → BFM text-encoded ground-truth label at the target granularity

Trained alternately with No-Thinking RFT. At inference, the BFM and projectors are discarded — zero overhead.

## Key Papers
- [[TARA-taxonomy-aware-alignment]]: Introduces TARA; SOTA hierarchical consistency on iNat21 + novel species generalization

## Relationships
- uses: [[BFM]] (BioCLIP2 as teacher)
- trained with: [[No-Thinking-RFT]]
- task: [[HVR]]
- contrasts with: explicit instruction tuning for FGVR, hyperbolic embedding methods (HiCLIP, HGCLIP), training-free approaches (SAV)

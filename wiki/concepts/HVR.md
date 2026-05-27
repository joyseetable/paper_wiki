---
aliases: [Hierarchical Visual Recognition]
related_papers: [[TARA-taxonomy-aware-alignment]]
---
# Hierarchical Visual Recognition (HVR)

A visual recognition task where the model must predict the full taxonomic path (root → ... → leaf) rather than just a single flat label. Stricter than fine-grained visual recognition (FGVR): a prediction counts as correct only if ALL levels along the path are correct (HCA metric). Partial credit is given by POR (average node correctness) and S-POR (longest contiguous correct segment).

## Why it matters
- Enables flexible usage: expert wants species, general user wants kingdom/phylum
- Predicting the full hierarchy improves robustness and enables generalization to novel child/parent categories
- LMMs are surprisingly bad at this — they break hierarchical consistency even when they get the leaf node right

## Key Metrics
- **HCA** (Hierarchical Consistent Accuracy): strict — all levels must match
- **Acc_leaf**: flat leaf-node accuracy (upper bounds HCA)
- **POR**: average fraction of correct nodes
- **S-POR**: longest contiguous correct run
- **TOR**: pairwise consistency between adjacent levels

## Key Papers
- [[TARA-taxonomy-aware-alignment]]: Proposes TARA to inject taxonomic knowledge from BFMs; identifies HVR as key LMM limitation

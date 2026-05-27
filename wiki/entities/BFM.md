---
type: model-family
---
# Biology Foundation Models (BFMs)

Large-scale pretrained models that encode biological taxonomic knowledge through hierarchical contrastive learning on species-level data. They produce embedding spaces where species representations reflect ecological and functional relationships.

## Key Models
- **BioCLIP** (CVPR 2024): Vision foundation model for the tree of life — hierarchical contrastive learning on 450K species
- **BioCLIP2** (2025): Scaled-up version with emergent taxonomic properties — used as teacher in [[TARA]]
- **BioCAP** (2025): Exploits synthetic captions beyond labels in biological foundation models

## Key Property
The embedding space naturally organizes by taxonomy: species in the same genus are closer than species in different genera, etc. This hierarchical structure in representation space makes BFMs ideal as "taxonomy teachers" for LMMs.

## Used By
- [[TARA]]: Uses BioCLIP2 as the teacher model for representation alignment

---
type: task
---
# Text-based Person Search (TBPS)

A fine-grained vision-language retrieval task: given a textual query describing a person's appearance (clothing, accessories, posture), retrieve the matching pedestrian image from a large gallery. Harder than generic image-text retrieval because distinguishing individuals requires subtle attribute-level discrimination.

## Key Benchmarks
- CUHK-PEDES: 40,206 images, 80,412 captions, 13,003 identities
- ICFG-PEDES: 54,522 image-text pairs, 4,102 identities
- RSTPReid: 20,505 images, 4,101 identities, 5 images + 2 captions per ID

## Key Challenge
Capturing fine-grained discriminative details without shortcut learning or spurious correlations. Global CLIP embeddings miss local attribute correspondences; explicit methods (parsing, MLLM captions) add cost and domain bias.

## Papers Using This
- [[ITSELF-attention-guided-alignment]]: Attention-guided implicit local alignment, SOTA among CLIP-based methods

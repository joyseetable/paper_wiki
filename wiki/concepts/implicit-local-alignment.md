---
aliases: []
related_papers: [[ITSELF-attention-guided-alignment]]
---
# Implicit Local Alignment

A family of methods for fine-grained vision-language tasks that learn region–phrase correspondences without external supervision (no parsing networks, no MLLM captions, no extra annotations).

## Variants
- **Embedding-space correspondence**: Infer region–phrase matches from implicit signals (weakly constrained, unstable with sparse labels)
- **Fully implicit feature learning**: Optimize local losses without explicit text–region pairing (no guarantee of precise correspondence)
- **Masked-modeling alignment**: Cross-modal reconstruction (global-context shortcuts can bypass true dependencies)
- **Attention-guided** ([[GRAB]]): Use model's own attention as a locality prior to select discriminative tokens

## Key Challenge
Without explicit constraints on phrase–region correspondence, these methods are prone to shortcut learning and spurious correlations that cause misalignment.

## Key Papers
- [[ITSELF-attention-guided-alignment]]: Attention-guided approach that turns internal saliency maps into reliable anchors

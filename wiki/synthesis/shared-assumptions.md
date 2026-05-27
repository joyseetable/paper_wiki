# Shared Assumptions

What the field takes for granted — often implicitly. These are the most fertile ground for new ideas, because questioning an assumption opens up entirely new directions.

---

## A1: The pretrained backbone must be preserved (not retrained)
**The assumption**: When adapting foundation models (CLIP, Qwen-VL) to fine-grained tasks, you should add lightweight adapters/training objectives rather than fully fine-tune or train from scratch.
**Evidence it's unexamined**:
- [[papers/ITSELF-attention-guided-alignment]]: Uses frozen-or-lightweight CLIP with GRAB as an added branch
- [[papers/TARA-taxonomy-aware-alignment]]: LMM fully fine-tuned but only 1 epoch; BFM is frozen
- [[papers/Fine-R1-fine-grained-recognition]]: Full fine-tuning but only 10 epochs on 404 samples
**What if it's wrong?**: If backbone plasticity is actually needed for fine-grained tasks, aggressive fine-tuning or even training small models from scratch on curated data might outperform adapter-based approaches.

## A2: External guidance signals are necessary for fine-grained alignment
**The assumption**: Standard supervised losses (cross-entropy, contrastive) are insufficient for fine-grained tasks; you need an additional structured signal (attention maps, BFM embeddings, triplet contrast).
**Evidence it's unexamined**:
- [[papers/ITSELF-attention-guided-alignment]]: Uses CLIP's own attention as guidance
- [[papers/TARA-taxonomy-aware-alignment]]: Uses BioCLIP2 embeddings as teacher
- [[papers/Fine-R1-fine-grained-recognition]]: Uses triplet labels (positive/negative pairs) + structured CoT
**What if it's wrong?**: Maybe the standard objectives ARE sufficient but the training recipes (learning rate, batch size, data sampling) haven't been optimized for fine-grained tasks. A well-tuned baseline might close much of the gap.

## A3: The teacher/guidance signal can be discarded at inference
**The assumption**: The external guidance (attention maps, BFM, projectors) is training-only scaffolding — discard it at deployment for zero overhead.
**Evidence it's unexamined**:
- [[papers/ITSELF-attention-guided-alignment]]: Attention maps used for token selection during training only; inference uses standard CLIP forward pass
- [[papers/TARA-taxonomy-aware-alignment]]: "During inference, both the BFMs and projectors are discarded"
- [[papers/Fine-R1-fine-grained-recognition]]: TAPO is training-only; at inference only CoT is generated (which IS retained)
**What if it's wrong?**: Keeping the teacher at inference — e.g., computing attention maps on-the-fly, or keeping BFM embeddings as retrieval targets — might provide additional gains. The "discard the teacher" assumption hasn't been ablated (no comparison of "training-only teacher" vs. "teacher at inference too").

## A4: Knowledge injection and knowledge deployment are separate problems
**The assumption**: Adding knowledge to a model (TARA) and teaching it to use existing knowledge (Fine-R1) are distinct approaches that address different bottlenecks.
**Evidence it's unexamined**:
- [[papers/TARA-taxonomy-aware-alignment]]: Frames the problem as "LMMs lack taxonomic knowledge" → inject it from BFMs
- [[papers/Fine-R1-fine-grained-recognition]]: Frames the problem as "LMMs have knowledge but can't deploy it" → teach CoT reasoning
- Same lab, same base models, different framing — but never tested together
**What if it's wrong?**: If knowledge injection AND knowledge deployment are both needed for best performance, combining TARA + Fine-R1 should yield super-additive gains. Conversely, if one subsumes the other, combining them should yield no improvement over the better single approach.

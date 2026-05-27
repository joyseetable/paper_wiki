---
aliases: [training-free grounding, logit-to-coordinate mapping]
related_papers: [[IQA-Spider-multi-granularity-quality]]
---
# Text-to-Point Grounding

A training-free paradigm for bridging LMM textual outputs to pixel-level segmentation without special tokens or additional fine-tuning. The key insight: native language model logits for positional terms (left/right, top/bottom) already encode spatial information — by applying closed-set softmax over these terms and computing a weighted average, you get (x,y) point coordinates usable as SAM prompts.

## Mechanism
1. LMM generates text containing positional terms (left/right for x-axis, top/bottom for y-axis)
2. Extract logits corresponding to {left, right} and {top, bottom} from the final hidden states
3. Apply temperature-scaled closed-set softmax to obtain probability distribution over the 2×2 spatial grid
4. Weighted average: x = p(left)×0 + p(right)×1, y = p(top)×0 + p(bottom)×1, then scale to image dimensions
5. Feed (x,y) as point prompt to frozen SAM for mask generation

## Key Properties
- **Reasoning-preserving**: No architectural modification to the language model — instruction-following and reasoning capabilities intact (unlike special-token approaches)
- **Training-free**: No grounding-specific training data or fine-tuning needed for the segmentation head
- **Plug-and-play**: Compatible with any LMM that can generate positional terms; SAM can be swapped for any point-prompted segmenter

## Key Papers
- [[IQA-Spider-multi-granularity-quality]]: Introduces text-to-point grounding for IQA; outperforms fine-tuned grounding baselines (Q-Ground, LISA) while being training-free

## Relationships
- contrasts with: special-token grounding (LISA, GLAMM) which requires fine-tuning and degrades instruction-following
- contrasts with: attention-map-based grounding (F-LMM) which has high memory overhead and needs an additional image encoder
- relates to: the "don't retrain, redirect" pattern — achieves new capability (grounding) by reusing native model outputs rather than adding or fine-tuning modules
- similar philosophy to: [[concepts/DIVA]]'s approach of reusing native model outputs for cross-task transfer

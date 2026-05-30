---
aliases: [AGSM, alignment-guided score matching]
related_papers: [[AGSM-alignment-guided-score-matching]]
---
# Score-Matching Alignment
A reward-free approach to text-image alignment that integrates preference signals directly into the diffusion score-matching objective. Key innovation: instead of using external reward models or contrastive losses, derives alignment from the diffusion model's own denoising error as an implicit reward, then formulates alignment as Plackett-Luce preference learning over score targets.

## How it differs from alternatives
- **vs. SoftREPA**: Contrastive loss pushes negative pairs off-manifold → failures (overcounting, repetition). AGSM uses bounded score-level guidance on negatives — prevents divergence.
- **vs. DPO/RL**: DPO uses human preference pairs + external reward models. AGSM uses intrinsic denoising error as reward — reward-free, only 1.8M trainable params.
- **vs. CFG**: CFG tilts sampling distribution; AGSM tilts training target scores.

## "Don't retrain, redirect" connection
AGSM is a strong instance of the "don't retrain, redirect" pattern: diffusion backbone fully frozen, only 8 soft text tokens trained per model, yet achieves substantial alignment gains and is complementary to full RL-based post-training. Like [[concepts/text-to-point-grounding]], it reinterprets native model outputs (denoising error) to unlock a new capability (better text alignment) without architectural changes.

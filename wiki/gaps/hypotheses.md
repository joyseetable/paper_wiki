# Hypotheses

Your testable research ideas. Each hypothesis must be falsifiable: "If X is true, then Y should be observable."

## Status Lifecycle
`draft` → `testing` → `confirmed` / `rejected`

---

## H1: TARA + Fine-R1 are complementary — combining knowledge injection and knowledge deployment yields additive gains

**Status**: draft
**Feasibility**: Low on 8×3090. Requires training TARA (~possible with Qwen3-VL-2B + ZeRO-3) AND Fine-R1 (~very tight for 3B, TAPO rollout memory is the killer), plus a combined run. Effectively 3-4 expensive experiments. Consider after validating H3 or H4 first.
**Addresses gap**: Cross-cutting: knowledge injection (TARA) vs. knowledge deployment (Fine-R1) solve orthogonal bottlenecks
**If true, then**: Training a model with TARA (BFM-guided representation alignment) followed by Fine-R1's CoT SFT + TAPO should achieve HCA on iNat-Plant that exceeds the sum of individual improvements over baseline. Specifically: TARA alone ≈ +6pp HCA, Fine-R1 alone ≈ +5pp HCA on closed-world FGVR → combined should be > +11pp (not ≈ +8pp which would suggest overlap).
**How to test**: 
1. Take Qwen3-VL-2B baseline on iNat21-Plant
2. Train TARA variant (No-Thinking RFT + BFM alignment)
3. Train Fine-R1 variant (CoT SFT + TAPO on FGVR-style labels)
4. Train combined variant (TARA → Fine-R1 sequential)
5. Compare: if combined >> max(TARA, Fine-R1), the bottlenecks are orthogonal
**Risks**: Sequential training may cause catastrophic forgetting of TARA's taxonomic structure during Fine-R1's RL phase. Mitigation: interleave or jointly optimize TARA alignment loss + TAPO objective.
**Created**: 2026-05-25

---

## H2: Attention-guided token selection (GRAB-style) generalizes beyond person search to biological FGVR

**Status**: draft
**Feasibility**: High on 8×3090. CLIP ViT-B/16 ~300M params total, easily fits in 24GB. Training on CUB-200 (6K images) is fast. Can be done in <1 day on a single 3090. Lowest risk to verify the "attention guidance is universal" claim.
**Addresses gap**: [[gaps/questions]] — "Does GRAB generalize beyond person search?" + A2 (all methods need external guidance — but what if the simplest guidance is already inside the model?)
**If true, then**: Applying GRAB (MARS + ATS) to a CLIP ViT fine-tuned on CUB-200 birds should improve R1 accuracy over the CLIP baseline by at least +3pp, comparable to the gains ITSELF achieves on TBPS benchmarks (+2-6pp over baseline).
**How to test**:
1. Take CLIP ViT-B/16 as frozen backbone
2. Add GRAB module (MARS with Middle+Late layers, ATS with ρ_start=0.65, ρ_end=0.5)
3. Train on CUB-200 with 4-shot per category (matching Fine-R1's data efficiency)
4. Compare R1 against: (a) CLIP zero-shot, (b) CLIP linear probe, (c) Fine-R1 on same data
5. If GRAB > CLIP linear probe but < Fine-R1, attention guidance is useful but weaker than CoT-based approaches
**Risks**: CLIP's attention quality for biological images may differ from person images (different pretraining distribution). The TBPS-specific design (MARS discard ratio 0.25, layer selection) may need re-tuning.
**Created**: 2026-05-25

---

## H3: The "teacher signal quality" threshold is surprisingly low — noisy guidance is nearly as good as clean guidance

**Status**: draft
**Feasibility**: Medium on 8×3090. Requires multiple TARA runs (3-4 variants × 1 epoch each). Qwen3-VL-2B + ZeRO-3 should fit. BioCLIP2 teacher is inference-only (~4GB). Bottleneck is training 3× TARA sequentially. 2B model → probably fine, ~1-2 days per run.
**Addresses gap**: [[gaps/questions]] — "What's the minimum viable BFM teacher?" + A3 (teacher discarded at inference — but how good does the teacher need to be during training?)
**If true, then**: Using a degraded BFM teacher (e.g., BioCLIP trained on 10% of species, or random-embedding teacher) for TARA should retain ≥70% of the gains compared to the full BioCLIP2 teacher. If false (gains collapse with weak teachers), then the teacher's quality/structure is the critical factor, not the alignment mechanism itself.
**How to test**:
1. Create 3 BFM variants: (a) Full BioCLIP2, (b) BioCLIP2 with 10% species (random subset), (c) Random-initialized ViT with same architecture (no taxonomic structure)
2. Run TARA on iNat21-Plant with each teacher, same base model (Qwen3-VL-2B)
3. Measure HCA: if (b) retains ≥70% of (a)'s gain, and (c) still shows non-zero improvement, the alignment mechanism matters more than teacher quality
4. If only (a) works, then the BFM's taxonomic structure is the active ingredient — you need a real biology model
**Risks**: Random teacher may introduce noise that actively harms training, making the results hard to interpret. Need a "no teacher" baseline for comparison.
**Created**: 2026-05-25

---

## H4: The gap between MLLMs and CLIP on FGVR is a training recipe problem, not a capability problem

**Status**: draft
**Feasibility**: High on 8×3090. Simple baseline sweep — no complex methods to implement, just careful training of existing models. Multiple small runs in parallel on different GPUs.
**Addresses gap**: [[synthesis/shared-assumptions#A2]] — "External guidance signals are necessary for fine-grained alignment." The counterfactual: maybe standard contrastive/classification objectives ARE sufficient, but the training recipes (learning rate schedule, data sampling strategy, augmentation, optimizer choice) haven't been properly optimized for FGVR tasks.
**If true, then**: A carefully tuned CLIP linear probe (or lightweight fine-tune) on CUB-200 should match or exceed the performance of complex methods (GRAB, TARA, Fine-R1) at the same data scale (4-shot per category). Specifically: well-tuned linear probe R1 should be within 2pp of Fine-R1-3B's closed-world accuracy, whereas current "standard" baselines trail by 5-10pp.
**How to test**:
1. Take CLIP ViT-B/16 on CUB-200 with 4-shot per category
2. Sweep: learning rates (1e-5 to 1e-1, log scale), optimizers (Adam, AdamW, SGD with momentum), schedulers (cosine, step, one-cycle), data augmentations (none, RandAugment, MixUp), class-balanced sampling vs. random
3. Also sweep the same for a small MLLM baseline (Qwen3-VL-2B with a simple classification head)
4. Compare best-found baseline against Fine-R1-3B and TARA results on the same data split
5. If tuned baseline closes >50% of the gap, the "external guidance is necessary" assumption (A2) is weakened — the field is overcomplicating the problem
**Risks**: The tuning space is large; need to be systematic to avoid cherry-picking. Also: this hypothesis being true doesn't mean complex methods are worthless — it means we don't know what the real baseline is. The result is interesting either way.
**Created**: 2026-05-25

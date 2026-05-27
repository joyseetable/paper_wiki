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

---

## H5: Text-to-point grounding generalizes beyond IQA — LMMs already have latent spatial grounding ability that can be unlocked training-free in other domains

**Status**: draft
**Feasibility**: High on 8×3090. No training required — purely inference-time evaluation. Take an existing LMM (Qwen2.5-VL-7B), prompt it with spatial questions on existing datasets, extract positional term logits, feed to SAM, measure mIoU. Can be done in <1 day on a single GPU.
**Addresses gap**: Cross-domain generalization of the text-to-point paradigm. [[gaps/questions]] — "Could this multi-granularity task taxonomy generalize to other visual understanding domains?" + challenges implicit assumption A5 (new capabilities require new training objectives)
**If true, then**: Applying the text-to-point grounding pipeline (closed-set softmax over positional term logits → SAM point prompt) to FGVR datasets (CUB-200 part localization, iNat attribute grounding) should yield non-trivial segmentation accuracy (mIoU > 0.15) without ANY domain-specific training, purely by repurposing the LMM's existing spatial language capability.
**How to test**:
1. Take Qwen2.5-VL-7B (or Qwen3-VL-7B) — no fine-tuning
2. Design spatial prompts for FGVR: "Which part of the bird shows the most distinctive color pattern?" → extract left/right/top/bottom logits
3. Map to point coordinates, feed to frozen SAM
4. Evaluate on CUB-200 with part annotations (head, wing, tail, breast) — measure mIoU
5. Also test on a non-visual domain (e.g., document layout: "Where is the title on this page?")
6. If mIoU > 0.15 on birds (vs. random point ~0.05), the spatial knowledge is latent and domain-general
7. Compare against a version where the same LMM is fine-tuned on IQA-Spider Stage 1 data — does IQA-specific tuning improve or hurt generalization?
**Risks**: LMMs may not naturally use positional language for FGVR attributes (they describe "red wing" but may not say "the red patch on the left wing"). FGVR part annotations may be too coarse for point-based grounding (a single point can't capture "the entire wing"). The success of this hypothesis depends on the LMM's pre-existing spatial language habits, which may be domain-dependent.
**Created**: 2026-05-27

---

## H6: DIVA-style shared/unique factorization can bridge knowledge injection and knowledge deployment in FGVR

**Status**: draft
**Feasibility**: Medium on 8×3090. Requires implementing DIVA's factorization on top of Fine-R1's two-path setup. Fine-R1-3B training is already tight on 8×3090; adding gated-MLP encoders and MI objectives increases memory. But the factorization operates only on middle layers + lightweight encoders (3-layer MLPs), not full model. Estimate ~1.5× Fine-R1 training cost. Doable with ZeRO-3 and gradient checkpointing.
**Addresses gap**: Cross-cutting thread: internal vs. external alignment signals. [[gaps/questions]] — "DIVA's internal MI-based alignment vs. TARA's external BFM teacher — could they be combined?" + challenges A4 (knowledge injection and deployment are separate problems — what if one mechanism bridges both?)
**If true, then**: Applying DIVA-style factorization to Fine-R1's two information flows — (a) CoT reasoning flow (visual analysis → comparison → prediction) and (b) direct classification flow (image → answer) — with shared components aligned via InfoNCE and unique components disentangled via NCE-CLUB, should improve closed-world FGVR accuracy by ≥2pp over Fine-R1 alone, because the shared factors would inject CoT's discriminative reasoning into the direct path without the inference cost of generating CoT.
**How to test**:
1. Take Fine-R1-3B after CoT SFT (before TAPO)
2. Construct two information flows from the same anchor image: Flow A = image + CoT prompt → full reasoning trace; Flow B = image + direct classification prompt → answer
3. Apply DIVA's two-stage post-training on middle layers (determine optimal layer range via diagnostic analysis):
   - Stage 1: Train gated-MLP shared/unique encoders with cross-path logit injection (backbone frozen)
   - Stage 2: Fine-tune middle layers with asymmetric InfoNCE alignment + NCE-CLUB disentanglement
4. At inference, use Flow B only (direct classification, no CoT) but with DIVA-enhanced representations
5. Compare against: (a) Fine-R1 baseline, (b) Fine-R1 + TARA
6. If DIVA-enhanced Flow B > Fine-R1 baseline, the factorization successfully transferred CoT knowledge into the direct path
**Risks**: The bias difference between CoT reasoning and direct classification may be weaker than the understanding vs. generation divergence in UMMs — without pixel-level supervision, the two FGVR paths may not naturally decouple. The "shared" component may end up capturing everything, leaving unique components as noise. A diagnostic experiment (gradient conflict analysis between the two flows, like DIVA's Figure 2) should precede full implementation.
**Created**: 2026-05-27

---

## H7: Inference-time teacher retention improves performance — challenging the universal "discard the teacher" assumption

**Status**: draft
**Feasibility**: Medium on 8×3090. Requires modifying inference pipelines of existing methods, not retraining. TARA inference: keep BFM projector, compute BFM embedding of input, use as additional retrieval/scoring feature. ITSELF inference: keep attention maps, use for token re-weighting. GRAB's local branch is already used at inference (λ_S blending) — so only TARA and IQA-Spider variants need testing. TARA variant requires loading BioCLIP2 at inference (~4GB), manageable.
**Addresses gap**: [[synthesis/shared-assumptions#A3]] — "The teacher/guidance signal can be discarded at inference." This is the most unexamined assumption in the wiki — NO paper has ablated "training-only teacher" vs. "teacher at inference too." If keeping the teacher helps, it opens a new design dimension. If it doesn't help, A3 is validated and we can confidently discard teachers.
**If true, then**: For at least one method, keeping the teacher signal at inference should yield measurable improvement over discarding it. Specifically: (a) TARA + BFM retrieval (compute BFM embedding of test image, use cosine similarity to BFM-text-encoded candidate labels as an auxiliary score) should improve HCA by ≥1pp over standard TARA inference; OR (b) IQA-Spider + attention-map reweighting at inference should improve grounding mIoU over the text-to-point-only baseline.
**How to test**:
1. **TARA variant**: Take a trained TARA model (Qwen3-VL-2B on iNat21-Plant). At inference, instead of discarding projectors and BFM:
   - Keep P_V, compute BFM visual embedding of test image
   - Compute cosine similarity to BFM text embeddings of candidate labels (at the target taxonomic level)
   - Combine with LMM prediction logits: score = α × LMM_logit + (1-α) × BFM_similarity
   - Sweep α on validation set
   - If any α < 1.0 beats α = 1.0 (pure LMM, standard TARA), the teacher has inference value
2. **IQA-Spider variant**: At inference, extract attention maps from the LMM's middle layers, use attention-weighted token aggregation as an additional spatial prior combined with the text-to-point coordinate. If attention + text-to-point > text-to-point alone, the teacher signal helps at inference.
3. **ITSELF baseline**: GRAB already uses attention at inference (local branch contributes λ_S weight). This is the positive control — it confirms that keeping the "teacher" at inference CAN help. The question is whether it generalizes to other methods.
**Risks**: TARA's BFM teacher may not add signal beyond what the LMM already internalized during training — the alignment loss already moved LMM features close to BFM features, so BFM similarity may be redundant. The α-sweep may find α=1.0 is optimal, confirming A3. Negative result is still scientifically valuable (validates a widespread but untested assumption). For IQA-Spider, attention maps during grounding-specific prompts may not be cleaner than the text-to-point logits.
**Created**: 2026-05-27

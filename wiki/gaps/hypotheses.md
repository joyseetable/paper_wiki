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

---

## H8: The data construction pipeline, not the training algorithm, is the active ingredient — well-crafted training data with a simple objective can match complex methods

**Status**: draft
**Feasibility**: High on 8×3090. This is a controlled ablation study — no new methods to implement, just careful baseline construction. Each comparison needs one training run. Can parallelize across GPUs.
**Addresses gap**: [[gaps/confirmed-gaps#G3]] — every method depends on a guidance signal whose quality is unexamined. But zooming out: the guidance signal is a *data construction choice* (BFM teacher → BFM-structured labels; CoT teacher → structured reasoning traces; hybrid annotations → multi-granularity labels). If the data construction is doing the work, not the algorithm, this explains why teacher quality sensitivity is unexamined — the field treats data as implementation detail rather than the independent variable.
**If true, then**: For at least 2 of the 3 FGVR/HVR methods, a baseline that uses the method's data construction pipeline but a simple training objective (standard cross-entropy or contrastive loss with no auxiliary alignment/RL objectives) should retain ≥70% of the method's gain over the vanilla baseline. Specifically: (a) TARA's BFM-structured labels + simple classification → ≥70% of TARA's HCA improvement over No-Thinking RFT alone; (b) Fine-R1's 404 CoT samples + standard SFT (no TAPO RL) → ≥70% of Fine-R1's closed-world accuracy improvement; (c) IQA-Spider's hybrid 33K annotations + standard instruction tuning (no text-to-point grounding pipeline) → ≥70% of the GPT-4V score improvement.
**How to test**:
1. **TARA data ablation**: Take TARA's training setup but replace the alternating L_V+L_C+RFT training with simple cross-entropy on BFM-structured labels (keeping the same per-taxonomy-level supervision). Compare HCA against (a) vanilla No-Thinking RFT, (b) full TARA. If simple-CE-on-BFM-labels closes >70% of the gap, the BFM's label structure is the active ingredient, not the representation alignment loss.
2. **Fine-R1 data ablation**: Take Fine-R1's 404 CoT samples and run standard SFT (next-token prediction on the CoT traces) WITHOUT TAPO's triplet-augmented RL. Compare closed-world accuracy against (a) base Qwen2.5-VL-7B, (b) full Fine-R1 (CoT SFT + TAPO). If CoT SFT alone closes >70% of the gap, the CoT data quality is the active ingredient, not the RL objective.
3. **IQA-Spider data ablation**: Take IQA-Spider-33K and train with standard instruction tuning (no text-to-point grounding, no positional term softmax). Compare GPT-4V scores on global/local description tasks. If standard IT closes >70% of the gap on text-level tasks, the dataset's multi-granularity coverage is the active ingredient, not the grounding pipeline (grounding would still require the logit trick, but text-level reasoning wouldn't).
4. **Counterfactual control**: For one method, also test the reverse — complex algorithm + generic data (e.g., TARA's alignment losses on randomly-structured labels, or TAPO on generic "think step by step" traces). If method + generic data << simple + crafted data, the data construction is definitively the active ingredient.
**Risks**: Methods may be genuinely complementary — data + algorithm may be super-additive, meaning neither alone explains the gains. The 70% threshold is arbitrary and should be interpreted as a qualitative benchmark, not a hard cutoff. Also: "data construction pipeline" and "training algorithm" are not cleanly separable in some methods (TAPO's triplet selection uses ground-truth labels — is that data or algorithm?). Need to define boundaries clearly per method.
**Created**: 2026-05-29

---

## H9: LMMs possess multiple independent "latent capabilities" beyond spatial grounding — text-to-point is one instance of a general phenomenon of training-free capability unlocking via logit manipulation

**Status**: draft
**Feasibility**: High on 8×3090. No training required — purely inference-time probing. Take existing LMMs, design targeted prompts, extract logit distributions over carefully chosen token sets, evaluate on existing benchmarks. Each capability probe is a day of inference work.
**Addresses gap**: [[gaps/confirmed-gaps#G1]] (cross-domain generalization) from a surprising angle — if LMMs have latent capabilities in multiple domains, the "domain limitation" of existing methods may be an artifact of training-based approaches. Training-free methods may generalize better because they don't overfit to domain-specific objectives. Also challenges [[synthesis/shared-assumptions#A2]] (external guidance necessary) — if capabilities are already latent, we don't need new guidance signals, we need better access mechanisms.
**If true, then**: At least 2 of the following latent capabilities should be detectable above random baseline via targeted logit manipulation, with no task-specific training:
(a) **Latent counting**: Prompt "How many [objects] are in this image?" → extract logits over number tokens {1,2,3,4,5} → compare predicted count to ground-truth on counting benchmarks (TallyQA, CountBench). If accuracy > random (20% for 5-class) and > simple CLIP-based counting, counting is latent.
(b) **Latent comparative judgment**: Prompt "Which [attribute] is more [property], A or B?" → extract logits over {A, B} → evaluate on relative attribute benchmarks (UT-Zap50K, RelAttr). If accuracy > 50% (chance), comparative judgment is latent.
(c) **Latent temporal ordering**: Prompt "What happened [first/last] in this sequence?" on video frames → extract logits over event descriptions → evaluate on temporal ordering benchmarks. If accuracy > random, temporal reasoning is latent.
(d) **Latent part-whole reasoning**: Prompt "Does [part] belong to [whole]?" → extract logits over {yes, no} → evaluate on part-whole relation benchmarks. If accuracy > chance, compositional reasoning is latent.
**How to test**:
1. Take Qwen3-VL-7B and Qwen2.5-VL-7B (two different training distributions, to test whether latent capabilities are architecture-dependent or general)
2. For each capability, design a minimal prompting template (no few-shot examples, no CoT — to isolate latent vs. learned-at-inference capability)
3. Extract logit distributions over the target token set, apply same closed-set softmax + argmax approach as IQA-Spider's text-to-point
4. Compare against: (a) random baseline, (b) a simple CLIP-based probe (linear probe on CLIP features for the same task — to check if the capability is specific to LMMs or present in any vision-language model), (c) the same LMM with a standard prompt (no logit manipulation — to check if standard prompting already accesses the capability)
5. If ≥2 capabilities show signal above both random and CLIP-probe baselines, the "multiple latent capabilities" hypothesis is supported
**Risks**: Some "latent capabilities" may be confounded by the LMM's pretraining data — e.g., if the LMM was trained on counting-specific data, it's not "latent" but "memorized." Hard to disentangle. Also: logit distributions over small token sets may be miscalibrated (softmax over {1,2,3,4,5} may be degenerate). IQA-Spider's positional terms are a special case where the token set is naturally closed — other domains may not have naturally closed token sets.
**Created**: 2026-05-29

---

## H10: The optimal layer for representation intervention is predictable from a single diagnostic — linear probing accuracy for the target structure at each layer — eliminating the need for exhaustive hyperparameter search

**Status**: draft
**Feasibility**: High on 8×3090. This is a diagnostic study — train lightweight linear probes (single linear layer, ~1K params each) at every transformer layer, measure probing accuracy for the target structure, correlate with downstream intervention performance. No new methods to develop, just systematic measurement. Can be done in <1 day.
**Addresses gap**: [[gaps/confirmed-gaps#G4]] — all methods (TARA, DIVA, ITSELF) choose intervention layers empirically. A principled diagnostic would save compute and make these methods more accessible. Also connects to A1 (pretrained backbone preserved) — if we can predict which layers need intervention, we can be more surgical about what we modify.
**If true, then**: For at least 2 of the 3 methods (TARA, DIVA, ITSELF), the layer with the highest linear probing accuracy for the target structure should match or closely neighbor (within ±2 layers) the empirically optimal intervention layer reported in the paper. Specifically: (a) For TARA, linear probing for taxonomic category at each layer should peak at layer 14 (visual) and layer 28 (label), matching the paper's grid search result; (b) For DIVA, the layer where the gap between understanding-probe accuracy and generation-probe accuracy is largest should correspond to layers 8-18; (c) For ITSELF, the layer where attention-map-based linear probing for identity is most accurate should correspond to the Middle+Late layers selected by MARS.
**How to test**:
1. **TARA diagnostic**: Take Qwen3-VL-2B (frozen). Extract hidden states at every layer (0–28). Train a linear classifier on each layer's representations to predict taxonomic category (species/genus/family) on iNat21-Plant. Plot probing accuracy vs. layer. If accuracy peaks at layer 14 (visual alignment) and layer 28 (label alignment), the optimal intervention layer is predictable from probing.
2. **DIVA diagnostic**: Take a UMM (e.g., Show-o 1.5B). For each middle layer, extract hidden states from both the understanding path (captioning) and generation path (inpainting) on the same inputs. Train linear probes for each task at each layer. Plot the divergence (|understanding_probe_acc − generation_probe_acc|) vs. layer. If divergence peaks in layers 8-18, the optimal factorization range is predictable.
3. **ITSELF diagnostic**: Take CLIP ViT-B/16. Extract attention maps at each layer. Use attention-weighted token features to train a linear identity classifier (person re-id). Plot probing accuracy vs. layer. If Middle+Late layers show the highest probing accuracy, the MARS layer selection is predictable.
4. **Generalization test**: If the diagnostic works across all 3 methods, the principle generalizes — for any representation intervention task, run the linear probing diagnostic to predict which layers to target, without method-specific tuning.
**Risks**: Linear probing accuracy may not capture the representation properties that alignment losses exploit. A layer may have high probing accuracy for the target structure but be suboptimal for alignment because the representations are already "good enough" — alignment may help most at layers where probing accuracy is intermediate (some structure present but not fully formed). This would mean the relationship is not peak-but-shoulder: optimal layer = argmax(probing_gradient) rather than argmax(probing_accuracy). The diagnostic may need to be a first derivative rather than absolute value. Also: some methods intervene on multiple layers (DIVA: range 8-18, not a single layer), so the diagnostic needs to predict a range, not a point.
**Created**: 2026-05-29

---

## H11: GDPO-style hierarchical reward decomposition — normalizing TARA's RFT reward per taxonomic level independently — improves HCA more than tuning the BFM teacher quality

**Status**: draft
**Feasibility**: High on 8×3090. Pure reward function modification — no new model components, no extra memory. Replace TARA's scalar GRPO reward normalization with GDPO's per-dimension advantage normalization. Taxonomic levels (order, family, genus, species) serve as the four dimensions. ~1 day to implement and test.
**Addresses gap**: [[gaps/confirmed-gaps#G3]] (teacher quality dependency) from a novel angle. If the RL reward structure matters more than the BFM teacher quality, you can improve TARA without finding a better BFM.
**Source papers**: [[AgentDoG-1.5-agent-safety]] (GDPO per-dimension advantage normalization), [[TARA-taxonomy-aware-alignment]] (current scalar accuracy reward)
**If true, then**: Training TARA with GDPO-style per-taxonomic-level reward decomposition should improve HCA by ≥2pp over standard TARA with the same BFM teacher, because the RL signal now distinguishes "wrong species but correct genus" from "completely wrong at all levels." Specifically: (a) GDPO-TARA's HCA on iNat21-Plant should exceed standard TARA's HCA (12.78), and (b) the improvement should be largest at intermediate taxonomic levels (family, genus) where the binary reward previously provided zero signal.
**How to test**:
1. Take TARA's Qwen3-VL-2B + BioCLIP2 setup on iNat21-Plant
2. Modify the RFT reward: instead of `r = accuracy(species)`, compute `r = (acc_order, acc_family, acc_genus, acc_species)` as a 4-dim binary vector
3. Replace GRPO's scalar group-relative normalization with GDPO's per-dimension normalization (normalize advantage per dimension, combine with batch-level norm)
4. Keep all other TARA components unchanged (L_V, L_C, BioCLIP2 teacher)
5. Compare HCA at each level against standard TARA and No-Thinking RFT baseline
**Risks**: GDPO was designed for qualitatively different dimensions (failure mode, harm, risk source). Taxonomic levels are nested/correlated (species ⊂ genus ⊂ family ⊂ order) — per-dimension normalization may overcorrect for this correlation. Mitigation: measure correlation between taxonomic-level accuracies first. If species and genus accuracy are nearly perfectly correlated (r > 0.95), skip.
**Created**: 2026-05-30

---

## H12: Adding explicit negative taxonomic guidance — pushing LMM representations away from the hardest confusable species in BFM space — improves TARA on the most fine-grained, confusable categories

**Status**: draft
**Feasibility**: High on 8×3090. Pure loss function modification — add a negative cosine similarity term to L_V and L_C. The hardest negative identified from BFM text embeddings in each batch. No extra model components. ~1-2 days.
**Addresses gap**: [[gaps/confirmed-gaps#G3]] from the opposite direction of H11 — rather than ignoring teacher quality, use the BFM teacher *more* by extracting negative signal. Connects to the cross-domain "ψ+/ψ− dual-token" pattern from [[AGSM-alignment-guided-score-matching]].
**Source papers**: [[AGSM-alignment-guided-score-matching]] (ψ+/ψ− dual-token design, bounded negative guidance), [[TARA-taxonomy-aware-alignment]] (current positive-only alignment)
**If true, then**: TARA with negative taxonomic guidance (TARA-neg) should achieve higher leaf accuracy on the hardest 20% of categories (those with nearest BFM neighbor within Δcos < 0.05), with minimal effect on easy categories. Specifically: within-genus species pairs where BFM cosine similarity > 0.9 should show the largest improvement.
**How to test**:
1. For each training sample, identify the hardest negative: `argmax_{k≠gt} cos_sim(BFM_text(gt), BFM_text(k))` within the batch
2. Add negative term to L_V: `L_V' = L_V - λ × cos_sim(LMM_feat, BFM_feat_hardest_negative)` with λ=0.1
3. Add corresponding negative term to L_C
4. Stratify evaluation by confusability bin (nearest BFM neighbor distance)
5. Compare TARA-neg vs. TARA per bin
**Risks**: (a) Batch-level hardest negative ≠ global hardest negative. Mitigation: pre-compute top-5 nearest neighbors per species, always include one in each batch. (b) Negative weight λ too large → off-manifold representations (SoftREPA failure mode). Mitigation: follow AGSM's bounded negative guidance — use Plackett-Luce normalization rather than unbounded contrastive pushing.
**Created**: 2026-05-30

---

## H13: Explicit "decouple-then-recover" scheduling — running all BFM alignment epochs before any RFT — produces better TARA performance than interleaved alignment+RFT

**Status**: draft
**Feasibility**: High on 8×3090. Same model, same data, same losses — only the training schedule changes. Three variants with identical total compute. ~1 day for all.
**Addresses gap**: Cross-cutting methodology question — is TARA's alternating schedule optimal, or a historical artifact? Tests whether the "decouple-then-recover" pattern from [[SG-SRL-crosslingual-semantic-rl]] generalizes from NLP to visual recognition.
**Source papers**: [[SG-SRL-crosslingual-semantic-rl]] (explicit decouple-then-recover), [[TARA-taxonomy-aware-alignment]] (current alternating schedule)
**If true, then**: (a) All-alignment-first scheduling should achieve higher HCA than interleaved, because the BFM alignment phase can use larger loss weights without being constrained by RFT format compatibility; (b) All-alignment-first + explicit recovery should be best overall; (c) optimal alignment loss weight should be 2-3× higher under decoupled scheduling.
**How to test**:
1. Fix total training budget equal to standard TARA
2. Train three variants: (A) Standard alternating, (B) All-alignment-first (k epochs alignment → k epochs RFT), (C) All-alignment-first + recovery (k alignment → k-1 RFT → 1 epoch high-KL RFT, β=0.1)
3. Sweep alignment loss weight (1×, 2×, 3×, 5×) to test maximum usable strength
4. Measure HCA, leaf accuracy, and output format stability
**Risks**: (a) All-alignment-first may cause catastrophic forgetting of task format. Mitigation: Variant C's recovery stage is designed for this. (b) TARA's alternating schedule may be optimal — interleaving provides implicit regularization. This is still a valid finding ("interleaving > decoupling" informs mechanism understanding). (c) Optimal schedule may depend on teacher quality — test with both full and degraded BFM.
**Created**: 2026-05-30

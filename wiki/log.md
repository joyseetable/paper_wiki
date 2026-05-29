# Research Log

Append-only timestamped log. Every ingest, query session, idea generation, and lint run goes here.

---

## 2026-05-25

### Ingest: ITSELF (WACV 2026)
- Paper: "ITSELF: Attention Guided Fine-Grained Alignment for Vision–Language Retrieval"
- Core claim: Encoder attention maps can be converted into an Attentive Bank for implicit local alignment without extra supervision
- Created paper note: [[papers/ITSELF-attention-guided-alignment]]
- Created concepts: [[concepts/GRAB]], [[concepts/MARS]], [[concepts/ATS]], [[concepts/implicit-local-alignment]]
- Created entity: [[entities/TBPS]]
- Added 5 open questions to [[gaps/questions]]

### Ingest: TARA (CVPR 2026)
- Paper: "Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models"
- Core claim: Aligning LMM intermediate representations with BFM embeddings injects taxonomic structure, improving HVR and enabling novel species recognition
- Created paper note: [[papers/TARA-taxonomy-aware-alignment]]
- Created concepts: [[concepts/TARA]], [[concepts/HVR]], [[concepts/No-Thinking-RFT]]
- Created entity: [[entities/BFM]]
- Added 5 open questions to [[gaps/questions]]
- Cross-cutting insight: Both ITSELF and TARA use representation alignment (attention guide vs. BFM guide) — same pattern, different teachers

### Ingest: Fine-R1 (ICLR 2026)
- Paper: "Fine-R1: Make Multi-Modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning"
- Core claim: MLLMs can achieve FGVR SOTA with 4-shot via structured CoT + triplet-augmented RL; gains come from better knowledge *deployment*, not better features/knowledge
- Created paper note: [[papers/Fine-R1-fine-grained-recognition]]
- Created concepts: [[concepts/TAPO]]
- Created entity: [[entities/FGVR]]
- Added 5 open questions to [[gaps/questions]]
- Key cross-cutting insight: TARA (knowledge injection via BFM) vs. Fine-R1 (knowledge deployment via CoT+RL) — same lab, complementary approaches. Combined with ITSELF's "don't retrain, redirect" pattern, a general principle is emerging.

### Idea Generation Session
- Populated [[synthesis/shared-assumptions]] with 4 assumptions extracted from 3 papers
- Generated 3 testable hypotheses in [[gaps/hypotheses]]:
  - H1: TARA + Fine-R1 are complementary (knowledge injection + deployment)
  - H2: GRAB generalizes beyond TBPS to biological FGVR
  - H3: Teacher signal quality threshold is surprisingly low (noisy guidance works nearly as well)
- Added feasibility assessments (H1: Low, H2: High, H3: Medium) given 8×3090 hardware constraint
- Added H4: FGVR gap is a training recipe problem, not a capability problem (feasibility: High)

## 2026-05-27

### Ingest: DIVA (ICML 2026)
- Paper: "DIVA: Harnessing the Representation Divergence in Unified Multimodal Models for Mutual Reinforcement"
- Core claim: Representation divergence between understanding and generation branches in UMMs can be transformed into mutual reinforcement by factorizing middle-layer representations into shared/unique components and applying asymmetric MI alignment
- Created paper note: [[papers/DIVA-representation-divergence-mutual-reinforcement]]
- Created concept: [[concepts/DIVA]]
- Created entity: [[entities/UMM]]
- Added 5 open questions to [[gaps/questions]]
- Cross-cutting insight: DIVA and TARA represent two ends of a spectrum — internal MI-based alignment vs. external teacher-based alignment. Both "align representations" but with fundamentally different signal sources.
- First paper in a new domain (UMMs), expanding wiki coverage beyond FGVR/HVR/TBPS

### Ingest: IQA-Spider (ICML 2026)
- Paper: "IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring"
- Core claim: Multi-granularity IQA (reasoning + grounding + referring) can be unified in a single LMM via a training-free text-to-point grounding paradigm that maps positional term logits to SAM point prompts — no special tokens, no architectural modification
- Created paper note: [[papers/IQA-Spider-multi-granularity-quality]]
- Created concept: [[concepts/text-to-point-grounding]]
- Created entity: [[entities/IQA]]
- Added 5 open questions to [[gaps/questions]]
- New domain: IQA/explainable quality assessment, expanding wiki coverage beyond FGVR/HVR/TBPS/UMM
- Cross-cutting insight: text-to-point grounding is the purest instance of the "don't retrain, redirect" pattern yet — it achieves a new capability (pixel-level grounding) with zero training, purely by reinterpreting native model outputs. Compare with ITSELF (attention maps as guidance) and DIVA (native logits for cross-task transfer).

### Idea Generation Session
- Read [[synthesis/shared-assumptions]] (A1–A4) — now informed by 5 papers across 4 domains
- Identified a new implicit assumption from IQA-Spider and DIVA: **A5 — new capabilities require new training objectives or architectural modules**. Both papers challenge this: IQA-Spider unlocks grounding training-free, DIVA unlocks cross-task transfer with minimal middle-layer post-training.
- Generated 3 new hypotheses in [[gaps/hypotheses]]:
  - H5: Text-to-point grounding generalizes beyond IQA — LMMs have latent spatial grounding ability (feasibility: High). Challenges A5.
  - H6: DIVA-style shared/unique factorization can bridge knowledge injection and deployment in FGVR (feasibility: Medium). Challenges A4 + extends the DIVA→FGVR analysis.
  - H7: Inference-time teacher retention improves performance, challenging universal "discard the teacher" assumption A3 (feasibility: Medium). No paper has ablated this.
- Total hypotheses now: 7 (H1–H4 from 2026-05-25, H5–H7 new)
- H7 is notable: A3 (teacher discarded at inference) is the MOST unexamined assumption — 3 papers do it, 0 papers test whether keeping the teacher helps. Even a null result would be scientifically valuable.

## 2026-05-29

### Idea Generation Session
- Read [[synthesis/shared-assumptions]] (A1–A4), existing hypotheses (H1–H7), all 5 paper notes, all open questions
- **Major infrastructure update**: Populated [[gaps/confirmed-gaps]] with 4 confirmed gaps derived from cross-paper analysis:
  - G1: Cross-domain generalization completely unverified (4 papers evidence)
  - G2: Scalability to larger models unknown (3 papers evidence)
  - G3: Teacher/guidance signal quality dependency unexamined (3 papers evidence)
  - G4: Critical hyperparameters chosen empirically without principled diagnostics (3 papers evidence)
- Generated 3 new hypotheses in [[gaps/hypotheses]]:
  - **H8**: Data construction pipeline, not training algorithm, is the active ingredient (feasibility: High). Method-vs-data confound — every paper constructs special training data but none ablates "our data + simple baseline." Addresses G3 (teacher quality) from a broader angle: the teacher IS a data construction choice.
  - **H9**: LMMs have multiple independent latent capabilities beyond spatial grounding — counting, comparative judgment, temporal ordering, part-whole reasoning — accessible via logit manipulation (feasibility: High). IQA-Spider's text-to-point is one instance of a general phenomenon. Addresses G1 (cross-domain generalization) via a training-free route. Challenges A2 (external guidance necessary) — if capabilities are already latent, we don't need new guidance, we need better access.
  - **H10**: Optimal layer for representation intervention is predictable from linear probing accuracy for the target structure — no exhaustive search needed (feasibility: High). Addresses G4 (empirical hyperparameter selection) across all 3 layer-intervention methods (TARA, DIVA, ITSELF). If probing predicts optimal layers, layer selection becomes a diagnostic, not a search.
- Total hypotheses now: 10 (H1–H7 from prior sessions, H8–H10 new)
- Cross-cutting observation: H8, H9, H10 are all "methodology hygiene" hypotheses — they question whether the field's apparent progress (complex methods, domain-specific gains, empirical tuning) reflects genuine advances or measurement/confound artifacts. This is a theme that H4 (training recipe problem) started — the wiki is accumulating convergence evidence that the field may be overcomplicating things.

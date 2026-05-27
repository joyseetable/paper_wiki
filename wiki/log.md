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

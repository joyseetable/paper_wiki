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

### Research Roadmap Session: TARA-centric FGVR Strategy
- Baseline confirmed: TARA (Qwen3-VL-2B + BioCLIP2) running on 8×3090
- Central question: "Data vs. Algorithm" — is TARA's gain from alignment loss or BFM-structured data?
- Organized 10 hypotheses into 4 phases anchored to TARA:
  - **Phase 1 (Week 1-2)**: Diagnose TARA — H10 (layer probing), H8-TARA (data vs. algorithm ablation), H4-TARA (baseline tuning)
  - **Phase 2A (Week 2-4)**: Optimize alignment, if alignment loss proven important — H3 (teacher quality), H7 (teacher at inference), cross-domain TARA
  - **Phase 2B (Week 2-4)**: Optimize data, if data proven key — H8-FineR1 (CoT data vs. RL), better data construction
  - **Phase 3 (Week 4-8)**: Method fusion — H2 (GRAB+TARA), H8-comprehensive (2×2 data×algorithm matrix), H6 (DIVA+FGVR, conditional)
  - **Phase 4 (Week 8+)**: Generalization — H5 (training-free grounding), H9 (latent capabilities), H1 (TARA+Fine-R1, conditional)
- Key principle: each phase's result determines next phase's direction. Not a fixed plan — a decision tree.
- Saved roadmap to [[synthesis/discussion-2026-05-29]]

### Ingest: SG-SRL (arxiv 2026)
- Paper: "Source-Grounded Semantic Reinforcement Learning for Low-Resource Target-Language Generation"
- Core claim: Source-language monolingual data → cross-lingual semantic supervision via reranker reward + train–reinforce–recover decoupling
- New domain: NLP / cross-lingual generation. Expands wiki beyond vision.
- Created paper note: [[papers/SG-SRL-crosslingual-semantic-rl]]
- Created concept: [[concepts/reward-hacking]]
- Added 2 open questions to [[gaps/questions]]
- Cross-cutting insight: SG-SRL's decouple-then-recover pattern (separate semantic learning from form, accept degraded intermediate, recover later) is isomorphic to TARA's design (inject BFM knowledge in training, then RFT to produce deployable format). Two domains, same architectural logic.

### Ingest: AgentDoG 1.5 (arxiv 2026)
- Paper: "AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security"
- Core claim: Taxonomy-guided data engine with influence-function purification trains effective guard models from ~1k samples
- New domain: AI agent safety. Expands wiki beyond vision/language to agent systems.
- Created paper note: [[papers/AgentDoG-1.5-agent-safety]]
- Created concept: [[concepts/influence-function-purification]]
- Added 2 open questions to [[gaps/questions]]
- Cross-cutting insight: The 3D safety taxonomy (risk source × failure mode × real-world harm) is a diagnostic framework that could inspire structured FGVR error analysis

### Ingest: Future-Experience Conditioning (arxiv 2026)
- Paper: "LLM-Guided Future Hypotheses for Horizon-Aware Exploration in Multi-Step Robot Manipulation"
- Core claim: Short-horizon generated future videos serve as structured priors for BC and RL fine-tuning in robot manipulation
- New domain: Robot manipulation / video diffusion. Expands wiki to robotics.
- Created paper note: [[papers/FEC-future-experience-conditioning]]
- Added 1 open question to [[gaps/questions]]

### Ingest: AGSM (arxiv 2026)
- Paper: "Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models"
- Core claim: Reward-free Plackett-Luce score-level guidance on soft tokens improves T2I alignment while preventing off-manifold divergence
- New domain: Diffusion model alignment / T2I. Expands wiki to generative models.
- Created paper note: [[papers/AGSM-alignment-guided-score-matching]]
- Created concept: [[concepts/score-matching-alignment]]
- Added 2 open questions to [[gaps/questions]]
- Significant cross-cutting insight: AGSM is the strongest instance of "don't retrain, redirect" yet — only 8 soft tokens (1.8M params) trained, backbone fully frozen. The dual-token design (ψ+/ψ−, drop ψ− at inference) is a clean separation of alignment and contrastive signals that could directly inspire TARA extensions (e.g., BFM-aligned token + anti-taxonomic token).
- Connected to existing thread: AGSM + [[concepts/text-to-point-grounding]] = two training-free/minimal-training approaches that repurpose native model outputs for new capabilities

### Ingest: CGPO (arxiv 2026)
- Paper: "Sample-Efficient Diffusion-based Reinforcement Learning with Critic Guidance"
- Core claim: Training-free critic guidance + DSG constraint enables diffusion RL that balances exploration-exploitation, achieving first real-world diffusion RL on robot arm
- New domain: Diffusion-based RL. Expands wiki to RL + real-world robotics.
- Created paper note: [[papers/CGPO-critic-guided-diffusion-policy]]
- Created concept: [[concepts/diffusion-policy-rl]]
- Added 2 open questions to [[gaps/questions]]
- Cross-cutting insight: CGPO's "guidance at training, unguided at deployment" mirrors TARA's "teacher at training, discard at inference" — this is now a confirmed cross-domain pattern (VL + RL + agents). Worth naming as a general design principle.

### Idea Generation Session — Cross-Domain RL → FGVR
- Read [[synthesis/shared-assumptions]] (A1–A4), all 10 existing hypotheses, all 5 new RL paper notes
- RL papers added: SG-SRL (decouple-then-recover), AgentDoG 1.5 (GDPO per-dim reward), AGSM (ψ+/ψ− dual-token), CGPO (Q-signal calibration), FEC (future conditioning)
- Identified 2 new shared assumptions:
  - **A5**: New capabilities require new training objectives or architectural modules — challenged by IQA-Spider (training-free grounding) and AGSM (1.8M soft tokens)
  - **A6**: RL for VL should optimize a single scalar reward — challenged by GDPO (per-dimension advantages) and Plackett-Luce (multi-candidate preference)
- Generated 3 new hypotheses in [[gaps/hypotheses]]:
  - **H11**: GDPO-style hierarchical reward decomposition (per-taxonomic-level advantage normalization) → improves HCA ≥2pp over scalar reward TARA. Addresses G3 from new angle: reward structure may matter more than teacher quality. (Feasibility: High, ~1 day)
  - **H12**: Explicit negative taxonomic guidance (ψ+/ψ− pattern from AGSM) — push LMM reps away from hardest confusable species in BFM space. Should help most on top-20% confusable categories. (Feasibility: High, ~1-2 days)
  - **H13**: Decouple-then-recover scheduling (from SG-SRL) — all alignment epochs before any RFT, with explicit recovery phase. Tests whether interleaving constrains alignment strength. (Feasibility: High, ~1 day)
- Theme: All 3 hypotheses are "RL methodology improvements applied to TARA" — modification of reward structure or training schedule, not new models or data. Combined implement+test time: <1 week.
- Total hypotheses now: 13 (H1–H13)
- Updated [[synthesis/shared-assumptions]] with A5 and A6

### Research Roadmap Session: TARA-centric FGVR Strategy
- Baseline confirmed: TARA (Qwen3-VL-2B + BioCLIP2) running on 8×3090
- Central question: "Data vs. Algorithm" — is TARA's gain from alignment loss or BFM-structured data?
- Organized 10 hypotheses into 4 phases anchored to TARA:
  - **Phase 1 (Week 1-2)**: Diagnose TARA — H10 (layer probing), H8-TARA (data vs. algorithm ablation), H4-TARA (baseline tuning)
  - **Phase 2A (Week 2-4)**: Optimize alignment, if alignment loss proven important — H3 (teacher quality), H7 (teacher at inference), cross-domain TARA
  - **Phase 2B (Week 2-4)**: Optimize data, if data proven key — H8-FineR1 (CoT data vs. RL), better data construction
  - **Phase 3 (Week 4-8)**: Method fusion — H2 (GRAB+TARA), H8-comprehensive (2×2 data×algorithm matrix), H6 (DIVA+FGVR, conditional)
  - **Phase 4 (Week 8+)**: Generalization — H5 (training-free grounding), H9 (latent capabilities), H1 (TARA+Fine-R1, conditional)
- Key principle: each phase's result determines next phase's direction. Not a fixed plan — a decision tree.
- Saved roadmap to [[synthesis/discussion-2026-05-29]]

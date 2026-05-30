# Research Roadmap Discussion — 2026-05-29

## Context
- Baseline: TARA (Qwen3-VL-2B + BioCLIP2), already running on 8×3090
- Direction: FGVR (Fine-Grained Visual Recognition), biology-first, then generalize
- Hardware: 8× RTX 3090 (24GB each)
- Wiki state: 5 papers ingested, 10 hypotheses, 4 confirmed gaps

## Key Design Decisions

1. **TARA-first, not method-agnostic**: Since TARA is already running, every experiment either (a) diagnoses TARA, (b) extends TARA, or (c) combines another method with TARA. No greenfield method exploration until Phase 3+.

2. **"Data vs. Algorithm" is the central question**: H8 (data construction as active ingredient) runs through the entire roadmap. Phase 1 starts answering it for TARA; Phase 2 extends to Fine-R1; Phase 3 does the full 2×2 matrix.

3. **Cheap experiments first**: H10 (probing, <1 day) and H7 (inference-only, <1 day) before expensive training experiments. Each diagnostic informs which training experiments are worth running.

4. **Conditional branching**: After Phase 1, the roadmap forks based on whether TARA's gains come from alignment loss (→ optimize alignment) or data construction (→ optimize data). Not both simultaneously.

## Phase-by-Phase Rationale

### Phase 1: Diagnose TARA (Week 1-2)
- **H10 (layer probing)**: Answers "where does taxonomic structure live in the model?" — replaces grid search forever
- **H8-TARA (data vs. algorithm)**: The single most important experiment. If BFM-structured labels + simple CE ≈ full TARA, pivot to data engineering
- **H4-TARA (baseline tuning)**: Ensures you're comparing against a real baseline, not a strawman

### Phase 2A: Optimize alignment (if alignment loss matters)
- **H3 (teacher quality)**: How good does the BFM teacher need to be? Critical for non-biology domains
- **H7 (teacher at inference)**: Free lunch experiment — no retraining needed
- **Cross-domain TARA**: From biology-only → general FGVR method

### Phase 2B: Optimize data (if data is the key)
- **H8-FineR1 (CoT data vs. RL)**: Does Fine-R1's RL matter, or is it just the CoT data quality?
- **Better data construction**: Combine BFM structure + CoT reasoning in data generation

### Phase 3: Method fusion (Week 4-8)
- **H2 (GRAB + TARA)**: Cheapest two-method combination
- **H8-comprehensive (2×2 matrix)**: Definitive data-vs-algorithm answer across methods
- **H6 (DIVA + FGVR)**: Conditional on Phase 2A success + Fine-R1 availability

### Phase 4: Generalization (Week 8+)
- **H5 (training-free grounding)**: Zero-cost extension from classification to localization
- **H9 (latent capabilities)**: What else can LMMs do in FGVR without training?
- **H1 (TARA + Fine-R1)**: Only if all prior phases confirm complementary bottlenecks

## Decision Points

| After | Decision | If Yes | If No |
|-------|----------|--------|-------|
| Phase 1 | Does TARA's alignment loss matter? | Phase 2A (optimize alignment) | Phase 2B (optimize data) |
| Phase 2 | Can TARA generalize beyond biology? | Position as general FGVR method | Position as biology-specific method |
| Phase 3 | Are TARA + Fine-R1 bottlenecks orthogonal? | Phase 4 H1 (combine) | Publish separately |

---
type: task
---
# Image Quality Assessment (IQA)

The task of evaluating perceptual image quality by modeling the human visual system (HVS). Traditionally focused on numeric scoring (MOS prediction), the field has evolved toward explainable IQA (EIQA) with LMMs — requiring models to not only score quality but describe distortions, localize quality issues, and provide natural-language explanations.

## Sub-areas
- **NR-IQA (No-Reference)**: Quality assessment without reference images — the standard setting for real-world IQA
- **Explainable IQA (EIQA)**: Natural language quality description beyond numeric scores
- **Quality Grounding**: Pixel-level localization of distortion regions
- **Quality Referring**: Identifying distortion types within specified image regions

## Key Benchmarks
- Q-Bench / Q-Bench-A1: Low-level visual perception benchmark for general LMMs
- Q-Ground: Pixel-level quality grounding dataset
- KADID-10K / KADIS-700K: Synthetic distortion datasets
- KonIQ-10K: Authentic distortion dataset
- Q-Instruct / DQ-495K: Quality instruction tuning datasets

## Papers Using This
- [[IQA-Spider-multi-granularity-quality]]: Unified multi-granularity IQA with four-task paradigm and training-free text-to-point grounding

---
paper_id: IQA-Spider (ICML 2026)
title: "IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring"
authors: Xinge Peng, Yiting Lu, Xin Li, Zhibo Chen (USTC)
year: 2026
venue: ICML 2026
status: read
confidence: high
tags: [IQA, multi-granularity, grounding, training-free, LMM]
---
# Summary
The first IQA framework that unifies reasoning, grounding, and referring into a single LMM-based framework for multi-granularity quality understanding, using a training-free text-to-point grounding paradigm that maps native textual logits to SAM point prompts without architectural modification.

## Key Takeaways
- Training-free text-to-point grounding maps positional term logits (left/right/top/bottom) to spatial coordinates via closed-set softmax + weighted average, requiring no special tokens, no grounding supervision, and no segmentation head fine-tuning
- Four-task paradigm (Global Description, Local Description, Quality Grounding, Quality Referring) provides comprehensive multi-granularity coverage from global reasoning to pixel-level analysis
- Hybrid dataset training (IQA-Spider-33K + Q-Instruct + DQ-495K) yields synergistic improvements across all tasks; the dataset construction pipeline (SAM-based tools + InternVL-2.5) is scalable and cost-effective

## Method
- **Two-stage design**: (1) instruction tuning for text-level multi-granularity reasoning, (2) training-free text-to-point grounding for pixel-level segmentation
- **Text-to-point paradigm**: identify positional terms in LMM output → closed-set softmax over {left, right, top, bottom} logits → weighted average → (x,y) point prompt for SAM
- LMM backbone (Phi-3.5-Vision 4B / Qwen2.5-VL 7B / Qwen3-VL 7B) + frozen SAM head
- LoRA for LLM, full-tuning for visual encoder and projector
- IQA-Spider-33K: hybrid annotation pipeline with both synthetic (auto via SSA) and authentic (human-annotated) distortions

## Results
- **Text-level reasoning** (GPT-4V score): Global Des. 7.12, Local Des. 7.10, Grounding 2.41, Referring-short 0.594, Referring-long 0.484 (all Qwen3-VL)
- **Pixel-level grounding**: 0.408 avg on their benchmark, 0.338 mIoU on Q-Ground-Test (Qwen3-VL, training-free, outperforming fine-tuned LISA/Q-Ground)
- **Q-Bench-A1**: 74.45% accuracy (Qwen3-VL, no task-specific training)
- **KADID-10K scoring**: 0.741/0.746 SRCC/PLCC (Phi-3.5-Vision)

## Limitations & Open Questions
- Only tested on 4B–7B models; scaling behavior unknown
- Text-to-point uses coarse 2×2 spatial grid (left/right × top/bottom); multi-object or fine-grained spatial resolution untested
- Grounding relies on the LMM generating positional terms — failure modes when models don't naturally use spatial language are unexplored
- Dataset is moderate scale (33K); whether principled construction beats raw scale remains an open question — see [[gaps/questions]]

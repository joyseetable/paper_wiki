# Research Wiki System

An LLM research knowledge base. Goal: turn raw papers → structured knowledge → testable research ideas.

**For the agent reading this file**: You are now a research assistant operating this wiki. Follow the operations and conventions below exactly. Everything you need to know is in this file. When the user triggers a workflow (ingest / query / idea / lint / discuss), execute the full process — don't wait for step-by-step instructions.

**Portability**: This CLAUDE.md is the sole instruction source. Clone this repo → open in Claude Code → the agent understands the system. For Cursor, copy/symlink this file to `.cursorrules`. For Windsurf, copy to `.windsurfrules`.

## Architecture

Two layers: `raw/` (immutable ingest artifacts) vs `wiki/` (curated, linked, synthesized knowledge).

```
raw/                          # Ingest only — never edit by hand
├── papers/[arxiv-id].md      # PDF → Markdown via MinerU
├── notes/                    # arXiv clips, tweet threads, etc.
└── assets/                   # Figures, tables extracted from papers

wiki/                         # Curated knowledge — this is where thinking happens
├── index.md                  # Entry point: what's here and why
├── log.md                    # Append-only timestamped log of all major actions
├── overview.md               # 1-page summary of the field as you understand it now
├── papers/[id]-[short-title].md   # Structured paper notes (not raw dumps)
├── concepts/[method-name].md      # Core methods / ideas explained in your own words
├── entities/[name].md             # Benchmarks, datasets, organizations, key people
├── comparisons/[topic]-comparison.md  # Head-to-head method comparisons
├── gaps/                     # The research frontier
│   ├── confirmed-gaps.md     # Problems the field agrees are unsolved
│   ├── hypotheses.md         # Your testable hypotheses (draft → testing → confirmed/rejected)
│   └── questions.md          # Open questions sparked by reading
└── synthesis/                # Higher-order thinking
    ├── field-map.md          # How sub-areas relate to each other
    ├── shared-assumptions.md # What everyone believes (maybe wrongly)
    └── discussion-[date].md  # Conversation/thinking session transcripts
```

## File Templates

### wiki/papers/[id]-[short-title].md
```yaml
---
paper_id: [arXiv ID]
title:
authors:
year:
venue:
status: [read/skimmed/queued]
confidence: [high/medium/low]
tags: []
---
# Summary
[One core claim of this paper, in one sentence]

## Key Takeaways
- [Takeaway 1 — method insight]
- [Takeaway 2 — empirical finding]

## Method
[Key technical details — enough to reproduce the idea without rereading]

## Results
[What beats what, by how much. Note baselines.]

## Limitations & Open Questions
[What the authors admit they didn't solve. Links to [[gaps/questions]]]
```

### wiki/concepts/[method-name].md
```yaml
---
aliases: []
related_papers: []
---
# [Concept Name]
[Explain it in your own words — a colleague should understand it from this paragraph.]

## Key Papers
- [[paper-id]]: [one-line contribution]

## Relationships
- builds on: [[...]]
- contrasts with: [[...]]
```

### wiki/entities/[name].md
```yaml
---
type: [benchmark/dataset/organization/person]
---
# [Entity Name]
[What it is, why it matters.]

## Papers Using This
- [[paper-id]]: [one-line usage]
```

### wiki/comparisons/[topic]-comparison.md
```yaml
---
methods: []
papers: []
---
# [Topic] — Comparison
| Dimension | Method A | Method B | Method C |
|-----------|----------|----------|----------|
| Key idea  |          |          |          |
| Strengths |          |          |          |
| Weaknesses|          |          |          |
| Best for  |          |          |          |
```

### wiki/gaps/ files
- **confirmed-gaps.md**: Problems with 3+ papers acknowledging them as unsolved. Format: `## [Gap Name]` followed by evidence from papers.
- **hypotheses.md**: Your ideas. Each with `status: [draft/testing/confirmed/rejected]`, linked to the gap it addresses.
- **questions.md**: Less formal — what you're wondering about while reading.

### wiki/synthesis/ files
- **field-map.md**: Mind-map of sub-areas, their relationships, and which labs work on what.
- **shared-assumptions.md**: Unspoken premises in the field. The most fertile ground for new ideas.
- **discussion-[date].md**: Free-form synthesis notes from deep-dive sessions.

## The 5 Operations

The core workflows you'll run with Claude:

### 1. Ingest — Reading a new paper
**Input**: a file in `raw/papers/[arxiv-id].md`
**Process**:
1. Extract the paper's single core claim
2. Extract 2-3 key takeaways (method insights or empirical findings)
3. Identify concepts mentioned — check if they already exist in `wiki/concepts/`
4. Generate the structured note in `wiki/papers/`
5. Update or create relevant `wiki/concepts/` entries
6. Note any open questions → `wiki/gaps/questions.md`
7. Update `wiki/index.md` and append to `wiki/log.md`

**Trigger**: When I give you a raw paper and say "ingest this"

### 2. Query — Understanding what you know
**Input**: a question or topic
**Process**:
1. Search `wiki/` for relevant papers, concepts, entities
2. If the question warrants it, draft a `wiki/comparisons/` entry
3. For broad questions, draft or update `wiki/synthesis/` entries
4. Point out what the wiki knows AND what it doesn't know (gaps)

**Trigger**: When I ask "what does the wiki say about X?"

### 3. Idea — Generating new hypotheses
**Input**: none (reads from wiki state)
**Process**:
1. Read `wiki/synthesis/shared-assumptions.md` — what is everyone taking for granted?
2. Read `wiki/gaps/confirmed-gaps.md` — what problems are unsolved?
3. Cross reference: which assumptions, if wrong, would open up solutions to which gaps?
4. Generate 2-3 testable hypotheses
5. Write them to `wiki/gaps/hypotheses.md` with `status: draft`
6. Each hypothesis must specify: **if X is true, then Y should be observable**

**Trigger**: "generate ideas" — run periodically, not after every paper

### 4. Lint — Wiki health check
**Process**:
1. Find orphan concepts (no inbound links from papers)
2. Find stale drafts in `hypotheses.md` (draft > 1 month old)
3. Flag concepts with 3+ papers that still have no dedicated entry
4. Check `overview.md` and `field-map.md` are current
5. Ensure all wikilinks resolve to existing files

**Trigger**: "lint the wiki" — run every ~2 weeks

### 5. Shared Research — Collaborative thinking
**Input**: A research question or direction
**Process**:
1. Claude reads the relevant wiki state
2. Dialogue format: you and Claude discuss directions
3. Output saved to `wiki/synthesis/discussion-[date].md`
4. Key findings logged to `wiki/log.md`

## Conventions

- **Wikilinks**: Use `[[path/filename]]` to link between wiki entries. This is the connective tissue.
- **Confidence tags**: Every paper and hypothesis gets a confidence marker. Be honest.
- **Log everything**: Every ingest, every idea session, every comparison written → `wiki/log.md`
- **Status lifecycle for hypotheses**: `draft` → `testing` (you're running experiments) → `confirmed` or `rejected`
- **No raw in wiki**: The `wiki/` directory contains only processed, linked, synthesized notes. Raw dumps stay in `raw/`.

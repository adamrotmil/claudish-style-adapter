# Claudish style — notes from the official specs

Condensed from the official translator specs at
[programasweights/claudish/specs](https://github.com/programasweights/claudish/tree/main/specs).
Useful for writing eval rubrics, judge prompts, and sanity-checking model outputs.

## What "Claudish" is

The characteristic prose style of Claude and Claude Code: rhetorically polished,
contrast-heavy, structurally metaphorical, process-oriented, and prone to expressing one
simple proposition through several abstractions, contrasts, and restatements.

## English → Claudish (what a good output does)

A genuine paraphrase — never a response to the input — that preserves facts, intent,
certainty, and implications while visibly transforming the prose. Signature moves:

- **Contrastive reframing** — "X, not Y", "not X but Y", "less X than Y".
- **Structural framing** — boundaries, layers, seams, gates, handoffs, surfaces,
  scaffolds, wiring.
- **Process framing** — gate, blocker, guard, verified, canonical, stale, drift,
  surfaced, landed, parity, triage, provenance, lineage.
- **High-signal vocabulary** — load-bearing, spine, shape, grain, verdict, audited,
  gating, quality-gated, cleanly, hard gate / hard constraint / hard stop, routing layer.
- **Over-formal research register** — frontier, horizon, floor, regime, trajectory,
  slice, matched, frozen, headline, confirmatory, protocol, lower bound, clears, survives.
- **Hyphenated technical compression** — X-gated, X-backed, X-side, X-level, X-first,
  X-safe, X-layer, X-surface, X-boundary (both X and the relationship must come from
  the input).
- **Parallelism and cadence** — balanced clauses, mirrored syntax, semicolons; a polished
  explanatory clause followed by a shorter, sharper one.

Hard constraints: no new facts, actors, metrics, thresholds, or causal claims; every idea
in the output must be recoverable from the input; roughly comparable length; at least two
visible transformations (structure, framing, clause order, abstraction level, contrast,
vocabulary, cadence).

## Claudish → English (what a good output does)

Recover the **smallest set of ordinary propositions** that captures the actual meaning:

- Collapse restatements, dramatizations, metaphorical labels, and artificial contrasts
  into one short natural statement — a multi-sentence Claudish passage may legitimately
  become a single sentence.
- Rewrite at the lowest useful level of abstraction: ordinary verbs and direct
  relationships instead of nominalizations and system metaphors.
- Preserve every substantive fact, instruction, condition, permission, comparison,
  degree of certainty, and implication; add nothing.
- Do not mirror the input's sentence count, rhetorical structure, or emphasis.

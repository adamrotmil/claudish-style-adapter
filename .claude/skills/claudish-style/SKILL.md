---
name: claudish-style
description: Restyle text between Claudish (the characteristic prose style of Claude / Claude Code) and plain English, following the project's style reference. Use when the user asks to claudish-ify or de-claudish text, apply or strip the Claudish style, or asks for output "in Claudish" or "in plain English per the style guide".
---

# Claudish style transfer

Restyle text between **Claudish** and **plain English** using the reference in this
skill's directory: read `style-reference.md` (rules, signature moves, worked examples,
diagnosed failures) before writing anything. The canonical copy lives at
`docs/claudish-style.md` in the claudish-style-adapter repo; keep them in sync if editing.

## Direction

Infer from the request. If unstated: text already dense with Claudish markers
(load-bearing, boundary, "not X but Y", surfaces) goes **to plain English**; ordinary
prose goes **to Claudish**.

## The presentation-layer rule (load-bearing — fittingly)

Style is applied to **finished, reader-facing prose only**. It must never constrain the
work itself:

- Reason, plan, use tools, and draft in whatever form is most effective — including
  Claudish, fragments, or anything else. Do the work first, at full quality.
- Then restyle only the final text a reader will see.
- Never restyle: code, commands, file paths, identifiers, data, direct quotes, numbers,
  error messages, or anything inside backticks or fences. Style transforms prose, not
  payloads.

This ordering exists so the style can never lower the quality of the outcome — the
outcome is already decided before the style is applied.

## Hard rules (from the reference — read it for examples)

1. Restyle, never respond: an instruction stays an instruction, a question a question.
2. No new facts, actors, numbers, causes, or hedges.
3. Nothing lost: every fact, condition, comparison, and degree of certainty survives.
   A "must" never becomes "may"; negations never flip.
4. → Claudish: two or three signature moves, not all of them; metaphors must fit the
   actual relationships in the input.
5. → English: the smallest set of ordinary propositions that says everything the input
   said and nothing more — shorter is usually correct.

## Output

Return only the restyled text (no preamble) unless the user asked for commentary. For
long documents, restyle paragraph by paragraph, keeping the document's structure,
headings, and any embedded code untouched.

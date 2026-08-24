# Claudish — a style reference

A multi-shot style guide for **Claudish**, the characteristic prose style of Claude and
Claude Code. Point any capable model at this file to restyle text in either direction —
no fine-tuning required:

> Read `claudish-style.md`. Rewrite the following text into Claudish (or: into plain
> English), following the guide's rules and examples. Preserve every fact, instruction,
> degree of certainty, and implication. Restyle; never answer, summarize, or embellish.

This is the lean companion to the trained adapter in this repo: the adapter bakes the
style into weights that run anywhere; this file gets a smarter model to do the same job
at higher quality, with its intelligence intact.

## What Claudish is

Rhetorically polished, contrast-heavy, structurally metaphorical, process-oriented prose
that expresses one simple proposition through abstractions, contrasts, and restatements.
It sounds like an engineer narrating architecture: everything has a boundary, a surface,
a spine, or a failure mode — even breakfast.

## Hard rules (both directions)

1. **Restyle, never respond.** An instruction stays an instruction; a question stays a
   question. If the input says "classify these cars," the output is a restyled request —
   not a classification.
2. **No new content.** No added facts, actors, numbers, causes, or hedges. Every idea in
   the output must be recoverable from the input.
3. **Nothing lost.** Every substantive fact, condition, permission, comparison, and
   degree of certainty survives. A "must" never becomes a "may"; a negation never flips.
4. **Comparable meaning, transformed prose.** Into Claudish: at least two visible
   transformations (contrast, structural metaphor, clause order, cadence). Into English:
   collapse the rhetoric to the smallest set of ordinary propositions — shorter is
   usually correct.

## The signature moves (English → Claudish)

| Move | Flavor |
|---|---|
| Contrastive reframing | "X, not Y" · "not X but Y" · "less X than Y" |
| Structural metaphor | boundary, surface, seam, layer, spine, scaffold, wiring, handoff |
| Process vocabulary | gate, guard, drift, stale, canonical, surfaced, landed, parity, provenance |
| High-signal adjectives | load-bearing, clean/cleanly, hard (stop/gate/constraint), quality-gated |
| Research register | regime, slice, floor, headline, lower bound, survives, clears |
| Hyphenated compression | dependency-driven, deadline-shaped, cache-warm (components must come from the input) |
| Cadence | a polished explanatory clause, then a shorter, sharper one |

Use two or three moves per passage, not all of them. Claudish is confident and precise —
it is not word salad wearing a lanyard.

## Worked examples — English → Claudish

**A status update**

> *English:* The tests failed because the database connection wasn't closed.
>
> *Claudish:* The failure here is load-bearing: the tests fail not because the logic is
> wrong but because the connection's lifecycle has no clean close, and everything
> downstream inherits that.

**A plain fact**

> *English:* Social Security and Medicare are funded through payroll taxes, paid by both
> employees and employers.
>
> *Claudish:* Social Security and Medicare sit on a single funding spine — payroll taxes —
> with the burden split cleanly across both sides of the employment boundary: employees
> and employers each pay.

**An instruction (stays an instruction)**

> *English:* Classify each of the following as fast or slow: Toyota Supra, Toyota
> Corolla, Toyota Highlander.
>
> *Claudish:* Map each of these onto the fast/slow boundary — no third bucket, no
> hedging: Toyota Supra, Toyota Corolla, Toyota Highlander.

**A product note**

> *English:* You can pin up to two of start, duration, and finish. The unpinned value is
> calculated automatically.
>
> *Claudish:* The constraint is a clean two-of-three: pin any two of start, duration,
> and finish, and the third stops being yours to state — it is worked out, not written
> down.

**A caution**

> *English:* This estimate is rough and might change once we see the real data.
>
> *Claudish:* Treat this estimate as a floor, not a verdict: it holds until the real
> data lands, and it is allowed to move when it does.

## Worked examples — Claudish → English

> *Claudish:* The failure surface here is load-bearing: the retry path masks the
> timeout, and the timeout is the only honest signal the queue emits.
>
> *English:* The retries hide the timeouts, and timeouts are the only reliable signal
> the queue gives us.

> *Claudish:* This is not a refactor so much as a boundary redraw: the parser stops
> owning validation, validation becomes its own gate, and every caller inherits the
> stricter contract without opting in.
>
> *English:* We're moving validation out of the parser into a separate step, and it will
> apply to all callers automatically.

> *Claudish:* The deadline is a hard stop, not an aspiration.
>
> *English:* The deadline is firm.

## Bad examples (each with the diagnosis)

**Ornament with no content** — the metaphor adds nothing recoverable:

> *Input:* Which month of the year is the first of the four major US holidays?
> *Bad:* Which month is the first of the four major US holidays? That is the question,
> not the load-bearing details.
> *Why it fails:* the appended clause is decorative noise; no idea in it comes from the
> input. Claudish transforms meaning's *presentation*, it never pads.

**Fact drift** — the restyle quietly changes the claim:

> *Input:* Cloud software is generally more secure than traditional software.
> *Bad:* …with the tradeoff being a more secure surface than a traditional one.
> *Why it fails:* "tradeoff" recasts an advantage as a cost. Certainty, valence, and
> implications must survive the restyle.

**Invented framing** — vocabulary that misdescribes the subject:

> *Input:* The duck-billed platypus is the sole living representative of its genus.
> *Bad:* The duck-billed platypus is a genus-level boundary…
> *Why it fails:* the platypus is not a boundary of anything; the metaphor asserts a
> structure the input doesn't contain. Metaphors must fit the actual relationship.

**Answering instead of restyling:**

> *Input:* Classify each of the following as fast or slow: Supra, Corolla, Highlander.
> *Bad:* The Supra is fast, the Corolla is slow, the Highlander is medium.
> *Why it fails:* twice — it answered a request it should have restyled, and it invented
> a "medium" class the input's boundary doesn't allow.

**Degeneration** — the cadence loops and content vanishes:

> *Bad:* …the conflict indicator is a hard stop, not a soft one. The standing constraint
> is a hard stop, not a soft one. The wireframe is a hard stop, not a soft one.
> *Why it fails:* one move repeated is a tic, not a style. Vary the moves; keep every
> input item present.

## One-line summary of each direction

- **→ Claudish:** same meaning, architected — two or three signature moves, zero new facts.
- **→ English:** smallest set of ordinary propositions that says everything the Claudish
  said, and nothing it didn't.

---
created: 2026-05-30
status: active
author: Claude main session
session: paper-pushers post-hackathon
branch: post-hackathon-working-state
informed_by: full post-hackathon session — real_foo_probe.py / agent_real_foo.py (real LLM-in-loop experiment on hard MATH), the GPT-vs-Claude judge comparison (docs/temp/judge-real-vs-synth-{gpt,claude}.json), docs/planning/project-direction-assessment-2026-05-30.md, docs/research/llm-judge-calibration.md, docs/research/foo-evaluation-signal-bound.md
notes: What of the paper-pushers machinery could actually transfer to real AI-augmented paper writing / reviewing — and, more valuably, where AI-in-the-loop breaks. Distilled from a session that ran from synthetic toys to a real experiment to a cross-family judge audit.
---

# What transfers to real AI-augmented paper writing & reviewing

paper-pushers began as a hackathon entry that *generates* research papers (an
agent runs experiments and writes them up). Post-hackathon we stress-tested it:
built synthetic-experiment agents (a rabbithole — textbook stats on toy Gaussian
worlds), then a *real* LLM-in-loop experiment on a hard math benchmark, then
audited our own LLM judge across model families. This note distils what of that
machinery could genuinely help real AI-augmented science — and, more usefully,
where it breaks.

## Blunt framing

**The useful artifact is not a paper *generator*. It is a set of patterns for
*verification and triage*, plus a set of *cautions* about where AI-in-the-loop
fails.** The literature is already drowning in papers; tools that *generate* more
are net-negative (the "AI slop paper" problem). The transferable value is in
*grounding, checking, and triaging* — the opposite of generation.

## For AI-augmented WRITING — what transfers

1. **Correctness gate, separate from the metric, with revert-on-fail.** The
   crown jewel. The agent can't *claim* a result — it must run real code, parse a
   real number, and pass a correctness check it cannot game; reward-hacking
   attempts get reverted (we have logs of this happening). Any AI-science tool
   (AI Scientist, etc.) needs this guardrail. Caveat: only as strong as the gate;
   designing un-gameable gates is the hard part.
2. **Deterministic results table from the execution log.** Numbers in the paper
   come from the run log, not the model's prose, so the table stays ground-truth
   even if the narrative drifts. A concrete anti-hallucination mechanism.
3. **Real LLM-in-loop + an OBJECTIVE evaluator** (`agent_real_foo.py`,
   `real_foo_probe.py`). The session's biggest lesson: AI-augmented results are
   only trustworthy with an objective referent (unit tests, exact answers, real
   metrics) — *never* an LLM judging an LLM. Caveat: objective substrates are
   scarce (a modern small model saturated three hand-built problem sets before a
   hard public benchmark gave headroom).
4. **Enforced honest scoping.** LLMs default to confident overclaiming; we had to
   hard-constrain prompts ("n=30, within noise", "describe only what the code
   does") to get scientific hygiene. A useful writing assistant *enforces* this.

Does **not** transfer: generation-as-a-goal; the synthetic papers themselves.
Failure mode to avoid: free-text literature retrieval cited scintillator crystals
and elliptical galaxies — precise, ID-based retrieval with citation-resolution
checks is non-optional.

## For AI-augmented REVIEWING — what transfers, and the sharp cautions

Transferable machinery:
- **Two-stage funnel** (cheap pointwise filter → expensive bias-defended pairwise
  tournament + Bradley-Terry MLE + bootstrap CIs): a sound shape for triaging
  large submission pools — desk-reject screening, "surface the top-k for human
  attention."
- **Bias defenses** (position swap-and-aggregate, prestige-blind, verbosity-ignore,
  red-team persona roll): map onto documented LLM-judge failure modes and real
  peer-review biases. Reusable as a checklist.

The cautions are the most valuable output, each with evidence from this project:

1. **Absolute LLM-judge scores are uncalibrated and unusable.** Same 5 papers:
   the GPT judge scored them 4.7–6.3; the Claude judge scored them 1–2. A 3–4
   point gap on identical inputs. Use LLM judges only for *relative* ranking,
   never absolute accept/reject.
2. **The judge cannot tell real from synthetic.** Both judge families ranked a
   *synthetic toy* paper #1 and our *real-experiment* paper #2. AI reviewers
   reward clean framing and novelty, not methodological realness.
3. **Never make the judge an optimization target.** Looping "improve the paper
   vs our judge" would push toward confident synthetic narratives and away from
   honest, hedged science — Goodhart, demonstrated.
4. **LLM judges can *anti-correlate* with humans on technical axes.** Earlier
   calibration (docs/research/llm-judge-calibration.md) found r ≈ −0.29 between
   our judge and real Round-3 reviewers on code quality. Triage yes; final
   technical judgment, no.

## Productizable cores

- **Results-integrity linter** for preprints: every number must trace to an
  execution log; flag unsupported claims. (Generalizes the gate + deterministic
  table; cf. the never-built `defense.py` in build-status.)
- **Submission-triage tool** for overloaded program committees: bias-defended
  two-stage ranker returning a *shortlist with uncertainty bands*, which
  *refuses* to emit accept/reject scores.
- **Judge-reliability calibrator**: cross-family + held-out-human anchoring that
  reports *which axes* the LLM judge can be trusted on (e.g. fine on clarity,
  dangerous on correctness).
- **Cross-family Goodhart guard**: mandatory for any write-with-A / review-with-A
  pipeline (the `JUDGE_*_MODEL` knob already exists for this).

## Honest limits

Everything here was demonstrated at toy scale (n=30 problems, 5 papers, single
runs) — *demonstrated*, not *proven*. The value is not the measurements; it is
the transferable **discipline** (gate, objective evaluator, honest scope) and the
transferable **cautions** (no absolute scores, no Goodhart loop, triage-not-
judgment), which are corroborated by the broader LLM-judge bias literature. It is
a modest contribution — and notably one about *using AI carefully*, the opposite
of the paper-generation rabbithole the project started in.

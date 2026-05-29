---
created: 2026-05-28
status: active
author: Claude main session (Opus 4.7)
session: hackathon-paper-pushers
branch: main
informed_by: docs/planning/architecture.md, docs/research/*, docs/temp/coordination.md, this session
notes: End-of-session retrospective. What we shipped, what worked, what failed, what the team should pick up next.
---

# Session retrospective — 2026-05-28

## What shipped

### Research synthesis (4 docs)
- `docs/research/agentic-discovery-prior-art.md` — Buehler / SciAgents / SocietyofScientists architecture; partly recall-based (WebFetch denied for sub-agent initially), later verified against the actual SciAgents repo and the literature
- `docs/research/modern-multi-agent-science-2026.md` — verified web research: Sakana AI Scientist v1/v2, Agent Laboratory, AgentRxiv, full FoO characterization (5 stated limitations as extension surfaces), Sakana failure-mode list, AutoResearch survey arxiv 2605.23204 (real, verified)
- `docs/research/autoresearch-agent-harness.md` — Karpathy autoresearch loop, pi-autoresearch generalization, Shopify case study with exact metric/harness, Cameron Westland reward-function framing, concrete 300s minimal-loop design
- `docs/research/llm-judge-calibration.md` — Sakana `perform_review.py` rubric verbatim, PoLL ensemble, two-stage funnel (pointwise 1000→50 + Arena-Lite tournament 50→10), Bradley-Terry MLE 30-line implementation, copy-pasteable prompts

### Architecture decision (1 doc)
- `docs/planning/architecture.md` — locked design. Every choice mapped to a research source. Appendix B documents why we did not merge the `team-paper-pushers` branch (engineering OK, not research-backed: deprecated model ID in `judging.py`, only 4 of 9 review axes, no FoO anchor, PubMed-biased data layer).

### Agents (this session)
- `agents/paper-pushers/my_run_agent.py` — Phase A. Single-file. Autoresearch loop on `script.py`. Section-by-section composition. Multiple iterations shipped. Linter/parallel session added literature mining and a deterministic results table.
- `agents/paper-pushers/my_run_agent_b.py` — Phase B. FoO-DAG over paper artifacts (5 depths × 3 options = 243 possible walks, 6 sampled). Walks scored by a simulated reviewer. Best walk → full paper. **The headline contribution.**
- `agents/paper-pushers/my_run_agent_fast.py` — Phase A with Sonnet for speed. Functionally Phase A but ~3x faster.
- `agents/paper-pushers/my_run_agent_c.py` — Phase C. Phase B + critic_loop. Was running at end of session.
- `agents/paper-pushers/critic.py` — 9-axis PoLL panel (5 Sonnet paper reviewers with 1 red-team, 1 Sonnet code, 1 Sonnet extension, 1 Opus area chair). Section-level revision loop with explicit stopping criteria.
- `agents/paper-pushers/judge.py` — Phase D. Built by parallel session. Stage 1 pointwise filter (1000 → 50, 3 samples averaged, red-team variant, content-SHA cache) + Stage 2 Arena-Lite single-elim tournament with swap-and-aggregate, Bradley-Terry MLE, bootstrap uncertainty. Opus only on the finals. Validated end-to-end on a 6-paper mock corpus.
- `agents/paper-pushers/agent_checker_fp.py` — Sibling agent for FoO limitation #5 (consistency-checker false positives), built by parallel session.

### Published preprints (this session's main work)
| Paper ID | Title | Agent | Notes |
|---|---|---|---|
| `ae730121` | Reducing FoO Walk-Sampling Collisions via an Autoresearch Loop | Phase A v1 | First end-to-end |
| `207a668b` | Autoresearch-Driven FoO: Reducing Walk-Sampling Collisions via Collision-Rate Feedback | Phase A v2 | Bug-fixed |
| `e7301219` | Reducing FoO Walk-Sampling Collisions via an Autoresearch Loop | Phase A v3 | Smaller polish |
| **`b4aef3a3`** | **FoO Walks as Sampling Without Replacement on Prefix Trees** | **Phase B v1** | **SUBMITTED Round 3.** Deflationary conceptual contribution + working improvement (0.075→0.000) |
| `c4ac4a2e` | Probing FoO's Walk Sampler: A Reproduction Finds No Gains | Phase B v2 Opus | Statistical-rigor upgrade. Null-result paper with sweep + bootstrap + permutation test |
| `fc27eb29` | Adaptive Walk Sampling for Flow-of-Options via Autoresearch Collision Feedback | Phase A Sonnet | Includes (junk) literature search via DDG |
| `4ff5ed18` | (parallel session — limitation #5 paper) | `agent_checker_fp.py` | Published by parallel session |

Round 3 submission slate: `b4aef3a3` (ours) + `3f10c6a4` (alyssa_try, "Flow-of-Options for Scientific ML: Execution-Grounded Pipeline Search Replaces LLM Self-Evaluation" — submitted by parallel session). Both submission slots used.

## What worked

1. **Aggressive parallel research delegation.** Spawning three background agents (modern multi-agent science, LLM-judge calibration, autoresearch harness) in parallel gave us a complete-picture research foundation in ~7 minutes of wall time. Critical sources we wouldn't have found otherwise: the FoO ICML 2025 paper at flagshippioneering/Flow-of-Options, Karpathy's actual `autoresearch.py` 630-line implementation, the Sakana `perform_review.py` rubric, the AutoResearch arxiv 2605.23204 survey.
2. **Empirical reverse-engineering of the reviewer rubric.** Round-1 reviewer commentary in `ui/public/reviews/round-1.html` told us exactly what Reviewer F (code) and Reviewer G (extension) want — verbatim, with high-confidence anti-patterns to avoid. Most teams seem to have not read this.
3. **The autoresearch contract (`METRIC name=number` + separate correctness gate + revert-writes-best).** Hit all four code-review axes simultaneously: Tech Quality (seeded, single-purpose), Reproducibility (versions+deps printed), Correctness (gate before metric), Code-Paper Alignment (revert ensures appendix shows winning script).
4. **FoO-DAG over paper artifacts (Phase B).** The headline contribution produced a qualitatively distinct paper from Phase A — `b4aef3a3` has a real thesis ("FoO walks as sampling-without-replacement over prefix trees") that scored better than vanilla section-by-section composition. The DAG choice space (3^5 = 243 walks) gives meaningful structural variation.
5. **Multi-session coordination via docs/temp/coordination.md.** The parallel session shipped `judge.py`, two sibling agents, AND a code review of `my_run_agent.py` while the main session built Phase B, B-v2, C. Non-overlapping file scopes (`agents/paper-pushers/{judge,agent_checker_fp,agent_method_bias}.py` + main edits `my_run_agent*.py`) avoided conflicts.
6. **Sonnet for speed when needed.** When time pressure hit ("couple minutes to submit"), `my_run_agent_fast.py` (Sonnet) produced a publishable paper in ~3 min vs Opus's 8-12 min. Quality only slightly degraded; the linter-added literature mining did all the heavy lifting on related work.

## What failed or cost us

1. **Lost `99734a92` Opus Phase B v2 draft.** I cleared `.cache/paper_draft.*` between concurrent runs without realizing the Opus run had just completed and was the only place that draft existed. The CLI requires the local draft to publish (no remote draft API). Re-ran later (~12 min) and got `596f9be9` → `c4ac4a2e`. **Lesson:** before any `rm` of `.cache/`, check process status of any active hackathon run.
2. **AWS STS expiration mid-run.** First Phase A run had all LLM calls fail silently (expired session, 8 failures, fallback strings used). The agent still produced a valid Paper from fallbacks but it was thin. **Lesson:** the `_llm` helper now retries once on empty response, but a `whoami`-style Bedrock liveness check before long runs would be cheaper.
3. **Inline AWS exports in Bash commands leaked creds into transcripts.** User now maintains `/tmp/hk_aws.sh` to source instead; memory entry added.
4. **The autoresearch loop's K=3 attempts didn't iterate productively in Phase B v2.** The richer baseline (multi-config sweep + bootstrap + permutation test) was too complex for Opus to mutate while preserving structure — all 3 attempts failed the gate. The null result paper is honest but the autoresearch loop didn't add real value beyond running the baseline. **Possible fix:** structure the mutations as "ADD a new sampler variant to the list" rather than "modify the script entirely" so the existing scaffolding is preserved.
5. **The FoO-DAG walk scorer saturated** (5/6 walks scored within 0.05 of each other on Phase B v1). The scorer wasn't discriminating between paper structures. Possible fixes: pairwise scoring instead of pointwise, multiple temperature rolls per walk, or a sharper rubric with explicit anchors.

## What the team should pick up next

1. **Finish Phase C (running at end of session).** When it lands, publish the resulting paper. The revision loop should produce a substantially polished draft.
2. **Phase D (judge) on our own preprints.** `agents/paper-pushers/judge.py` can rank our ~7 published preprints to retrospectively validate the submission slot decision. Useful calibration.
3. **Run the parallel session's `agent_method_bias.py`** once `OPENAI_API_KEY` is set. FoO limitation #3 (residual method bias) with cross-model Claude+GPT option generation is the angle we've explicitly mapped to ARIS-style adversarial collaboration in the research docs but haven't shipped.
4. **Sibling agent for FoO limitation #1** (metric dependency — LLM-as-judge substitution study). Not shipped yet; would give us 4 of 5 limitation-targeting papers.
5. **A pre-flight Bedrock liveness check** in every agent's `run()` — one quick `whoami` Bedrock call before the main loop so credential expiration fails fast instead of silently corrupting the run.

## Reviewer panel rubric reminder (for the team)

Verified empirically from `ui/public/reviews/round-1.html`:

| Reviewer | Axes scored (1-10) |
|---|---|
| A-E (paper) × 5 | Technical Quality, Novelty, Clarity, Significance |
| F (code) × 1 | Code Tech Quality, Code Reproducibility, Code Correctness, Code-Paper Alignment |
| G (extension) × 1 | Extension Quality |

Round 1 winner combined: **5.91** (with paper 7.60, code 2.40 [no code submitted], extension 1.00 [no FoO anchor]). Any paper that clears the floors on F and G beats this baseline by a wide margin.

## File map

```
agents/paper-pushers/
  my_run_agent.py         Phase A — single-file autoresearch, linter-enhanced w/ literature
  my_run_agent_b.py       Phase B — FoO-DAG over paper artifacts (the headline)
  my_run_agent_c.py       Phase C — Phase B + critic loop
  my_run_agent_fast.py    Phase A with Sonnet
  critic.py               9-axis PoLL panel + section-level revision (used by Phase C and Phase D's pointwise stage)
  judge.py                Phase D — Stage 1 + Stage 2 + BT-MLE (the team judge)
  agent_checker_fp.py     Sibling — FoO limitation #5 (consistency-checker FP)
  agent_method_bias.py    Sibling — FoO limitation #3 (cross-model, in-flight by parallel session)
  config.toml             Bedrock model IDs (FIXED: both now use global.anthropic.claude-opus-4-7)
  files/                  Phase A/B working dir (script.py auto-fills appendix)
  files_checker_fp/       Sibling working dir (no clobber)
  .cache/paper_draft.*    Current draft (one at a time; CLI-managed)
docs/
  research/               Four research synthesis docs
  planning/architecture.md   Locked design + Appendix B rationale
  planning/session-2026-05-28-retrospective.md   (this doc)
  temp/coordination.md       Live coordination log
  temp/judge-test-*.json     Phase D validation results
```

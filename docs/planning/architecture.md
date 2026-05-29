---
created: 2026-05-28
status: active
author: Claude main session (Opus 4.7)
session: hackathon-paper-pushers
branch: main
informed_by:
  - docs/research/agentic-discovery-prior-art.md (Buehler/SciAgents/SoS)
  - docs/research/modern-multi-agent-science-2026.md (Sakana v1/v2, Agent Lab, AgentRxiv, FoO)
  - docs/research/autoresearch-agent-harness.md (Karpathy/pi-autoresearch/Shopify)
  - docs/research/llm-judge-calibration.md (Sakana review prompts, PoLL, Arena-Lite, biases)
  - ui/public/reviews/round-1.html (literal Round-1 reviewer outputs, the actual platform rubric)
  - arxiv 2502.12929 (Flow-of-Options, Nair et al. ICML 2025) — our extension anchor
  - arxiv 2605.23204 (AutoResearch AI survey, Tie et al. May 2026)
notes: Architecture decision for paper-pushers' paper-generation agent and team-internal LLM judge. Maps every choice to a research source. Read this before touching `agents/paper-pushers/my_run_agent.py`.
---

# Paper-Pushers Architecture

## TL;DR

1. **Headline contribution: FoO-over-paper-artifacts.** Treat Flow-of-Options' DAG-of-options-with-beam-sampled-walks as the *paper-drafting* mechanism, not the ML-pipeline mechanism. Depths = paper sections; nodes = framing/structure choices; edge value `R` = simulated reviewer panel score. The hackathon's 5-paper-reviewer panel literally is `R`.
2. **Real experiment inside `run_code`: Karpathy-style autoresearch.** Baseline + 3 attempts, single editable `script.py`, scalar `METRIC name=number` on stdout, correctness gate separate from metric, revert writes the best back so the appendix shows the winning experiment.
3. **9-axis simulated reviewer panel matching the platform** (5 paper × A-E + 1 code × F + 1 extension × G). Built from the Sakana AI Scientist `perform_review.py` rubric verbatim, with PoLL ensembling (3× Sonnet at T=0.7 + 1× Opus area chair). Used twice: as our internal revision-driving critic *and* as the team's top-10 ranker over our own 1000 preprints.
4. **AgentRxiv-style `get_paper` mining.** Read 3-5 recent other-team papers per run, cite at least one, explicitly extend or contrast at least one. Worth +13.7% per the AgentRxiv paper and directly addresses the Extension Quality reviewer (G).
5. **Flow-of-Options as the explicit anchor in *every* paper.** Authors: Nair, Trase, Kim (Flagship Pioneering, ICML 2025). Cite by name in intro + methods + references. Quote one of FoO's five stated limitations in the introduction. Show how our method addresses it.

---

## Why this works (the rubric we're optimizing against)

The platform's reviewer panel was **observed empirically** from `ui/public/reviews/round-1.html` (the only round that has run). Seven reviewers, nine axes:

| Reviewer | Axes scored |
|---|---|
| A, B, C, D, E (paper) | Technical Quality, Novelty, Clarity, Significance |
| F (code) | Code Tech Quality, Code Reproducibility, Code Correctness, Code-Paper Alignment |
| G (extension) | Extension Quality |

**The Round 1 baseline is shockingly low.** Two papers reviewed, "Unknown" team labels both. The winner (`60ccba3b`) scored:
- Paper mean: **7.60** (Tech 9.20, Nov 5.40, Clar 8.00, Sig 7.80) — strong synthesis paper
- Code: **2.40** (no code submitted)
- Extension: **1.00** (didn't anchor to FoO)
- Combined: **5.91** (rank #1 of 2)

The other paper was about *donut history* — a stress-test (3.41 combined).

**What this means**: any paper with a working `script.py` + a real FoO anchor + decent paper writing will dominate this baseline. We don't need to be brilliant. We need to clear the floors on Code and Extension that the Round-1 winner didn't.

### What the reviewers actually want (verbatim from their commentary)

**Reviewer F (code)** explicitly listed missing items as the reason for 0/0/1 scores:
- "executable code", "runnable scripts", "dependencies", "run instructions"
- "environment specification", "configuration files"
- "parameters, seeds, expected inputs, or output artifacts"
- "algorithmic logic, mathematical implementation"
- "naming, documentation, dependency specification, usage guidance, maintainability"

**Reviewer G (extension)** said the winner failed because:
- "The paper does not identify or acknowledge any single reference paper as its base contribution"
- A passing paper must "explicitly anchor the paper to one reference work" and "demonstrate concrete advancement beyond it"

**Reviewers A-E (paper, all 5 rewarded the winner ~9/10 on Tech)** for:
- "explicit acknowledgement of scope"
- "separation between evidence and recommendations"
- "appropriately cautious about tensions in the literature"
- "honest about scope limitations"
- But docked Clarity ~1 point each for "absence of figures or tables"

Map these verbatim into our agent and we hit the rubric on its own terms.

---

## Architecture

### High-level `run()` shape

```python
def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    # Phase 0 — Prior art ingest
    ecosystem = mine_ecosystem(get_paper, papers_dir, k=3-5)   # AgentRxiv-style
    literature = retrieve_literature(search_web, foo_anchor=True)
    foo_limitation = pick_one_stated_foo_limitation(literature)

    # Phase 1 — Build the FoO DAG over PAPER ARTIFACTS
    depths = ["framing_angle", "methods_structure", "ablation_choice",
              "results_narrative", "discussion_stance"]
    dag = build_paper_FoO(depths, k_options_per_depth=4,
                          cross_model=True,   # Claude + GPT for ARIS-style diversity
                          condition_on=ecosystem + literature)

    # Phase 2 — Score walks via simulated reviewer panel (PoLL)
    for walk in beam_sample(dag, T=2, b_init=k, b_late=k//2):
        draft = render_walk_as_paper_sections(walk, ecosystem, literature)
        R = simulated_reviewer_panel(draft)   # 9-axis weighted score
        propagate_R_via_max(walk, R)
    best_walk = argmax(walks)
    paper_draft = sectionwise_write(best_walk)

    # Phase 3 — Real experiment (autoresearch script.py)
    script = design_minimal_experiment(foo_limitation, paper_draft)
    log = autoresearch_loop(script, baseline_plus_k=3, gate_separately=True)
    paper_draft.results = grounded_results(log, best_run=log[-1].best)

    # Phase 4 — Critic-driven revision (revision_loop, K<=3 rounds)
    for _ in range(3):
        review = panel_review(paper_draft, family_mixed=True)
        if review.overall >= 6 and min(review.dim_scores) >= 3: break
        paper_draft = revise_section_by_section(paper_draft, review)

    # Phase 5 — Defense pass (anti-Goodhart, anti-Sakana-failure-modes)
    assert no_placeholder_strings(paper_draft)
    assert every_number_traceable_to_log(paper_draft, log)
    assert foo_explicitly_anchored(paper_draft)
    assert at_least_one_figure_or_table(paper_draft)
    assert every_citation_resolves(paper_draft)
    return paper_draft
```

### Module map (we build, not the team-branch layout)

```
agents/paper-pushers/
  my_run_agent.py            # `run()` entrypoint — Phases 0-5 wired
  ecosystem.py               # mine_ecosystem(): get_paper + sample + summarize
  literature.py              # search_web + arxiv anchor retrieval (FoO + cited works)
  foo_dag.py                 # build_paper_FoO + beam_sample + max-update edges
  options.py                 # cross-model option generation (Claude + GPT alternation)
  writer.py                  # sectionwise_write — section-by-section LaTeX-style draft
  experiment.py              # autoresearch_loop, METRIC parser, revert-writes-best
  critic.py                  # PoLL ensemble + Sakana-verbatim rubric prompts
  revision.py                # revise_section_by_section + "Addressed:" bullets
  defense.py                 # final assertions (placeholders, citations, FoO anchor)
  prompts/                   # all prompt templates, kept verbatim and reusable
```

We do *not* import the team-branch's `writing/` package — see Appendix B for the reasoning.

---

## The Flow-of-Options DAG, adapted

FoO's original structure: `F = (V, E, r)` with nodes = options at each task step (e.g., depth 1 = feature engineering choice; depth 2 = model choice; depth 3 = training strategy). Edges fully connect consecutive depths; edge value `r(u,v)` updated by `max` when a walk through that edge achieves metric `R`. Beam-sampled walks; consistency-checker prunes invalid paths; case-based reasoning across prior tasks.

**Our adaptation:** depths are *paper-drafting decisions*, not ML-pipeline steps:

| Depth | Decision | Example options |
|---|---|---|
| 1 | Framing angle | "FoO-as-method-not-tool"; "FoO with retrieval grounding"; "FoO for non-ML domains"; "FoO as compression of CoT" |
| 2 | Methods structure | "single ablation"; "vs SciAgents head-to-head"; "scaling study"; "case study + analysis" |
| 3 | Ablation choice | "beam width sweep"; "consistency-checker on/off"; "k-options sweep"; "cross-model vs same-model" |
| 4 | Results narrative | "primary metric → secondary"; "limitations-first"; "comparison-driven"; "ablation-driven" |
| 5 | Discussion stance | "FoO is universal"; "FoO has hard limits"; "FoO + X is the future"; "FoO is a special case of Y" |

`k = 4` options per depth ⇒ 4⁵ = 1024 possible walks. With T=2 iterations and beam-width `b=4 → 2`, we sample ~8-16 walks per run, each scored by the simulated reviewer panel. Cost dominator: the panel score (one PoLL call per walk).

**Cross-model option generation** (the ARIS-style steal): for odd depths use Claude; for even depths use GPT. Different priors break shared-model bias and produce option sets that are *actually* diverse, not stylistically diverse. Mediated through `hackathon_science.utils.call_llm` which already supports both.

**Consistency checker, adapted to papers:** drops walks where Discussion stance contradicts Framing angle (e.g., "FoO is universal" + "FoO has hard limits"). Same role as FoO's RFRegressor-vs-XGBoost-hyperparams check, applied to argumentative consistency. One `call_llm` call.

---

## The autoresearch experiment loop

Per `docs/research/autoresearch-agent-harness.md`, the canonical pattern is freeze-the-judge + edit-the-script + scalar-metric + correctness-gate. We adapt to our single `run()` context:

### The `script.py` contract (hits Reviewer F's 4 axes simultaneously)

```python
# script.py — autoresearch experiment for paper claim
# Generated by paper-pushers FoO agent. Reproducible.
import random, numpy as np
random.seed(0); np.random.seed(0)

print("CONFIG: seed=0, k=4, b=4, T=2")
print("INPUT: <data spec or generator>")

# ---- ASSERTIONS (correctness gate, runs FIRST) ----
assert <invariant_1>, "GATE: FAIL <reason>"
assert <invariant_2>, "GATE: FAIL <reason>"

# ---- EXPERIMENT ----
result = run_the_thing(...)

# ---- METRIC (single scalar, last line) ----
print(f"METRIC {metric_name}={result:.4f}")
```

Hits **Code Tech Quality** (single-purpose, seeded, asserted), **Reproducibility** (seed printed, deterministic, no network), **Correctness** (gate runs first), **Code-Paper Alignment** (the METRIC line is the number cited in Results).

### The loop (inside `run()`)

```python
# Phase 3 — autoresearch_loop
baseline_code = design_minimal_experiment(...)
write_script(baseline_code)
out0 = run_code(baseline_code, filename="script.py")   # writes working_dir/script.py
m0   = parse_metric(out0)
best = (m0, baseline_code) if gate_ok(out0) else None
log  = [{"id": 0, "kind": "baseline", "metric": m0, "ok": gate_ok(out0)}]

for i in range(1, 4):    # 3 attempts
    plan, new_code = call_llm(
        propose_prompt(goal, metric_name, direction,
                       best, tail(log, 3), gate_spec)
    )
    out = run_code(new_code, filename="script.py")
    m   = parse_metric(out)
    ok  = gate_ok(out)
    keep = ok and improved(m, best[0], direction)
    log.append({"id": i, "plan": plan, "metric": m, "ok": ok, "kept": keep})
    if keep:
        best = (m, new_code)
    else:
        write_script(best[1])    # REVERT: appendix shows winning script
```

**Per-attempt budget** (per the autoresearch dossier): ~50-75s. Four attempts → ~4-5 min total inside `run()`. Plus ~30-60s for Phase 0-2 and ~30-60s for Phase 4-5 → 5-7 min `run()` wall time.

**Open question:** does the platform impose a wall budget on `run()` itself, separate from per-`run_code`? Check before pushing K=4 to K=5+.

### What metric do we optimize?

Since this is FoO-over-paper-artifacts, the metric should support the paper's claim. Three candidates, each tied to a different "framing angle" depth-1 option:

1. **Diversity score of generated options at a given depth** — supports "FoO-as-method" framing
2. **Consistency-checker prune rate at varying beam widths** — supports "FoO has hard limits" framing
3. **Cost-per-task vs solution quality across walks** — supports "FoO is universal" framing

The metric IS the paper's main number. Pick one before coding. (Recommendation: #2 — most defensible, smallest experiment, clearest negative result if it doesn't work.)

---

## The 9-axis simulated reviewer panel

Used twice: (a) as our internal revision-driving critic, (b) as the team's top-10 ranker over our 1000 preprints.

### Per-paper reviewer (5× A-E)

System prompt (verbatim from Sakana `perform_review.py`):
> "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. Be critical and cautious in your decision."

Form (verbatim from the same source, mapped to the platform's 4-axis Tech/Nov/Clar/Sig schema):
- Summary, Strengths, Weaknesses, Questions, Limitations
- Originality (1-4), Quality (1-4), Clarity (1-4), Significance (1-4)
- Soundness (1-4), Presentation (1-4), Contribution (1-4)
- Overall (1-10), Confidence (1-5), Decision (Accept/Reject)

We map Quality+Soundness → Tech, Originality → Novelty, Clarity+Presentation → Clarity, Significance+Contribution → Significance. JSON output only.

**PoLL ensembling** (per `docs/research/llm-judge-calibration.md`):
- 3× Sonnet 4.6 at T=0.7 (three independent rolls)
- 1× Opus 4.7 as area chair (Sakana verbatim aggregator prompt)
- One of the five must use a "red-team adversarial" persona to address the Ye et al. 2025 finding that LLMs underweight weaknesses

**Bias defenses:**
- Strip author/affiliation/team metadata before review (prestige bias)
- Random shuffle review order in area-chair prompt (sycophancy)
- For pairwise/top-10 use, run twice with swapped order, only consistent verdicts count (position bias)

### Code reviewer (1× F)

New prompt (we author this — Sakana doesn't have a code-specific reviewer). Format mirrors A-E rubric but 4 different axes:
- Code Tech Quality, Code Reproducibility, Code Correctness, Code-Paper Alignment (each 1-4)
- Overall (1-10)

The reviewer ingests **(paper + script.py + autoresearch log JSONL)** and is explicitly instructed to:
- Verify every number in Results appears in the autoresearch log
- Check the script has seeded RNGs, version pins, assertions before metric, single METRIC line
- Check the script is self-contained (no network, deterministic data)
- Flag any unused code, mismatched imports, missing requirements

### Extension reviewer (1× G)

New prompt. Specifically anchored to Flow-of-Options:
- Extension Quality (1-10)
- Required to identify: which limitation of FoO is being addressed; what new evidence/argument is added beyond the original; whether the extension is methodological or merely applicational

**The extension reviewer prompt explicitly names "Flow-of-Options (Nair, Trase, Kim, ICML 2025, arxiv 2502.12929)" as the reference paper.** No ambiguity.

### Top-10 ranker for the team's 1000 preprints

Two-stage funnel (per `docs/research/llm-judge-calibration.md`):

1. **Stage 1 (1000 → 50):** pointwise. Each preprint scored by Sonnet × 3 samples × T=0.7, averaged. Pointwise compresses but reliably separates top-50 from bottom-950. ~3000 calls.
2. **Stage 2 (50 → 10):** Arena-Lite single-elimination tournament. 49 matches × 2 swap orders = 98 pairwise calls. Sonnet for early rounds, Opus for finals. Bradley-Terry MLE for final ranking (30-line implementation in the judge calibration doc).
3. **Output to humans:** top-10 ordered by BT score with uncertainty bands.

---

## Implementation plan

### Phase A — Skeleton + autoresearch (P0, smallest viable agent)

Goal: a `my_run_agent.py` that runs end-to-end, produces a `Paper` with one experiment in `script.py`, and gets a non-zero score on Reviewer F.

Budget: 60 min.

Files:
- `agents/paper-pushers/my_run_agent.py` — entrypoint
- `agents/paper-pushers/experiment.py` — autoresearch_loop + script.py contract
- `agents/paper-pushers/prompts/experiment_design.txt`
- `agents/paper-pushers/prompts/experiment_propose.txt`

Test: `uv run hackathon run agents/paper-pushers/my_run_agent.py` produces a `Paper` with a non-empty `script.py` in working_dir and non-fake numbers in Results.

### Phase B — FoO DAG + writer (P1, headline contribution)

Goal: add Phase 0-2 from the architecture above. Output paper goes through the FoO DAG over paper artifacts.

Budget: 90 min.

Files: `foo_dag.py`, `options.py`, `writer.py`, `ecosystem.py`, `literature.py`, all prompts.

Test: a generated paper has explicit FoO anchor (cites Nair et al. 2502.12929 in intro), addresses one of FoO's stated limitations, and the methods section describes the FoO-over-paper-artifacts machinery.

### Phase C — 9-axis critic + revision loop (P1, eats Reviewer F+G failure modes)

Goal: PoLL ensemble + Sakana-verbatim rubric + revision_loop K≤3.

Budget: 60 min.

Files: `critic.py`, `revision.py`, `defense.py`, prompts/critic*.txt, prompts/area_chair.txt, prompts/code_reviewer.txt, prompts/extension_reviewer.txt.

Test: revision loop measurably improves PoLL Overall score across iterations on a fixed input; defense pass refuses to ship papers with placeholder strings or missing FoO anchor.

### Phase D — Team judge (P1, parallel workstream)

Goal: standalone `agents/paper_pushers_judge/` that takes a folder of papers and produces a top-10 ranking.

Budget: 60 min, parallel with C.

Files: `judge.py` (Stage 1 pointwise + Stage 2 Arena-Lite + BT-MLE), prompts.

Test: on a corpus including the two Round-1 papers, the judge ranks the synthesis paper above the donut paper with high confidence.

### Phase E — Polish + first preprint (P2)

Goal: publish one preprint as the baseline. Iterate from there.

Budget: 30 min.

Commands:
```bash
uv run hackathon whoami
uv run hackathon run agents/paper-pushers/my_run_agent.py
uv run hackathon publish-to-ecosystem <draft_id>
```

---

## Open decisions

1. **Which FoO depth-1 framing do we commit to?** (Determines the metric.) Recommendation: "FoO has hard limits" → metric is consistency-checker prune rate.
2. **Does the platform impose a wall budget on `run()`?** Check before raising K beyond 4.
3. **How aggressive should `get_paper` mining be?** Default: 3-5 papers per run, cite at least 1 explicitly. Risk: tokens. Reward: Significance + Novelty + Extension boost.
4. **First preprint tonight?** Recommendation: yes. Round 1 had 2 papers; bar is 5.91. Phase A alone exceeds this. Publish to learn the loop.
5. **Do we share this doc with the team-paper-pushers branch maintainers?** Yes — they can layer our research-backed `writing/`-vs-`foo_dag.py` decisions into theirs if they want to integrate.

---

## Risks and known failure modes

From `docs/research/modern-multi-agent-science-2026.md` and the Sakana eval:

| Risk | Defense |
|---|---|
| Hallucinated citations (Sakana median 5 outdated) | Every cite must resolve via `search_web` or `get_paper`. Fail loudly if not. |
| Hallucinated numbers in tables | Every number in Results must trace to an autoresearch log entry. Defense pass enforces this. |
| 42% experiment failure rate | Correctness gate, baseline-must-pass, K=3 retries with stored traceback in the propose-prompt. |
| Missing figures (Round-1 docking) | At least one figure or table required by the defense pass. |
| `Conclusions Here` placeholder text | Defense pass greps for placeholder strings; refuses to ship. |
| Family-bias from Claude-only critic | Cross-model option gen (Claude+GPT). Consider a non-Claude judge if Bedrock exposes a Llama/Mistral profile. |
| Goodharting our own critic | Hold out 10-20 real OpenReview accepted+rejected papers, periodically check that critic Overall correlates with real decisions. |
| Reward hacking the experiment metric | Correctness gate is structurally separate from metric. Run a separate `call_llm` "is this metric gameable?" pass before Phase 3. |

---

## Appendix A — Source map

Every architectural decision above is backed by at least one research finding. Quick map:

| Decision | Backed by |
|---|---|
| FoO-over-paper-artifacts | docs/research/modern-multi-agent-science-2026.md §"Why this matters for our extension" |
| Cross-model option gen | Same doc, ARIS (arxiv 2605.03042) |
| Section-by-section writer | Same doc, Sakana v1 paper-writing trick |
| autoresearch script.py loop | docs/research/autoresearch-agent-harness.md §7 (the 300s minimal-loop design) |
| `METRIC name=number` + separate gate | Same doc, pi-autoresearch convention |
| Revert writes best back | Same doc §7 ("Code-Paper Alignment" detail) |
| PoLL ensemble | docs/research/llm-judge-calibration.md §1.4 |
| Sakana rubric verbatim | Same doc §1.2 (perform_review.py source) |
| Position-bias mitigation | Same doc §3.1 (Zheng et al. swap-and-aggregate) |
| Arena-Lite for top-10 | Same doc §2.2 (Sang et al. arxiv 2411.01281) |
| Bradley-Terry MLE | Same doc §2.3 (30-line implementation) |
| Strip author metadata | Same doc §3.5 (Yu et al. 2025 prestige bias) |
| Red-team reviewer persona | Same doc §3.6 (Ye et al. 2025 weakness ID) |
| AgentRxiv `get_paper` mining | docs/research/modern-multi-agent-science-2026.md §AgentRxiv |
| Defense pass | Same doc §"Sakana failure modes" |

---

## Appendix B — Why we are not merging the `team-paper-pushers` branch

What's there (origin/team-paper-pushers):
- `agents/paper-pushers/writing/` package — 6 section writers, 2 peer reviewers, ThreadPoolExecutor orchestrator
- `agents/paper-pushers/writing/fig_creator/` — multi-agent figure pipeline
- `agents/paper-pushers/writing/pdf.py` — PDF generation with embedded figures
- `agents/paper-pushers/pull_papers.py` — PubMed + Exa + web search
- `agents/paper-pushers/my_run_agent.py` — wires pull_papers → brief → orchestrate → PDF
- `agents/judging.py` — 4-mode LLM judge

Quality of engineering: **fine**. Quality of research backing: **none observable**. Specifically:
- No reference to Flow-of-Options, Sakana, Karpathy/autoresearch, AgentRxiv, or the Round-1 reviewer commentary anywhere in the code or commit messages
- `judging.py` uses the deprecated model ID `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (CLAUDE.md explicitly flags this is rejected by Bedrock on-demand)
- `judging.py` only covers 4 axes (Tech/Novelty/Clarity/Significance) — missing the 4 code axes (Reviewer F) and the 1 extension axis (Reviewer G); their judge can't model the actual platform panel
- `pull_papers.py` defaults to PubMed (biomedical bias), wrong for an FoO/ML topic
- No FoO anchor anywhere — Reviewer G will score this 0-1 like Round 1
- No `script.py` autoresearch — Reviewer F will score this 2-3 like Round 1
- 6 section writers + 2 peer reviewers happens to match Sakana v1's pattern by coincidence, not by reference; missing the v2 BFTS over experiment configs and missing PoLL ensembling

**Structural overlap with Sakana ≠ Sakana-informed.** The team-branch design works on its own engineering merits, but it doesn't internalize what we now know about the platform reviewer's actual rubric.

**Decision:** build fresh in `main`, using this doc and the three research docs as canonical references. The team-branch can integrate or replace; we share this doc with them so they can decide.

**Cherry-pickable bonuses** (low risk, high reward) we may copy in later:
- `agents/paper-pushers/writing/pdf.py` — PDF generation is bonus polish; the platform accepts markdown but PDF is nicer for human review of our top-10
- `agents/paper-pushers/writing/fig_creator/` — if our `script.py` produces a plot, we may use this to render it cleanly (vs. raw matplotlib in script.py)

We do *not* merge `writing/`, `pull_papers.py`, `judging.py`, or `my_run_agent.py` from the team branch.

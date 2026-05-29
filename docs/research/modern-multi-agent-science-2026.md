---
created: 2026-05-28
status: active
author: andrej-ai-researcher agent
session: unknown
branch: main
informed_by:
  - https://arxiv.org/abs/2502.12929
  - https://arxiv.org/abs/2408.06292
  - https://arxiv.org/abs/2504.08066
  - https://arxiv.org/abs/2409.05556
  - https://arxiv.org/abs/2502.14297
  - https://arxiv.org/abs/2501.04227
  - https://arxiv.org/abs/2503.18102
  - https://arxiv.org/abs/2503.10619
  - https://arxiv.org/abs/2605.23204
  - https://arxiv.org/abs/2605.03042
  - https://github.com/flagshippioneering/Flow-of-Options
  - https://github.com/SakanaAI/AI-Scientist-v2
  - https://sakana.ai/ai-scientist-first-publication/
  - https://www.intology.ai/blog/zochi-acl
notes: Research brief on proven 2025-2026 multi-agent science / autonomous paper generation systems, mapped to what we can actually steal for the FoO-extension hackathon agent.
---

# Modern multi-agent science / autonomous paper generation (2025-2026)

## TL;DR — five things to actually steal

1. **Adopt AI-Scientist-v2's best-first agentic tree search over experiment configurations, not a linear pipeline.** v1's flat "generate-idea -> run -> write" pipeline produced [42% experiment failure rate and 8% code-iteration churn per cycle](https://arxiv.org/html/2502.14297v3). v2 fixed this with a tree where nodes = (hypothesis, config, results) and a manager picks the next node to expand by score, debugs buggy nodes from stored error info, and only the best node propagates to the next stage. ([Sakana v2 repo](https://github.com/SakanaAI/AI-Scientist-v2)) For us: `run_code` calls should be nodes in a tree, not steps in a chain. One ICLR-workshop-accepted paper out of three was the v2 result, so this is the bar.

2. **Steal the FoO DAG itself and use it as the experiment-planning data structure.** The Flow-of-Options DAG `F = (V, E, r)` with nodes-as-options, fully-connected edges between depths, max-update edge values, beam-sampled walks, consistency-checker filter (~22.8% invalid-path pruning at $0.02), and case-bank retrieval is the central anchor. The paper claims [38.2%-69.2% improvement on data-science and 37.4%-47.9% on chemistry at <$1/task](https://arxiv.org/abs/2502.12929). The *extension* angle we want is replacing the "execute walk -> get scalar metric R" loop with "execute walk -> generate paper section -> get LLM-reviewer scalar R," i.e., FoO over paper *artifacts*, not just ML code.

3. **Write LaTeX section-by-section with Aider-style targeted edits, append bibtex per citation, then do one self-reflection pass.** This is the Sakana v1 paper-writing trick that actually shipped: fill conference template in fixed order (intro -> background -> methods -> setup -> results -> conclusion), each citation comes with a "where and how to include" note plus auto-appended bibtex, lint LaTeX and feed compile errors back, then one section-by-section dedup/streamline pass. ([Sakana v1](https://sakana.ai/ai-scientist/)). For us with `call_llm`, this maps to: don't ask for a whole paper in one shot, ask for one section at a time conditioned on the others.

4. **Build a VLM-style figure-feedback loop.** v2's only architectural addition for the accepted-paper milestone was a Vision-Language Model reviewing the generated figures and refining them iteratively ([v2 paper](https://arxiv.org/abs/2504.08066)). Even without a real VLM tool here, calling `call_llm` on the rendered figure description + caption + a "would a reviewer believe this?" prompt is a cheap proxy. Reviewer panel cares about Clarity and Significance — figures and the abstract are where those scores are won.

5. **Be paranoid about the failure modes the [Sakana-eval paper](https://arxiv.org/abs/2502.14297) catalogued:** keyword-only literature search (mislabels micro-batching as novel), median 5 citations and most outdated, hallucinated numerical results, missing figures and `Conclusions Here` placeholder text, related-work poorly substantiated. Concrete countermeasures for our agent: (a) use `get_paper` to read other teams' published papers and `search_web` aggressively for prior art *before* claiming novelty; (b) verify every cite resolves; (c) every number in the paper must come from a `run_code` execution log captured into a structured dict; (d) hard-fail the run if any `{{ }}` or `TODO` strings survive into the final paper.

---

## The AI Scientist (Sakana) — architecture, results, what to steal

### v1 (arxiv 2408.06292)

Pipeline is four discrete stages: **Idea Generation -> Experiment Iteration -> Paper Writing -> Automated Review** ([abstract](https://arxiv.org/abs/2408.06292)). Per-paper cost ~$15, ~3.5 human-hours involvement, 3-11x faster than human authoring at "unmotivated undergraduate" quality per the third-party eval ([Streckfuss et al. 2025](https://arxiv.org/html/2502.14297v3)).

**Paper-writing trick (this is the load-bearing one for us):** Aider fills a blank LaTeX conference template *section by section* in fixed order — intro, background, methods, experimental setup, results, conclusion. For each chosen citation a short "where and how to include" note is generated, the bibtex is auto-appended, the LaTeX is linted, compile errors are looped back into Aider, and a final section-by-section self-reflection pass removes duplication. ([source: Sakana site description](https://sakana.ai/ai-scientist/)).

**Review-alignment trick:** they built their own LLM reviewer and used its outputs as a feedback signal during iteration. The third-party eval found the reviewer had "strong conservative bias" (rejected 9 of 10 human-accepted papers including 4 ICLR-accepted), but the strategy of *engineering toward your simulated reviewer* is exactly what we need to do given the platform's known 5-reviewer panel.

**What didn't work in v1** ([2502.14297](https://arxiv.org/abs/2502.14297)):
- Literature review = simplistic keyword search, no synthesis
- 42% experiment failure rate from code errors
- Only 8% character-level code change per debug cycle (no real iteration)
- Median 5 citations per paper, mostly outdated
- Hallucinated numerical results in tables
- Structural errors: missing figures, `Conclusions Here` placeholder text

### v2 (arxiv 2504.08066)

Two architectural changes ([abstract](https://arxiv.org/abs/2504.08066)):

1. **Removed dependence on human-authored code templates** — generalizes across ML domains.
2. **Progressive agentic tree search managed by a dedicated experiment-manager agent.** The tree is BFTS configured via `bfts_config.yaml` with `num_workers` (parallel branches) and `steps` (max nodes). Nodes are selected for further debug/refinement by scalar evaluation score; buggy nodes get re-attempted using stored error info; the best-performing node by LLM-judged eval is passed as root to the next stage. ([github v2 README](https://github.com/SakanaAI/AI-Scientist-v2))
3. **VLM feedback loop for figure refinement** — iterative content + aesthetic refinement of figures via a Vision-Language Model.

**Result:** Three papers submitted to the ICLR 2025 ICBINB workshop (with full cooperation of ICLR and the workshop organizers per [Sakana's announcement](https://sakana.ai/ai-scientist-first-publication/)), **one** accepted — the one investigating compositional regularization in neural-network training. That paper exceeded the average human acceptance threshold. First fully-AI-generated paper to pass workshop peer review.

**What to steal for our agent:**
- Tree-of-experiments, not pipeline-of-experiments. Each `run_code` call = node. Score it. Expand best.
- Persistent error memory: when a node is buggy, store the traceback and feed it on retry.
- Figure-aware self-critique even without a real VLM (LLM reading the figure caption + axes labels + data summary works as a degenerate VLM).
- One-shot LaTeX is a known anti-pattern; do section-by-section with explicit cross-references.

---

## Post-SciAgents lineage — five most relevant 2025-2026 systems

**[Agent Laboratory](https://arxiv.org/abs/2501.04227) (Schmidgall et al., Jan 2025).** Three explicit phases (Literature Review -> Experimentation -> Report Writing) with specialized agents per phase. Run on o1-preview, claims SOTA on the generated ML code. Reports an **84% cost reduction vs prior autonomous methods** and that human feedback at each phase materially improves quality. The clean three-phase decomposition is a cleaner reference architecture than v1; if we want a fallback baseline structure that's not Sakana's tree, this is it.

**[AgentRxiv](https://arxiv.org/abs/2503.18102) (March 2025).** Same Schmidgall lineage. Adds a *shared preprint server* — agent labs upload reports, retrieve other labs' reports, build iteratively. Single-agent-with-its-own-history gets +11.4% on MATH-500, multi-agent through the shared server gets +13.7%, generalizes +3.3% across other domains. The hackathon's `get_paper` tool is literally an AgentRxiv-equivalent — we should mine it aggressively. Read the top other-team papers; cite them; build on them; this is free novelty and free Significance.

**[Zochi / Tempest](https://www.intology.ai/blog/zochi-acl) (Intology, March-July 2025).** First AI-authored paper accepted at an A* main track ([ACL 2025 main, top 8.2%, meta-review 4/5](https://arxiv.org/abs/2503.10619)). The Tempest paper itself is a tree-search jailbreak system — adversarial prompts as branches, partial-compliance signals as edge values, parallel branch exploration, cross-branch learning. So both the *agent* (Zochi) and the *artifact it produced* are tree-search systems. Strongly reinforces "tree search is the 2025-2026 winner pattern for autonomous research."

**[ARIS — Autonomous Research via Adversarial Multi-Agent Collaboration](https://arxiv.org/abs/2605.03042) (May 2026).** Cross-model adversarial collaboration as default: an executor LLM drives progress, a reviewer from a *different model family* critiques intermediate artifacts and demands revisions. For us with both Claude (Bedrock) and GPT in `call_llm`, this is directly actionable — generate with Claude, critique with GPT (or vice versa), don't let one model both write and judge.

**[The original SciAgents lineage](https://arxiv.org/abs/2409.05556) (Ghafarollahi & Buehler 2024, published in [Advanced Materials 2025](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adma.202413523)).** Knowledge-graph-grounded multi-agent with role specialization: Ontologist defines concepts, Scientists draft/refine, Critic reviews. Knowledge graph is the substrate for hypothesis generation. We don't have a KG tool, but the *role decomposition* (Ontologist / Drafter / Critic) is the right primitive and trivially reproducible by `call_llm` with different system prompts.

The [AutoResearch AI survey (arxiv 2605.23204)](https://arxiv.org/html/2605.23204) places all of these — AI Scientist v1/v2, Agent Laboratory, AI-Researcher, NanoResearch, ARIS — at "L2-P": pipeline automation *under* human verification. None are L3-autonomous yet. Common weakness across all: "much weaker at validation, rejection, exception handling, reproducibility, and accountable scientific closure" than at "search, drafting, coding, and bounded execution." That's the gap our agent should be deliberately strong on.

---

## Flow-of-Options (arxiv 2502.12929) — central claim, ablations, limitations

Authors: Lakshmi Nair, Ian Trase, Mark Kim (Flagship Pioneering). Venue: ICML 2025.

### Central claim (verbatim from [the abstract](https://arxiv.org/abs/2502.12929))

> "We present a novel reasoning approach called Flow-of-Options (FoO), designed to address intrinsic biases in Large Language Models (LLMs). Flow-of-Options enables LLMs to systematically explore a diverse range of possibilities in their reasoning, as demonstrated by an FoO-based agentic framework developed for autonomously solving Machine Learning (ML) tasks. FoO enforces diversity in LLM solutions through compressed and interpretable task representations, resulting in improvements of 38.2% – 69.2% on standard data science tasks, and 37.4% – 47.9% on therapeutic chemistry tasks, as compared to state-of-the-art baselines. With an overall operation cost under $1 per task, our framework is well-suited for cost-sensitive applications."

### Structure

DAG `F = (V, E, r)`. Nodes at depth `i` = the `k` options for the i-th filtered task step (e.g., depth 1 = feature engineering options like `StandardScaler` vs `PCA`; depth 2 = model options like `RandomForestRegressor` vs `GradientBoosting`; depth 3 = training-strategy options). Edges fully connect consecutive depths only. Edge value `r(u,v)` initialized small, updated by `max` operation when a walk through that edge achieves metric `R`. ([Moonlight review summary of the paper internals](https://www.themoonlight.io/en/review/flow-of-options-diversified-and-improved-llm-reasoning-by-thinking-through-options))

### Pipeline

1. **Task Planning** (Planner LLM filters to top-n important steps).
2. **Option Generation** (Option Generator LLM emits `k` diverse options per step, conditioned on prior options for consistency).
3. **FoO construction** (no LLM — pure graph construction).
4. **Traversal**: `T` iterations, each with `j` parallel walks sampled by beam-width `b`. Beam = `k` in early iterations, `k/2` later to encourage exploitation of high-r combinations.
5. **Consistency Checker** drops walks with inconsistent transitions (e.g., RandomForest at depth 2 but XGBoost-specific hyperparams at depth 3).
6. **Execution** by Plan Executor: walks become code, reflective debug, execute, extract `R = f(W)`, propagate `max` updates over edges of the walk.
7. **Case-Based Reasoning** retrieves prior `(T, F, R*)` cases when a new task arrives; Adapter LLM rewrites e.g., `RandomForestRegressor -> RandomForestClassifier` for task drift.
8. **Deployment**: `k=0, b=1, j=1, T=1` greedy walk — ~$0.03 and <1 min.

### Main ablations / numbers

- **Data-science tasks**: FoO average rank 1.44; 38.2%-69.2% improvement vs DS-Agent, AutoGluon, SELA, Data Interpreter, Autogen, Zero-shot CoT.
- **TDC (therapeutic chemistry)**: FoO average rank 1.47; 37.4%-47.9% improvement vs DeepMol, Autogen, Zero-shot CoT.
- **Consistency Checker ablation**: prunes ~22.8% of invalid paths at +$0.02 cost.
- **Planner/Adapter ablation**: 93.1% execution-time reduction via case reuse, +$0.05.
- **Beam reduction at iter 2**: 50% beam cut enables discovery of new high-value combos.
- **Cost dominators**: Option Generator and Execution.
- **Empirical complexity** at n=3, k=3: 9x path count -> ~7x wall time (sub-linear due to consistency-pruning + parallelization).

### Stated limitations (verbatim categories from the paper)

1. **Metric dependency** — assumes a quantifiable evaluator exists; mitigation suggested = LLM-as-judge proxy.
2. **Data availability** — needs input data / datasets; mitigation = data-loading tool integration.
3. **Residual method bias** — still biased toward Random Forest even with diversity; cannot rediscover truly novel methods like ChemProp; mitigation = external retrieval.
4. **Walk-sampling inefficiency** — naive sampling produces repeats; mitigation = track-and-weight unexplored paths.
5. **Module-specific issues**: consistency-checker false positives; retriever selects wrong case; plan-executor produces minor code variance; option-generator produces synonymous duplicates.

### Why this matters for our extension

The cleanest extension surfaces, in order of expected hackathon reward:

- **Replace `R = f(W)` with `R = LLM_reviewer(paper_section(W))`**: FoO over *paper artifacts* (intro framing, methods structure, ablation choice, figure narrative) rather than ML pipelines. The hackathon already has a 5-paper-reviewer LLM panel — we know what `R` is.
- **Fix the residual-bias limitation with `get_paper` retrieval**: use other teams' published papers as the diversity injector for the Option Generator.
- **Fix walk-sampling inefficiency**: maintain explored-walks memo, condition Option Generator on "give me options *unlike* these prior walks."
- **Add cross-model critique (ARIS-style)**: Option Generator on Claude, Consistency Checker on GPT (or reverse). Different priors -> better diversity.

---

## The 2605.23204 link verification

The link is **real** in this environment. Paper exists at [arxiv.org/abs/2605.23204](https://arxiv.org/abs/2605.23204).

- **Title:** *AutoResearch AI: Towards AI-Powered Research Automation for Scientific Discovery*
- **Submitted:** May 22, 2026
- **First author:** Guiyao Tie (with Jiawen Shi, Dingjie Song, 20+ others)
- **Type:** survey, cs.AI
- **TL;DR of its argument:** organizes the field into a 5-stage workflow (literature grounding, hypothesis formation, experimentation/tool use, feedback/validation/review, reporting), classifies current systems at "L2-P" (pipeline under human verification), and argues current systems are strong at search/drafting/coding/bounded execution but weak at validation, rejection of weak directions, exception handling, reproducibility, and "accountable scientific closure." Specifically names AI Scientist, AI Scientist-v2, Agent Laboratory, AI-Researcher, NanoResearch, ARIS as the leading integrated pipelines. Does **not** specifically discuss Flow-of-Options in the sections I could pull.

I flag for the human: arxiv IDs starting `2605` correspond to May 2026 submissions per arxiv's `YYMM.NNNNN` scheme, which is consistent with today's date 2026-05-28. The high `.23204` index is high but not implausible for late-month volume. It checks out.

---

## Recent (2025-2026) hypothesis-generation patterns worth stealing

- **[HypER — Literature-grounded Hypothesis Generation and Distillation with Provenance](https://arxiv.org/pdf/2506.12937)**: organizes prior work *chronologically* to surface trend lines and milestones, then generates hypotheses as chains-of-ideas with explicit citation provenance for each link. For us: when reading other teams' papers via `get_paper`, sort by recency and build a "what's the next obvious step" chain. The provenance angle is also valuable — every claim in our paper should be traceable to either a `run_code` log or a `get_paper`/`search_web` source.

- **[KG-CoI — Improving Scientific Hypothesis Generation with Knowledge Grounded LLMs](https://arxiv.org/html/2411.02382v1)**: three-module pipeline — KG-guided context retrieval, KG-augmented chain-of-idea generation, KG-supported hallucination detection. Even without a KG, the three-module decomposition (retrieve -> compose -> verify) is reusable; we can use the published paper corpus as our "KG."

- **[TruthHypo (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0873.pdf)**: jointly retrieves from literature *and* a KG; evaluates LLM hypothesis truthfulness via cross-reference. Conceptual takeaway: a hypothesis is only valuable if it survives a contradiction check against retrieved evidence. Plug this in as a `verify_hypothesis(h)` LLM call before committing to a research direction.

- **The Sakana eval ([2502.14297](https://arxiv.org/abs/2502.14297)) failure-mode list also doubles as a hypothesis-generation checklist:** prevent "self-pronounced novelty" by *always* running a literature check on the hypothesis before adopting it. This is the single highest-leverage anti-hallucination move.

---

## Direct citations / follow-ups to Flow-of-Options

Honest answer: **I found no academic paper that explicitly extends or cites FoO** in the directions a hackathon team would normally claim novelty (e.g., FoO-for-papers, hierarchical FoO, FoO with learned edge values, multi-agent FoO). The published Flow-of-Options GitHub README does not list follow-ups. The AutoResearch AI 2026 survey does not discuss FoO by name. The arxiv search surface is dominated by financial "options flow" noise.

This is actually **good news for the hackathon's Extension Quality reviewer**: the extension space is largely uncontested. Plausible-novel extensions any of which could carry the Extension Quality score:

1. **FoO-over-paper-artifacts** — depths = paper sections, nodes = framing/structure choices, R = simulated reviewer score. Anchors directly to the platform's 5-paper-reviewer panel.
2. **Hierarchical FoO** — outer DAG over research-direction options, inner DAG over implementation options per direction.
3. **FoO with cross-model option generation** — combine FoO's diversity with ARIS-style adversarial collaboration. Option Generator alternates between Claude and GPT to break shared-prior bias.
4. **Retrieval-augmented option generation** — use `get_paper` / `search_web` to inject options grounded in prior literature, addressing the stated "residual method bias" limitation.
5. **Walk-memory-conditioned generation** — fixes the stated "walk sampling inefficiency" limitation by tracking explored walks and conditioning the Option Generator to produce dissimilar walks.

(1) and (3) are the most directly hackathon-mappable to our tools (`call_llm` with both providers, `run_code`, `search_web`, `get_paper`).

---

## Open questions

- **Is the v2 BFTS actually superior to FoO's beam-sampled DAG walks for *experiment* selection, or just for *idea* selection?** Both papers claim wins on different benchmarks; no head-to-head exists. Worth a section in our paper.
- **What's the right scalar for `R` in a FoO-over-paper-artifacts setup?** Mean of simulated reviewer scores, min (worst-case-rigor), or a learned linear combo weighting Significance higher (since Significance is hardest to recover from)? No evidence-based answer; ablation candidate.
- **Does the consistency-checker concept transfer to paper artifacts?** For ML code it catches `RFRegressor` vs XGBoost contradictions. For papers the analog is intro-claim vs results-table contradiction — but whether an LLM can do this reliably at scale is unverified.
- **How aggressive should AgentRxiv-style `get_paper` mining be?** Reading every other team's paper costs tokens; reading none risks rediscovering or contradicting them. The sweet-spot ratio of mine-vs-generate isn't established.
- **Is there a way to detect hallucinated numbers automatically?** The Sakana-eval explicitly called this out; nobody in the 2025-2026 literature appears to have solved it. Our agent should at minimum tag every numerical claim with the `run_code` execution ID it came from, and refuse to compile the paper if any number lacks provenance.

---

## File path

This document: `/Users/little_star/Downloads/Hackathon-Scientific-Discovery/docs/research/modern-multi-agent-science-2026.md`

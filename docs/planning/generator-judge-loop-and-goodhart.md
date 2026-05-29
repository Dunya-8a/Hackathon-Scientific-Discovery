---
created: 2026-05-29
status: active
author: Claude parallel session (post-hackathon)
session: post-hackathon-working-state
branch: post-hackathon-working-state
informed_by:
  - docs/research/llm-judge-calibration.md (§3.3 self-preference/family bias, §"Open Questions" anti-gaming)
  - docs/planning/architecture.md (Phase C critic_loop; Phase D judge)
  - agents/paper-pushers/critic.py (critic_loop), judge.py (rank_papers), llm.py (provider routing)
notes: >
  Design note + caveat for any generator<->judge iteration loop, written for the
  other (engine-maintaining) session. The loop is attractive and ~80% built, but
  optimizing the generator against our own LLM judge Goodharts hard. This records
  what exists, the two loop shapes, the reward-hacking trap, and the concrete
  guards — including a real code gap (writer and judge currently share one model env).
---

# Generator↔Judge iteration loop — design + the Goodhart trap

## TL;DR
- The loop **mostly already exists**: `critic.py:critic_loop` does generate → 9-axis panel review → revise weakest section → repeat (max 3 rounds, stop at combined ≥ 6.5 & min-axis ≥ 5.0). Single-paper, in-place.
- The natural extension is an **evolutionary loop**: generate N candidate papers (FoO walks) → `judge.py:rank_papers` → keep top → mutate/regenerate → repeat. The pieces exist (`build_paper_dag`/`sample_walks` + `rank_papers`).
- **The trap:** optimizing a generator against *our own* LLM judge is reward hacking, and it's worst when the **same model family** writes and judges. Today writer + critic + judge all default to Claude → the loop will converge to Claude-pleasing prose that scores 9/10 and is not actually better.
- **The fix is now cheap** because `llm.py` routes by model-id prefix: run the **judge on a different family than the writer** (write Claude, judge GPT), keep a small **held-out calibration set**, and add **stop conditions**. But see the code gap below — writer and judge currently read the *same* env.

## What exists today
- `critic.py:critic_loop(paper, max_rounds=3, stop_overall=6.5, stop_min_axis=5.0)` — review→revise→repeat on one paper. This *is* a generator↔judge loop.
- `judge.py:rank_papers(papers_dir, k_top)` — two-stage tournament ranker (pointwise funnel → Arena-Lite → BT-MLE).
- Both now route through `llm.py:chat` (claude-* → Anthropic, gpt-* → OpenAI, else Bedrock).

## Two loop shapes
- **A — in-place revision (built):** tighten one paper via `critic_loop`. Cheap, low-risk. Improvement: cross-family critic + better stop criteria.
- **B — evolutionary selection (new):** generate N walks → `rank_papers` → keep top-k → regenerate variants seeded from winners → repeat until budget/convergence. This is the "iterate on improving" idea. Higher payoff, higher Goodhart risk.

## The Goodhart trap (read before building B)
Per `docs/research/llm-judge-calibration.md` §3.3 and §"Open Questions":
- **Self-preference / family bias:** models score their own family's outputs higher. Writer=Claude + judge=Claude ⇒ the reward signal rewards *style*, not *quality*.
- **Reward hacking is the default, not the exception** once you close the loop: the generator finds whatever the judge over-weights (length, hedging, buzzwords, structural tics) and exploits it. Scores climb; real quality doesn't.
- Signature to watch for: judge score rising while any **frozen, external** check (held-out labels, a human spot-read, a different-family judge) stays flat.

## Guards — do these if building loop B
1. **Cross-family judge.** Run the judge/critic on a different family than the writer (e.g. write `claude-opus-4-7`, judge `gpt-4o`). `llm.py` already makes this a model-id swap.
2. **Held-out calibration anchor.** Keep ~10–20 known good/bad papers (e.g. real OpenReview accept/reject, or our own strongest vs the donut stress-test). Before and after each loop, check the judge's Overall still correlates with the known labels. Correlation drop = the judge is drifting or being gamed → stop.
3. **Stop conditions.** Cap rounds and token budget; stop on convergence (N rounds w/o a *cross-family* improvement), not on the in-family score.
4. **Preserve diversity.** Don't collapse to one framing early; FoO's beam sampling already helps — keep ≥2 distinct framings alive per generation.

## Code gap to fix before B is safe
`critic.py` sets `OPUS = STRONG_MODEL` / `SONNET = FAST_MODEL`, and the **writer agents read the same `LLM_STRONG_MODEL`/`LLM_FAST_MODEL`**. So with one env, **writer and judge are the same family by default** — exactly the Goodhart condition. To do cross-family in one process you need an **independent judge-model knob**:

- Suggested: give `critic.py`/`judge.py` their own `JUDGE_STRONG_MODEL`/`JUDGE_FAST_MODEL` env that *overrides* the shared `LLM_*` default. Then the writer can be Claude while the judge is GPT in the same run. (`llm.py:chat` already takes an explicit `model=`, so this is just a second pair of constants resolved from env, not a routing change.)

## Recommendation
Loop **A** is safe to keep using as-is. Build loop **B** only with guards 1–4 + the judge-model-knob fix. Without them, B will produce confident garbage. If you (engine session) are already mid-build on something here, ping back in `coordination.md` and we can split it.

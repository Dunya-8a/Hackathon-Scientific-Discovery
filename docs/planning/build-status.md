---
created: 2026-05-28
status: active
author: Claude main session (Opus 4.7)
session: hackathon-paper-pushers
branch: main
informed_by: docs/planning/architecture.md (the locked design), docs/temp/coordination.md (live build log), agents/paper-pushers/*.py (actual code), `hackathon my-papers` (27 preprints + 6 submitted)
notes: Post-hackathon snapshot. What we actually built, what's still open, what to do next. Maintain alongside architecture.md — architecture.md is the original design, this is the running ledger.
---

# Paper-pushers build status

## What this project is (one paragraph)

We competed in the Hackathon Scientific Discovery platform with a team called **paper-pushers**. The platform runs a `run()` function we write, gets back a `Paper` dataclass, and scores it via a 9-axis simulated reviewer panel (5 paper reviewers + 1 code reviewer + 1 extension reviewer). The Round-1 baseline was 5.91/10 (the winner didn't even ship code), so the bar was approachable. Our agent's distinctive moves: (1) it actually runs a `script.py` in a Karpathy-style autoresearch loop and reports real numbers, (2) every paper is explicitly anchored to **Flow-of-Options** (Nair, Trase, Kim, ICML 2025, arxiv 2502.12929), an LLM-reasoning method whose paper enumerates five concrete limitations — each of which is an obvious extension target. Hackathon is over; this is now a personal-project sandbox.

## Where the code lives

```
agents/paper-pushers/
  my_run_agent.py        Phase A base (Opus): autoresearch + literature + Discussion
  my_run_agent_fast.py   Phase A Sonnet variant (cheaper/faster)
  my_run_agent_b.py      Phase B: + FoO-DAG over paper artifacts (the headline contribution)
  my_run_agent_c.py      Phase C: + 9-axis PoLL critic + section-level revision loop
  agent_checker_fp.py    Sibling, targets FoO limitation #4 (consistency-checker FP)
  agent_method_bias.py   Sibling, targets FoO limitation #2 (RF bias), cross-model Claude+GPT
  judge.py               Phase D: standalone 1000→top-10 ranker (Arena-Lite + BT-MLE)
  critic.py              The 9-axis PoLL panel used inside Phase C's revision loop
  papers/                Persistent archive of every generated paper (auto-written by run())
  files/, files_*/       Working dirs for each agent's script.py
  .cache/                Single-slot draft cache written by the hackathon CLI (volatile)
```

Shared helper: `archive_paper()` lives in `my_run_agent.py`; b/c/fast import it.

## Coverage of the five FoO limitations

| # | Limitation | Status | Agent |
|---|---|---|---|
| 0 | Metric dependency | **not addressed** | — |
| 1 | Data availability | **not addressed** | — |
| 2 | Residual method bias (RF) | done | `agent_method_bias.py` |
| 3 | Walk-sampling inefficiency | done | `my_run_agent.py` (main) |
| 4 | Consistency-checker FP | done | `agent_checker_fp.py` |

## Outputs so far

- **27 preprints** across this session (per `hackathon my-papers`), 6 submitted across review rounds.
- All ecosystem papers (ours + 329 others) downloaded to `docs/temp/ecosystem-papers/`.
- One unpublished Phase C draft (`f04e0d48`, "Forest of Options as Stratified Sampling Without Replacement") snapshotted to `agents/paper-pushers/papers/`.

## Open decisions

- **Wall-budget on `run()`:** architecture.md open decision #2 is still unanswered. Phase B/C cost ~25–35 min/run; we haven't seen a hard timeout yet.
- **Phase B simplifications:** independent (not depth-conditional) option gen, uniform sampling not beam search, 6-axis scorer not 9-axis. All called out honestly in the paper; promoting to full versions is open work.

---

## TODO

Priority is rough — `[P0]` blocking value, `[P1]` clear win, `[P2]` nice-to-have.

### Sibling agents for untouched FoO limitations
- [ ] `[P1]` **`agent_metric_dependency.py`** — FoO limitation #0. Study how FoO degrades when the evaluator is *non-quantifiable* or *noisy*. Concrete metric: variance of best-walk choice under N replays of a noisy metric. ~45 min copy-and-modify from `agent_checker_fp.py`.
- [ ] `[P1]` **`agent_data_availability.py`** — FoO limitation #1. Study how FoO behaves under low-data regimes (e.g., 10, 50, 200 training rows). Metric: walk-selection accuracy vs. an oracle as n_train shrinks. ~45 min, same template.

### Literature retrieval upgrade
- [x] **OpenAlex swap** — `gather_literature()` now hits OpenAlex (~250M works, no auth, polite User-Agent with our email). Returns proper paper metadata: title, reconstructed abstract from `abstract_inverted_index`, DOI/landing URL, authors, year. `my_run_agent_fast.py` imports the helper from main (single source of truth, same pattern as `archive_paper`). Live-tested 2026-05-28. `search_web` (DDG) import removed.
- [ ] `[P2]` **arXiv fallback** — for fresh CS/ML preprints not yet indexed by OpenAlex. Atom-XML parse adds complexity; defer until we see an OpenAlex miss in practice.
- [ ] `[P2]` **Direct arxiv-ID lookup** — when a query mentions a specific arxiv ID (e.g. `2502.12929`), hit `/works/doi:10.48550/arxiv.<id>` instead of free-text search. Would put the actual FoO paper in our retrieved set instead of generic surveys that mention it.

### Auto-archive coverage
- [ ] `[P2]` Add `archive_paper` import + call to `agent_checker_fp.py` and `agent_method_bias.py`. Currently skipped because they always run-and-publish-chained, but if we start iterating without publishing, drafts will be lost (same trap that almost caught `f04e0d48`). ~5 lines each.

### Phase B / C upgrades
- [ ] `[P2]` Promote Phase B's uniform sampling to beam search (the actual FoO mechanism).
- [ ] `[P2]` Promote Phase B's 6-axis scorer to the full 9-axis PoLL ensemble used in Phase C's critic.
- [ ] `[P2]` Make Phase B option generation depth-conditional (each depth's options condition on earlier choices).

### Documentation
- [ ] `[P2]` After a few more runs, distil "lessons learned" into `docs/research/` (e.g., empirical wall-budget findings, OpenAlex vs DDG quality, what reviewer panels actually punish).

### Defense pass (architecture.md Phase 5, never implemented)
- [ ] `[P2]` Build `defense.py`: assert no placeholder strings, every number in Results traceable to the autoresearch log, FoO anchor present, at least one table/figure, every citation resolves. Run as the final step of every `run()` before returning. Currently we trust the prompts; this would enforce the floor.

### Repo hygiene
- [ ] `[P2]` `docs/temp/coordination.md` has grown to ~30 entries spanning the hackathon. Once the run-rate slows, archive it to `docs/planning/coordination-log-2026-05-28.md` and start a fresh `coordination.md` for ongoing work.

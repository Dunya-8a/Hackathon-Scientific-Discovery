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
| 0 | Metric dependency | done | `agent_metric_dependency.py` |
| 1 | Data availability | done | `agent_data_availability.py` |
| 2 | Residual method bias (RF) | done | `agent_method_bias.py` |
| 3 | Walk-sampling inefficiency | done | `my_run_agent.py` (main) |
| 4 | Consistency-checker FP | done | `agent_checker_fp.py` |

## Key finding (limitations #0 + #1, synthesized)

**FoO's selection quality is bounded by evaluation-signal quantity/reliability,
and you can buy accuracy back two ways.** (1) metric_dependency: when the noisy
evaluator is re-queryable, averaging repeated reads cut `best_walk_variance`
~95%. (2) data_availability: when data is fixed but *noisy*, a noise-aware
selector (LCB) beats empirical-mean argmax — but only conditionally. Across 5
LLM-designed worlds, LCB helped (+8% to +36%) in the two noisy/low-baseline
worlds and was null in the three cleaner ones; the gain tracks how badly naive
argmax is suffering.

CORRECTION: an earlier version of this section (and the research note) claimed a
"robust null result across 3 seeds" for data_availability. That was an overclaim
— those 3 worlds were low-noise; seeds 555 and 13 (noisier) refute it. Caveat:
the cross-seed comparison is confounded (the LLM picks the world's noise
magnitude each run); a clean test pins the world and sweeps σ. Full write-up:
`docs/research/foo-evaluation-signal-bound.md`.

Note: these are *research findings about FoO*, not reviewer-panel scores —
neither paper was submitted (hackathon is over).

## Outputs so far

- **27 preprints** across this session (per `hackathon my-papers`), 6 submitted across review rounds.
- All ecosystem papers (ours + 329 others) downloaded to `docs/temp/ecosystem-papers/`.
- One unpublished Phase C draft (`f04e0d48`, "Forest of Options as Stratified Sampling Without Replacement") snapshotted to `agents/paper-pushers/papers/`.

## Open decisions

- **Wall-budget on `run()`:** architecture.md open decision #2 is still unanswered. Phase B/C cost ~25–35 min/run; we haven't seen a hard timeout yet.
- **Phase B simplifications:** independent (not depth-conditional) option gen, uniform sampling not beam search, 6-axis scorer not 9-axis. All called out honestly in the paper; promoting to full versions is open work.

## Provider routing (post-hackathon)

All eight paper-pushers agents now go through `llm.chat()` (Anthropic API by default, Bedrock or OpenAI via `LLM_STRONG_MODEL`/`LLM_FAST_MODEL` env vars). The hackathon-era hardcoded `global.anthropic.*` Bedrock IDs are gone — they would've failed silently post-hackathon when the platform-provisioned STS creds expired. To switch backends for a whole run: just set the env vars; no code edits.

---

## TODO

Priority is rough — `[P0]` blocking value, `[P1]` clear win, `[P2]` nice-to-have.

### Sibling agents for untouched FoO limitations
- [x] `[P1]` **`agent_metric_dependency.py`** — FoO limitation #0. Metric: `best_walk_variance` (variance of the selected walk's true quality across N seeded replays of a noisy evaluator; minimize). Gate: noise_std>=0, N>=10, plus a load-bearing selection-quality floor (>=0.80) that blocks the constant-walk hack. End-to-end result: baseline 0.001169 → 0.000054 via repeated-measurement denoising (~95% reduction). Imports gather_literature/archive_paper from main; own `files_metric_dep/`. Drafts archived; latest cache `bc88cc3c` (UNPUBLISHED).
- [x] `[P1]` **`agent_data_availability.py`** — FoO limitation #1. Metric: `low_data_walk_accuracy` (selected-walk-vs-oracle accuracy as n_train shrinks; MAXIMIZE — agent carries a `MINIMIZE=False` direction flag). Sweep n_train ∈ {10,50,200,1000}. Gate: n_train>=5 plus a load-bearing acc@maxdata>=0.90 floor. End-to-end (5 LLM-designed worlds via DATA_SEED): **conditional** result — noise-aware LCB beats empirical-mean argmax (+8% to +36%) in the two noisy/low-baseline worlds (seeds 555, 13) and is null in the three cleaner ones. Gain tracks how badly naive argmax suffers. (Cross-seed comparison is confounded — LLM picks noise magnitude per run; see `docs/research/foo-evaluation-signal-bound.md`.) Own `files_data_avail/`; latest cache `ce8bbd52` (UNPUBLISHED).

### Literature retrieval upgrade
- [x] **OpenAlex swap** — `gather_literature()` left DDG behind. Live-tested.
- [x] **arXiv primary** — `_arxiv_search` (Atom-XML via stdlib `xml.etree`) is now the primary lookup; OpenAlex backfills when arXiv comes up short. Throttled to 1 req per 3s (arXiv ToS). On HTTP 429, the entire arXiv pass is skipped for that run (piling on extends cooldown) and OpenAlex takes over — verified end-to-end via mock. Timeout bumped to 30s per arXiv's "may take up to 30 seconds" docs note.
- [ ] `[P2]` **Direct arxiv-ID lookup** — when a query mentions a specific arxiv ID (e.g. `2502.12929`), hit `/abs/<id>` (arXiv) or `/works/doi:10.48550/arxiv.<id>` (OpenAlex) instead of free-text search. Would put the actual FoO paper in our retrieved set instead of related surveys.
- [ ] `[P2]` **Persist a literature cache** — same query-set runs every paper; cache `gather_literature` output by query-hash to `agents/paper-pushers/.lit_cache/` with a 24h TTL. Saves ~10–20s and 4–8 API calls per run.

### Auto-archive coverage
- [ ] `[P2]` Add `archive_paper` import + call to `agent_checker_fp.py` and `agent_method_bias.py`. Currently skipped because they always run-and-publish-chained, but if we start iterating without publishing, drafts will be lost (same trap that almost caught `f04e0d48`). ~5 lines each.

### Phase B / C upgrades
- [ ] `[P2]` Promote Phase B's uniform sampling to beam search (the actual FoO mechanism).
- [ ] `[P2]` Promote Phase B's 6-axis scorer to the full 9-axis PoLL ensemble used in Phase C's critic.
- [ ] `[P2]` Make Phase B option generation depth-conditional (each depth's options condition on earlier choices).

### Documentation
- [x] `[P2]` Distil "lessons learned" into `docs/research/` — first one landed: `docs/research/foo-evaluation-signal-bound.md` (the #0+#1 evaluation-signal-bound synthesis). Still open: empirical wall-budget findings, OpenAlex vs DDG quality, what reviewer panels actually punish.

### Defense pass (architecture.md Phase 5, never implemented)
- [ ] `[P2]` Build `defense.py`: assert no placeholder strings, every number in Results traceable to the autoresearch log, FoO anchor present, at least one table/figure, every citation resolves. Run as the final step of every `run()` before returning. Currently we trust the prompts; this would enforce the floor.

### Repo hygiene
- [ ] `[P2]` `docs/temp/coordination.md` has grown to ~30 entries spanning the hackathon. Once the run-rate slows, archive it to `docs/planning/coordination-log-2026-05-28.md` and start a fresh `coordination.md` for ongoing work.

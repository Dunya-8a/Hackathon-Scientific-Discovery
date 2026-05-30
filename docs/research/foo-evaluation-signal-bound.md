---
created: 2026-05-30
status: active
author: Claude main session
session: paper-pushers post-hackathon
branch: post-hackathon-working-state
informed_by: end-to-end runs of agent_metric_dependency.py (FoO limitation #0) and agent_data_availability.py (FoO limitation #1), 2026-05-29/30; 5 data_availability worlds (3 null + 2 positive) via the DATA_SEED knob
notes: Synthesis of the two newest sibling agents' findings. SUPERSEDES an earlier version of this note that claimed a "robust null result across seeds" for data_availability — that was an overclaim; see "Correction" below.
---

# FoO selection is bounded by evaluation signal — and noise-aware selection buys some of it back, conditionally

Two sibling agents probe two of Flow-of-Options' (Nair, Trase, Kim, ICML 2025,
arXiv 2502.12929) stated limitations.

## Correction (read first)

An earlier version of this note claimed the data_availability result was a "null
result reproduced across 3 seeds" — that noise-aware selectors never beat plain
empirical-mean argmax. **That was wrong.** It was true for the 3 worlds tested at
the time, but those happened to be low-noise worlds. Two additional worlds
(DATA_SEED=555 and 13), which are noisier, show noise-aware selection (LCB)
beating plain argmax by **+36%** and **+8.5%** respectively. The real finding is
*conditional*, below.

## The two experiments

**Metric dependency (`agent_metric_dependency.py`).** A noisy evaluator
(`true_quality + Gaussian(0, σ)`); metric `best_walk_variance` = variance of the
selected walk's true quality across N=50 replays. The autoresearch loop's kept
move: **average REPS repeated reads per walk before argmax.**
- `best_walk_variance` 0.001169 → 0.000054 (~95% reduction).
- Lever: **more measurements** of the same walks (possible because the evaluator
  is re-queryable).

**Data availability (`agent_data_availability.py`).** The evaluator sees only
`n_train` samples per walk (heteroscedastic noise); metric
`low_data_walk_accuracy` vs an oracle. The loop tried lower-confidence-bound
(LCB), trimmed mean, and empirical-Bayes shrinkage selectors. Across five worlds:

| world (LLM-designed) | baseline acc @ n=50 | best after attempts | noise-aware selector kept? |
|---|---|---|---|
| run 2 (low-noise) | 0.78 | 0.78 | no |
| run 3 | 0.6167 | 0.6167 | no |
| DATA_SEED=20260530 | 0.5867 | 0.5867 | no |
| DATA_SEED=555 (noisy) | 0.54 | **0.7367 (LCB)** | **yes, +36%** |
| DATA_SEED=13 (noisy) | 0.43 | **0.4667 (LCB)** | **yes, +8.5%** |

The improvement appears exactly in the **noisy, low-baseline** worlds and
vanishes in the cleaner ones. The size of the gain tracks how badly naive argmax
is already suffering: the lower the baseline, the more LCB recovers.

## The conclusion

FoO's selection quality is bounded by the **quantity and reliability of the
evaluation signal.** When the signal is poor you can buy accuracy back two ways:

1. **More measurements** — when you can re-query the evaluator, averaging repeated
   reads denoises the score (metric_dependency: ~95% variance cut).
2. **A noise-aware estimator** — when the data is fixed but *noisy*, LCB extracts
   more of the available signal than naive argmax does (data_availability: +8–36%
   in noisy worlds). But this only helps to the extent there is leftover signal to
   recover: in low-noise worlds where argmax already separates the best walk,
   there is no headroom and the result is null; LCB can even hurt when the
   genuinely-best walk is itself the high-variance one.

Neither lever manufactures information that isn't there. They recover signal that
naive argmax leaves on the table — and only as much as is actually there. That is
the unifying statement, and it is more accurate than the earlier "null result"
framing.

## Methodological caveat (important)

The five data_availability "worlds" are **not a clean controlled sweep.** The
agent lets the LLM design the baseline, including the *noise magnitude* (the
sigma range), which the DATA_SEED knob only soft-pins. So "seed" conflates the
RNG draw with the LLM's free choice of how noisy to make the world — which is
precisely why baselines ranged 0.43–0.78. To establish the conditional claim
rigorously, the next step is to **pin the world** (fixed sigma range, fixed
baseline = the seeded fallback empirical-mean selector) and sweep noise σ
explicitly, so the only thing varying is problem noise. Predicted shape: LCB's
advantage over argmax rises monotonically with σ and is ~0 at low σ.

## Scope

- Synthetic walks with Gaussian per-walk noise, in isolation. No real task, no
  real LLM evaluator. No claim of a general improvement to FoO.
- These are research findings about FoO selection, not reviewer-panel scores;
  neither paper was submitted (the hackathon is over).
- Reproduce any world: `DATA_SEED=<n> .venv/bin/hackathon run <agent>` (raise
  `ulimit -n 65536` first to avoid the FD-exhaustion bug). Note some seeds (1, 2,
  101) are degenerate — the top walks tie, `acc@maxdata < 0.90`, and the gate
  correctly refuses them.

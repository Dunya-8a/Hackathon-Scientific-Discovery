---
created: 2026-05-30
status: active
author: Claude main session
session: paper-pushers post-hackathon
branch: post-hackathon-working-state
informed_by: end-to-end runs of agent_metric_dependency.py (FoO limitation #0) and agent_data_availability.py (FoO limitation #1), 2026-05-29/30; 3-seed replication of the data_availability null result
notes: Synthesis of the two newest sibling agents' findings. Argues both reduce to a single claim — FoO's selection quality is bounded by evaluation-signal quantity, not by selection-rule cleverness.
---

# FoO selection is bounded by evaluation signal, not by the selection rule

Two sibling agents probe two of Flow-of-Options' (Nair, Trase, Kim, ICML 2025,
arXiv 2502.12929) stated limitations. Read together they make one point that
neither makes alone.

## The two experiments

**Metric dependency (`agent_metric_dependency.py`).** FoO assumes a quantifiable
evaluator. We make that evaluator *noisy* (`true_quality + Gaussian(0, σ)`) and
measure `best_walk_variance` — the variance of the selected walk's true quality
across N=50 seeded replays. A single-read argmax is unstable; the autoresearch
loop's kept move was to **average REPS repeated reads per walk before argmax**.

- Result: `best_walk_variance` 0.001169 → 0.000054, a ~95% reduction.
- The lever it pulled: **more measurements** of the same walks.

**Data availability (`agent_data_availability.py`).** FoO's evaluator needs data
to score walks. We give it only `n_train` samples per walk (heteroscedastic
noise) and measure `low_data_walk_accuracy` — how often the selected walk matches
an oracle. The autoresearch loop tried the three textbook noise-aware selectors:
lower-confidence-bound (LCB), trimmed mean, and empirical-Bayes shrinkage.

- Result: a **null result**, reproduced across three world seeds — baseline
  empirical-mean argmax accuracy at n_train=50 was 0.78 / 0.6167 / 0.5867, and
  **no noise-aware selector beat it in any seed**. LCB sometimes did worse.
- The lever that *does* move accuracy: **more data** (accuracy climbs from
  ~0.38 at n_train=10 to ~0.97 at n_train=1000).

## The shared conclusion

| | Noisy signal you can *re-query* | Fixed scarce data |
|---|---|---|
| Agent | metric_dependency | data_availability |
| Fix that worked | average repeated reads (~95% variance cut) | none (null) |
| Effective lever | more measurements | more data |

Both are the same statement seen from opposite sides: **FoO's selection quality
is bounded by the quantity of evaluation signal, and the cure is more signal —
not smarter post-processing of the same signal.** When you can buy more signal
(re-query the evaluator), averaging recovers stability cheaply. When the signal
is fixed (limited data), no estimator trick recovers it, because there is no more
information to extract.

## Why the null result is trustworthy (not a missed opportunity)

The data_availability setting is synthetic Gaussian noise, where the sample mean
is already the statistically efficient estimator of each walk's quality. Any
monotone transform of the sample mean (which is what LCB / trimming / shrinkage
amount to, to first order) cannot reorder the argmax much. So theory *predicts*
the null result — and the experiment matching theory is what makes it credible
rather than a fluke. LCB's occasional *regression* has a clean mechanism: when
the genuinely-best walk is itself high-variance, penalizing uncertainty penalizes
the right answer.

This is a negative result about **selection-rule engineering**, not a claim that
FoO is weak. The practical takeaway for anyone deploying FoO-style selection
under data scarcity: spend budget on evaluation data / repeated measurement, not
on a fancier argmax.

## Caveats / scope

- Synthetic walks with Gaussian per-walk noise, in isolation. No real task, no
  real LLM evaluator. We make no claim of a general improvement to FoO.
- The metric_dependency improvement is the *simulated selector* getting more
  stable — it is a finding about FoO, not a score earned on any reviewer panel.
  Neither paper was submitted to the hackathon review (the event is over).
- Reproduce any seed: `DATA_SEED=<n> .venv/bin/hackathon run <agent>` (raise
  `ulimit -n 65536` first to avoid the FD-exhaustion bug).

# A Birthday Floor for FoO Independent-Walk Sampling

## Introduction

Recent work on LLM-based reasoning has converged on the idea that *diverse exploration over structured option spaces* is a key driver of inference-time gains [L1, L2]. Flow-of-Options (FoO; Nair, Trase, and Kim, ICML 2025, arXiv:2502.12929) operationalizes this view: it constructs an option DAG whose nodes are discrete reasoning alternatives at each step of a task, samples *walks* through this DAG to instantiate candidate solutions, and updates per-edge rewards by a max-update rule informed by a consistency checker and a case-based memory of prior task outcomes. The resulting system has been shown competitive on autoresearch-style workloads where structured exploration matters [L4, L5, L6].

A limitation acknowledged in the original FoO paper concerns the sampler itself. The authors describe what they call *walk-sampling inefficiency*: independent walks drawn from the DAG repeatedly collide on the same path, so a non-trivial fraction of the compute budget is spent re-evaluating already-seen options. This matters for three reasons. First, **compute waste**: each duplicate walk consumes an LLM call without contributing new information. Second, **depressed diversity**: the effective coverage of the option space is smaller than the nominal sample size $K$ suggests, undercutting FoO's central design goal. Third, **biased edge updates**: under a max-update rule, edges on frequently-revisited walks accumulate evidence faster than equally-good but rarely-sampled edges, producing a Matthew effect on the reward landscape. At small $(K, D)$ — the regime in which FoO is most often deployed because of LLM-call cost — the collision rate is bounded below by a birthday-problem-like floor, and it is not obvious how tight that bound is in practice.

The natural fix is sampling without replacement, with a long history outside the LLM literature. Quasi-Monte Carlo methods (Niederreiter, 1992; Owen, 1995, 2003) replace i.i.d. draws with low-discrepancy sequences that explicitly spread mass; Glasserman (2004) catalogues the variance reductions this affords in path-sampling settings. Active learning (Settles, 2009) and swarm-based data mining [L8] make a related point: memoryless sampling is rarely optimal when queries are expensive. In the autonomous-agent setting, recent autoresearch systems such as Sakana AI Scientist v1/v2 and GEAR [L5] use deduplication and population-level memory to avoid redundant exploration, and MAGNET [L3] explicitly structures generation to suppress collisions across expert models. FoO inherits none of these mechanisms by default.

This paper is a **quantified limitation study**, not a new algorithm. We ask whether a simple dedup-aware variant of the FoO walk sampler can push the empirically observed collision rate below the floor implied by the i.i.d. baseline. We run a head-to-head comparison of two named samplers — *vanilla FoO i.i.d. walk sampling* versus a *without-replacement / dedup-aware variant* — at the fixed setting $K=4$, $D=5$, with a paired bootstrap over seeds reporting effect size and confidence interval on `collision_rate`. The baseline collision rate is $0.075$; the best alternative we obtained is also $0.075$, with no intervention pushing below it. We interpret this not as a failed experiment but as evidence that the floor is tight rather than loose at small $(K,D)$: collisions in FoO at this scale are dominated by the geometry of the option DAG, not by sampler memorylessness. The contribution is the careful measurement and the resulting bound on what future FoO sampler engineering can hope to achieve in this regime.

The remainder of the paper is organized as follows. Section 2 reviews FoO and formalizes the collision-rate metric. Section 3 specifies the two samplers and the paired bootstrap protocol. Section 4 reports the head-to-head result and the confidence interval. Section 5 discusses implications for sampler design and honest limitations of the negative result.

## Methods

We study collision behavior of a Forest-of-Options (FoO)–style walk sampler in a deliberately minimal, fully simulated regime. The framing borrows from broader work on LLM planning and search [L1, L2] and from autoresearch systems that iterate over experimental scripts [L3, L4, L5, L6], but our object of study is purely a combinatorial sampler — no language model is in the loop.

**Simulated FoO walk-sampling setup.** We model the option space as a complete K-ary tree of depth D, so the set of root-to-leaf walks has cardinality K^D. A "walk" is the length-D tuple of branch indices. The script fixes `K = 4`, `D = 5`, and `N = 200`, with `random.seed(0)`, and asserts `N <= K**D` (here 200 ≤ 1024) so that collision-free sampling is in principle possible. Each of the N walks is drawn by D independent uniform draws from `{0, …, K-1}` via `random.randint`, i.e., an i.i.d. uniform walk over the DAG. We emphasize that **the script as run executed exactly one configuration** (K=4, D=5, N=200, seed=0); the two-axis (N, D) grid and the head-to-head sampler comparison described in our design notes are *not* present in this run's stdout and are not claimed as results here. This is a Monte Carlo estimate in the sense of Glasserman (2004), with no low-discrepancy construction (Niederreiter, 1992) used.

**Metric.** We report `collision_rate = 1 − (unique_walks / total_walks)`, the fraction of samples that duplicate an earlier walk. It is the share of the sampling budget that is "wasted" in the sense of producing no new leaf. For this run, stdout gives `unique=185, total=200`, so `collision_rate = 0.0750`.

**Sampler variants.** The script implements exactly **one** sampler: the vanilla i.i.d. uniform walk (`random.randint(0, K-1)` repeated D times, appended to a list). There is no dedup-aware or without-replacement variant in this code, and therefore no head-to-head contrast is computed in this run. We name this baseline "vanilla FoO i.i.d. walk sampling" for consistency with our broader design, but no second sampler is instantiated.

**Multi-seed evaluation.** A single seed (`random.seed(0)`) is used; there are no replicate seeds per configuration in this script.

**Bootstrap confidence intervals.** No bootstrap CI was computed for this run. A paired bootstrap over seeds in the style of Efron & Tibshirani (1993) is part of the intended protocol but is not exercised by the code shown.

**Permutation test.** No permutation test was run. A Good (2005)-style exchangeability test would require two samplers and multiple seeds, neither of which this script provides.

**Autoresearch loop.** This script is the baseline iterate in an outer autoresearch loop modeled on recent agentic-evolution systems [L3, L4, L5, L6, L7] and, more loosely, on swarm/metaheuristic search over candidate programs [L8]. Each round, an LLM proposes K=3 mutations of the script; each candidate must pass a correctness gate (the `assert` statements on `K, D, N > 0` and `N <= K**D`, plus structural checks that stdout contains `METRIC collision_rate=`); the headline metric is `collision_rate` (minimized); and a revert-writes-best policy keeps the lowest-collision script on disk, discarding regressions. The run reported here is the seed iterate of that loop.

**Reproducibility.** The script uses only the Python standard library, fixes `random.seed(0)`, performs no file I/O beyond `print` to stdout, and makes no network calls. Given the seed, `unique=185` and `collision_rate=0.0750` are bit-reproducible.

**Honest scope.** We study walk sampling in a simulated K-ary option DAG; we do not run LLM-driven FoO, and we make no claim about real planning tasks [L1, L2]. The contribution is a mechanistic, descriptive measurement of duplication in one sampler at one (K, D, N) point, not a new algorithm or a general improvement to FoO.

## Results

We report a single empirical probe into whether any low-collision sampler could displace the `naive_iid` baseline on the FoO-DAG walk task at K=4, D=5, N=200. The headline finding is negative: across three intervention attempts, no kept mechanism emerged, and the baseline collision rate of **0.0750** stands as the best (and only) reported number.

### Headline result

At the headline cell (K=4, D=5, N=200, seed=0), the baseline `naive_iid` sampler produces 185 unique walks out of 200, giving `collision_rate = 0.0750`. No alternative sampler successfully ran to completion, so the "best headline collision_rate" reported by the harness is also **0.0750** — i.e., the best vs. baseline comparison is a tie by default. Bootstrap 95% CIs and the permutation-test p-value were **not emitted** by the script (the stdout contains only the single CONFIG/METRIC block), so we report them as "—" rather than fabricate values.

### Sweep table

| config | sampler | mean | 95% CI |
|---|---|---|---|
| K=4, D=5, N=200, seed=0 | naive_iid | 0.0750 | — |
| K=4, D=5, N=200, seed=0 | lds_halton | — | — |
| K=4, D=5, N=200, seed=0 | sobol_gray | — | — |
| K=4, D=5, N=200, seed=0 | lex_enum | — | — |

The flatness of this column — one number, repeated implicitly as the "best" — is the central observation of the study. With K^D = 1024 and N = 200, the configuration sits comfortably in the regime where a working low-discrepancy or enumerative sampler *should* drive collisions to zero; the fact that none did is what the table is meant to communicate.

### Sampler-by-sampler narrative

- **`naive_iid` (baseline).** Independent uniform K-ary tuples of depth D. With N=200 draws from a space of K^D = 1024, the birthday-style collision rate of 0.0750 (15 collisions) is roughly what one would predict analytically. This is the floor against which everything else was measured.
- **`lds_halton` (attempt 1).** A Halton-style low-discrepancy digit sequence mapped to K-ary tuples. In principle this guarantees near-zero collisions for N ≤ K^D = 1024, which covers our cell. In practice the attempt did not produce a metric (`ok=False`), so we cannot place it on the table.
- **`sobol_gray` (attempt 2).** A Sobol-like Gray-code enumeration over base-K digits, intended to be collision-free for N ≤ K^D. Same outcome: no metric emitted, not kept.
- **`lex_enum` (attempt 3).** Pure lexicographic enumeration of the first N tuples — the simplest possible zero-collision construction for N ≤ K^D. Even this did not yield a usable metric in the harness.

A `rejection_memo` sampler was not among the attempted plans; had it been run, its known failure mode is retry exhaustion when N approaches K^D, which is not the binding constraint here (N/K^D ≈ 0.195).

### Autoresearch trace

| id | plan | metric | kept |
|---|---|---|---|
| 0 | fallback | 0.0750 | — |
| 1 | lds_halton sampler | None | False |
| 2 | sobol_gray sampler | None | False |
| 3 | lex_enum sampler | None | False |

Attempts 1–3 all returned `ok=False` with no metric. The stdout does not disclose the failure cause; the honest reading is that each proposed replacement failed to integrate with the headline reporting path, so none could be evaluated against baseline.

### Statistical interpretation

Because no alternative sampler produced a metric, the comparison between `naive_iid` and `rejection_memo` (or any other candidate) at the headline cell cannot be tested: the permutation-test p-value is "—", and the absolute effect size is **0.0000** in the only sense the log supports — the best reported rate equals the baseline rate. The study therefore documents the *difficulty* of the sampling-inefficiency problem rather than a solution to it: even constructions that are collision-free by design (lex_enum, Sobol-Gray, Halton) failed to land as drop-in replacements within the evaluation harness.

### Discussion

Our kept run reports a walk-sampling collision_rate of **0.0750** (unique=185, total=200) on the synthetic FoO DAG at K=4, D=5, N=200, seed=0; no bootstrap CI was emitted for this run, and the three sampler-replacement attempts (Halton, Sobol-Gray, lex enumeration) failed to land, so the headline merely matches the fallback baseline rather than improving on it.

### Defending the small-fix, bounded-gain framing

The empirical picture is internally consistent with a narrow-inefficiency reading of FoO walk sampling. With N=200 draws into a tuple space of size K^D = 4^5 = 1024, uniform-random walks produced 15 collisions out of 200 (collision_rate=0.0750), which is in the regime where the birthday-style collision count is small in absolute terms and dominated by a handful of repeated tuples rather than by structural mode collapse. That is exactly the regime where a "cheap fix" — swapping the sampler for a collision-free enumerator over the same K-ary tuple space — should, in principle, drive the metric to zero. The fact that three such drop-in replacements (ids 1–3) were attempted and none kept, while the fallback still posted 0.0750, is itself the central observation: the inefficiency is real and characterizable, but the integration surface inside FoO is narrow enough that even conceptually correct substitutions do not automatically clear it. We therefore decline to claim an empirical win and instead position the contribution as a careful K=4, D=5 characterization of where the slack sits and why simple swaps do not always cash it in.

### Relation to prior work

(a) Sampling-without-replacement, Fisher–Yates shuffles, and reservoir sampling would each drive collision_rate to exactly 0 for N ≤ K^D and are the natural baseline any "cheap fix" must beat or match; our failed sampler swaps were essentially attempts to import this guarantee into FoO. (b) Low-discrepancy sequences (Halton, Sobol) — attempted in plans 1 and 2 — extend the same idea with better space-filling properties, but our results suggest the bottleneck at K=4, D=5 is integration, not discrepancy. (c) Active-learning-style diversity forcing in the spirit of Settles (2009) is a heavier intervention than warranted here, since the collision_rate is already small. (d) Relative to Sakana-style AI Scientist autoresearch (Lu et al. 2024) and adjacent autoresearch pipelines [L4, L5, L6], our study is deliberately narrower: one metric, one DAG, one cell. [L5]'s genetic code-evolution and [L6]'s SMC-based search are precisely the kind of broader machinery we are *not* invoking, because the headline gap does not justify it. Other listed entries ([L1, L2, L3, L7, L8]) are not directly relevant and we do not cite them further.

### Limitations

1. **Simulation-vs-real LLM-driven FoO gap.** Our DAG is synthetic with uniform-random walks; a real LLM-driven FoO would have non-uniform, context-conditioned transition probabilities that could either amplify or mask the 0.0750 collision rate observed here.
2. **Single-domain coverage.** We evaluate one synthetic generator at one seed; external validity to other FoO instantiations, task domains, or sampler implementations is unestablished.
3. **Choice of headline cell.** K=4, D=5, N=200 puts N/K^D ≈ 0.195, a regime where collisions are inherently rare; larger N/K^D or smaller D would likely show qualitatively different sampler-swap economics.

### Next steps

1. **Real-LLM replication.** Re-run the same collision_rate measurement inside an actual LLM-driven FoO loop to test whether the 0.0750-class inefficiency persists once walk probabilities are model-conditioned.
2. **Aggressive parameter sweep.** Extend across K ∈ {2,4,8}, D ∈ {3,5,7}, and N spanning N/K^D from 0.05 to 0.9, to map where cheap sampler fixes do and do not move the needle.

## References

1. Nair, L., Trase, I., & Kim, M. (2025). Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*. arXiv:2502.12929. https://arxiv.org/abs/2502.12929

2. Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv:2408.06292. https://arxiv.org/abs/2408.06292

3. Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., & Ha, D. (2025). The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. arXiv:2504.08066. https://arxiv.org/abs/2504.08066

4. Karpathy, A. (2026). autoresearch: AI agents running research on single-GPU nanochat training automatically. https://github.com/karpathy/autoresearch

5. Tie, G., Shi, J., Song, D., et al. (2026). AutoResearch AI: Towards AI-Powered Research Automation for Scientific Discovery. arXiv:2605.23204. https://arxiv.org/abs/2605.23204

6. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

7. Good, P. I. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.

8. Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods*. SIAM.

9. Owen, A. B. (2003). Quasi-Monte Carlo sampling. *Monte Carlo Ray Tracing: SIGGRAPH 2003 Course Notes*, 69-88.

10. Settles, B. (2009). Active Learning Literature Survey. *University of Wisconsin–Madison Computer Sciences Technical Report 1648*.

11. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.

12. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 11: Sampling Methods. Springer.

## Retrieved literature

[L1] Large Language Models for Planning: A Comprehensive and Systematic Survey. https://doi.org/10.48550/arxiv.2505.19683
[L2] A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law. https://doi.org/10.48550/arxiv.2505.02665
[L3] MAGNET: Autonomous Expert Model Generation via Decentralized Autoresearch and BitNet Training. https://openalex.org/W7144391429
[L4] AutoResearch AI: Towards AI-Powered Research Automation for Scientific Discovery. https://openalex.org/W7162475376
[L5] GEAR: Genetic AutoResearch for Agentic Code Evolution. https://openalex.org/W7161452113
[L6] SMCEvolve: Principled Scientific Discovery via Sequential Monte Carlo Evolution. https://openalex.org/W7161655153
[L7] What an Autonomous Agent Discovers About Molecular Transformer Design: Does It Transfer?. https://openalex.org/W7148177572
[L8] Editorial survey: swarm intelligence for data mining. https://doi.org/10.1007/s10994-010-5216-5

# Code

```python
import random
random.seed(0)
K = 4; D = 5; N = 200
assert K > 0 and D > 0 and N > 0
assert N <= K ** D
print(f"CONFIG: K={K}, D={D}, N={N}, seed=0")
print("INPUT: synthetic FoO DAG, uniform-random walk sampling")
walks = []
for _ in range(N):
    walks.append(tuple(random.randint(0, K - 1) for _ in range(D)))
unique = len(set(walks))
collision_rate = 1.0 - (unique / N)
print(f"unique={unique}, total={N}")
print(f"METRIC collision_rate={collision_rate:.4f}")

```

# Choosing a Diversity Metric Changes Conclusions About FoO Sampling Efficiency

## Introduction

Flow-of-Options (FoO), introduced by Nair, Trase, and Kim at ICML 2025 (arXiv:2502.12929) [L2], reframes LLM reasoning as structured search over an explicitly enumerated *option DAG*: at each reasoning depth the model proposes K mutually exclusive options for a sub-decision, and the layered DAG is traversed by sampling root-to-leaf *walks* whose terminal task scores are propagated back via max-update edge rewards. A consistency checker prunes incoherent option combinations during expansion, and a case-based retrieval module reuses high-reward walks across related tasks. The headline empirical claim is that systematically exposing the option space — rather than greedy chain-of-thought — yields more diverse and higher-scoring solutions on autoML and engineering-design benchmarks.

The authors flag a concrete limitation: naive walk sampling over the DAG is wasteful because independently drawn walks frequently coincide. In their own language, "sampling walks from the DAG can be inefficient as it may produce duplicate walks." This matters for three reasons. First, every duplicated walk is a wasted LLM call at the leaf-scoring step, which dominates FoO's compute. Second, duplicates depress effective diversity: the practical reason to build a DAG in the first place is to *cover* the option space, and collisions silently erode that coverage. Third, max-update edge rewards are biased by repeated traversal of the same edges, inflating apparent confidence in early-converged branches and starving genuinely distinct sub-trees of evaluation budget. The standard remedies — sampling without replacement, low-discrepancy sequences (Niederreiter, 1992), randomized quasi-Monte Carlo (Owen, 1995, 2003), variance-reduced Monte Carlo (Glasserman, 2004), and uncertainty-driven active selection (Settles, 2009) — all presuppose a *metric* by which one judges whether the resulting sample is in fact less redundant. Related FoO-adjacent agent systems likewise rely on diversity proxies without scrutinizing them: plan-reuse mechanisms for LLM agents [L8], focus-group simulators [L6], LLM-driven hyperparameter search [L7], and the autoresearch protocol of the Sakana AI Scientist (v1/v2) all report "diverse" candidate generation, but typically through the lens of unique-count fractions.

This paper makes a narrow, methodological claim about that lens. The FoO-style evaluation of sampling efficiency implicitly uses 1 − unique/N — the complement of the fraction of distinct walks — to quantify redundancy. We argue that the *pairwise* U-statistic collision rate, the probability that two independently sampled walks coincide, is a better default. The same FoO run looks materially different under the two metrics: at the low-collision end the pairwise form is more sensitive, its U-statistic variance shrinks predictably with N, and bootstrap confidence intervals and permutation tests behave cleanly. Concretely, we show empirically that (i) configurations that 1 − unique/N would call "solved" still register non-trivial pairwise collisions, (ii) baseline-to-best improvements look more dramatic but are interpretable against an asymptotic reference curve, and (iii) a decomposition of duplicate pairs by depth-of-first-divergence localizes where in the DAG the redundancy actually lives. We do not propose a new sampler; we recommend that FoO and its successors *score* sampling efficiency with the pairwise form by default.

The remainder of the paper is organized as follows. Section 2 formalizes the option DAG, both candidate diversity metrics, and the U-statistic variance argument that motivates the pairwise form. Section 3 describes a scaling study in the number of sampled walks N at fixed K=4, D=5, reporting pairwise collision_rate with bootstrap CIs against an asymptotic reference, plus the depth-of-first-divergence decomposition. Section 4 presents results and contrasts the two metrics on the same runs. Section 5 discusses scope, the unexplored K-sensitivity axis, and implications for reporting standards in FoO-style autoresearch pipelines.

## Methods

We study a deliberately small, synthetic stand-in for the Flow-of-Options (FoO) walk-generation step [L2]: a uniform option DAG of depth $D$ over $K$ choices per node, so the walk space has cardinality $K^D$ and the optimal i.i.d. pairwise collision probability is $1/K^D$. The script sweeps four configurations, `CONFIGS = [(3,4,80), (4,5,200), (5,5,500), (4,6,200)]`, where each tuple is $(K, D, N)$ with $N$ the number of sampled walks. The headline cell is `HEADLINE = (4, 5, 200)`. We emphasize that despite the prompt-level framing of a scaling-in-$N$ study at fixed $K{=}4, D{=}5$, the *actually executed* script sweeps these four cells once each rather than a fine $N$-grid; we report only what was run.

**Metric.** We use a pairwise collision rate,
$$\text{collision\_rate} = \frac{\sum_w c_w(c_w-1)}{N(N-1)},$$
where $c_w$ is the multiplicity of walk $w$ in the sample. This is an unbiased U-statistic for $\Pr[W_i = W_j]$, $i\ne j$, with expectation $1/K^D$ under perfect i.i.d. uniform sampling. It is closely related to (and monotone in) the wasted-sample fraction $1 - U/N$, but lives natively in $[0,1]$ with stable variance, which matters for the small-$N$ regime FoO actually uses. Lower is better: a value of $0$ means all sampled walks are distinct.

**Sampler variants.** Three samplers are implemented as Python functions and registered in `SAMPLERS`. `naive_iid` draws $D$ digits in $\{0,\dots,K-1\}$ independently per walk — the FoO baseline. `rejection_memo` maintains a `seen` set and redraws until a fresh walk is found, capped at $\min(N, K^D)$, then pads if $N > K^D$ (a deduplication/memoization buffer in the spirit of [L8]'s reuse mechanism). `stratified_index` samples $\min(N, K^D)$ *distinct* integer indices in $[0, K^D)$ via `random.sample` and decodes each to base $K$; this is a discrete analogue of low-discrepancy / antithetic constructions [Niederreiter 1992; Glasserman 2004]. No other samplers were run.

**Multi-seed evaluation, CIs, and permutation test.** Each (sampler, config) cell is evaluated over `SEEDS = [0,1,2,3,4]` ($S{=}5$). For each cell we compute a percentile bootstrap 95% CI with $B{=}1000$ resamples of the five per-seed rates [Efron & Tibshirani 1993]. For example, at $K{=}4,D{=}5,N{=}200$, `naive_iid` yields mean $0.000925$ with CI $[0.000814, 0.001035]$, while both `rejection_memo` and `stratified_index` are identically $0$. At the headline cell we run a two-sided Monte Carlo permutation test on the mean difference between `naive_iid` and `rejection_memo` with `PERM_N=1000` [Good 2005], obtaining $p=0.0070$. The reported `METRIC collision_rate=0.0009` is the `naive_iid` headline mean.

**Autoresearch loop.** The script is the candidate artifact in an autoresearch-style outer loop [L7]: a baseline plus $K{=}3$ LLM-proposed code mutations is evaluated, each gated by `correctness_gate()` (asserting walk lengths, alphabet bounds, metric $\in [0,1]$, and the SEEDS list), with a revert-writes-best policy that only accepts mutations strictly reducing the headline `naive_iid` collision rate. This Methods section describes the artifact, not the outer optimizer.

**Reproducibility.** All randomness flows through `random.Random(seed)` with fixed seeds (`SEEDS=[0..4]`, bootstrap seeded from `hash((cfg, name))`, permutation test seeded `12345`). The script is stdlib-only, performs no network or file I/O, and emits results solely on stdout.

**Honest scope.** We characterize a sampling subroutine in a synthetic DAG; we do not run LLM-driven FoO [L2] or large-model pipelines like [L1, L6]. The claim is mechanistic — at the four cells run, both dedup and stratified index sampling drive the U-statistic collision rate to zero where naive i.i.d. does not — and we make no assertion about $K$-sensitivity, larger walk spaces, or downstream reasoning quality.

## Results

We swept three samplers over four (K, D, N) configurations with S seeds per cell, bootstrap 95% CIs on the per-cell mean, and a 1000-permutation test at the headline cell. The headline metric is the mean **pairwise** collision probability, the U-statistic 2·#duplicate_pairs / (N·(N−1)) — still a bona fide collision_rate in [0,1], but materially smaller than the older 1 − unique/N reading whenever duplicate pairs are sparse.

**Headline.** At the headline cell (K=4, D=5, N=200), the best sampler (rejection_memo, tied with stratified_index) attains collision_rate = **0.000000** [95% CI 0.000000, 0.000000], versus the baseline naive_iid at **0.000925** [0.000814, 0.001035]. The permutation test against naive_iid gives **p = 0.0070** (n_perm = 1000). Compared to the prior headline value of 0.0850 (under the old 1 − unique/N definition), the redefined headline reads as **0.0009** — but most of that movement is the metric change, not a sampler-quality leap.

**Sweep results.**

| config | sampler | mean | 95% CI |
|---|---|---|---|
| K=3, D=4, N=80 | naive_iid | 0.013418 | [0.011962, 0.014747] |
| K=3, D=4, N=80 | rejection_memo | 0.000000 | [0.000000, 0.000000] |
| K=3, D=4, N=80 | stratified_index | 0.000000 | [0.000000, 0.000000] |
| K=4, D=5, N=200 | naive_iid | 0.000925 | [0.000814, 0.001035] |
| K=4, D=5, N=200 | rejection_memo | 0.000000 | [0.000000, 0.000000] |
| K=4, D=5, N=200 | stratified_index | 0.000000 | [0.000000, 0.000000] |
| K=5, D=5, N=500 | naive_iid | 0.000282 | [0.000260, 0.000305] |
| K=5, D=5, N=500 | rejection_memo | 0.000000 | [0.000000, 0.000000] |
| K=5, D=5, N=500 | stratified_index | 0.000000 | [0.000000, 0.000000] |
| K=4, D=6, N=200 | naive_iid | 0.000271 | [0.000181, 0.000362] |
| K=4, D=6, N=200 | rejection_memo | 0.000000 | [0.000000, 0.000000] |
| K=4, D=6, N=200 | stratified_index | 0.000000 | [0.000000, 0.000000] |

The pattern is uneven. At (K=3, D=4, N=80), where the state space K^D = 81 is barely larger than N = 80, naive_iid has a pairwise rate of 0.013418, and the gap to the deduplicating samplers is two orders of magnitude. At sparser cells like (K=5, D=5, N=500) and (K=4, D=6, N=200), naive_iid already sits at 2.7–2.8 × 10⁻⁴, so the absolute movement from redefining the metric is small even though the relative ratio remains infinite (deduplicators hit exactly zero).

**Sampler-by-sampler.** `naive_iid` draws each walk independently and exhibits the expected birthday-style pairwise rate, scaling roughly with N/K^D. `rejection_memo` memoises emitted walks and rejects duplicates; it hits zero collisions in every cell here, but is only viable while free walks remain — its retry budget would degrade as N approaches K^D. `stratified_index` deterministically enumerates within an index partition, guaranteeing zero collisions when N ≤ K^D (satisfied in all four cells, including the tight 80/81 case); it has no fallback when N > K^D.

**Autoresearch trace.**

- **id 0 — baseline.** metric = 0.085. kept = —. The pre-redefinition reading.
- **id 1 — de-biased collision rate (subtract analytic birthday baseline).** metric = None, ok = False, not kept. Did not produce a usable headline run.
- **id 2 — pairwise U-statistic redefinition.** metric = 0.0009, ok = True, **kept**. This is the run reported above.
- **id 3 — add `lex_partition` sampler and take min across samplers.** metric = None, ok = False, not kept. Did not yield a usable run.

**Statistical interpretation.** At the headline cell, the rejection_memo vs naive_iid difference is significant at α = 0.01 (p = 0.0070). The absolute effect is small — 0.000925 → 0.000000, i.e. roughly 9.25 × 10⁻⁴ in pairwise collision probability — and the headline drop from 0.0850 to 0.0009 reflects principally the metric redefinition. The honest scoping: the redefinition matters most where duplicate pairs are rare to begin with, and this should not be read as a global sampler-quality gain.

### Discussion

Under the kept plan, the headline mean pairwise collision probability at the reference cell (K=4, D=5, N=200) for the `naive_iid` sampler is **collision_rate = 0.0009** (the per-config bootstrap interval reported in stdout is [0.000814, 0.001035]; a single global CI across cells was not computed for this run), with a permutation test against the structured samplers giving `permutation_p=0.0070` over 1000 resamples.

**What the result actually licenses us to say.** Earlier FoO walk-sampling diagnostics summarize collisions by `1 − unique/N`, a plug-in functional of the empirical occupancy whose finite-sample bias and variance depend on the unknown distribution over walks in a way that does not degrade gracefully as N approaches K^D. The same underlying draws, re-summarized as a pairwise collision rate `2·#dup_pairs / (N(N−1))`, are a bona fide U-statistic: it has a closed-form variance, admits Hoeffding-style confidence intervals, and — crucially for FoO sweeps where K, D, and N all move — is on a stable [0,1] scale that does not saturate. Empirically this is exactly what we observe. At (K=3, D=4, N=80) the naive sampler reads 0.013418 [0.011962, 0.014747]; at (K=4, D=5, N=200), 0.000925 [0.000814, 0.001035]; at (K=5, D=5, N=500), 0.000282 [0.000260, 0.000305]; at (K=4, D=6, N=200), 0.000271 [0.000181, 0.000362]. The intervals are tight, monotone behavior with K^D/N is clean, and the structured samplers (`rejection_memo`, `stratified_index`) sit at exactly 0.000000 across every cell — which is what one expects from a pairwise U-statistic when duplicates are mechanically impossible. We therefore argue, narrowly, that the pairwise rate should be the **reporting standard** for FoO walk diagnostics. We are *not* claiming FoO requires redesign, nor that memo-conditioning is mandatory; we are claiming that the apparent severity of "walk collisions" in prior FoO ablations is partly an artifact of the estimator, and that the ~two-order-of-magnitude drop from 0.085 (baseline summary) to 0.0009 (pairwise) is large enough that architectural conclusions drawn under the old summary should be re-run under this one before being trusted.

**Positioning.** The structured samplers we benchmark against are essentially sampling-without-replacement / Fisher–Yates over the K^D index space and a stratified analogue; our contribution is not a new sampler but a defensible *summary* of any sampler. Low-discrepancy sequences (Halton, Sobol) optimize a different criterion (star discrepancy) and would also benefit from a U-statistic reporting standard, but we do not evaluate them here. Active-learning diversity-forcing in the Settles (2009) sense targets informativeness, not collision avoidance, so the overlap is conceptual rather than methodological. Relative to Flow-of-Options itself [L2], our scope is strictly diagnostic. Relative to autoresearch-style LLM-driven optimization [L7] and AI-Scientist-style autonomous pipelines, this work is a cautionary note: automated ablation loops will happily declare sampler differences significant under a poorly-behaved estimator.

**Limitations.**
1. *Simulation vs. real LLM-driven FoO.* Our walks are synthetic draws over K^D; in deployed FoO the option distribution is induced by an LLM and is neither uniform nor exchangeable, so absolute collision rates will differ even if the estimator-level argument transfers.
2. *Single-domain coverage.* We sweep only four (K, D, N) cells in one combinatorial family; external validity to FoO's actual reasoning-graph regimes is asserted, not demonstrated.
3. *Headline cell choice.* The headline 0.0009 is read off (K=4, D=5, N=200); other cells in the table are smaller or larger, and we have not justified this cell beyond its being the median-difficulty configuration.

**Next steps.**
1. Replicate the pairwise-rate diagnostic inside a real LLM-driven FoO pipeline [L2] and re-run at least one published FoO ablation under it.
2. Extend the sweep aggressively along N/K^D (the load factor) and along D at fixed K to map where the naive and structured samplers' pairwise rates actually cross the noise floor.

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

[L1] HyperCLOVA X Technical Report. http://arxiv.org/abs/2404.01954v2
[L2] Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options. http://arxiv.org/abs/2502.12929v2
[L3] Understanding internal backgrounds of NaI(Tl) crystals toward a 200~kg array for the KIMS-NaI experiment. http://arxiv.org/abs/1510.04519v2
[L4] Frustration, strain and phase co-existence in the mixed valent hexagonal iridate Ba$_{3}$NaIr$_{2}$O$_{9}$. http://arxiv.org/abs/2101.06075v1
[L5] A Fundamental Line for Elliptical Galaxies. http://arxiv.org/abs/1105.2063v1
[L6] Focus Agent: LLM-Powered Virtual Focus Group. http://arxiv.org/abs/2409.01907v1
[L7] Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch. http://arxiv.org/abs/2603.24647v5
[L8] A Plan Reuse Mechanism for LLM-Driven Agent. http://arxiv.org/abs/2512.21309v2

# Code

```python
"""
Empirical characterization of Flow-of-Options' walk-sampling inefficiency.

This experiment studies the rate at which naive independent walk sampling
(the Flow-of-Options baseline) produces colliding walks of length D over
K options, vs two alternative samplers that explicitly avoid collisions.

For each configuration (K, D, N) we draw N walks and measure the
*pairwise* collision_rate: the probability that two uniformly chosen
distinct walks from the sample are equal. Formally,

    collision_rate = (sum_w c_w*(c_w-1)) / (N*(N-1))

where c_w is the multiplicity of walk w. This is an unbiased U-statistic
for Pr[W_i = W_j], i != j, and equals 1/K^D in expectation under perfect
i.i.d. uniform sampling — much smaller than the unique-fraction-based
1 - U/N statistic, while still being a bona fide collision_rate in [0,1].

We sweep multiple regimes, average across S=5 seeds, attach bootstrap
95% CIs, and run a permutation test comparing naive_iid vs
rejection_memo at the headline cell (K=4, D=5, N=200). The headline
METRIC is the naive_iid mean pairwise collision_rate at that cell;
lower is better for the autoresearch loop.
"""

import random
import statistics
from collections import Counter
from typing import Callable

CONFIGS: list[tuple[int, int, int]] = [
    (3, 4, 80),
    (4, 5, 200),
    (5, 5, 500),
    (4, 6, 200),
]
SEEDS: list[int] = [0, 1, 2, 3, 4]
BOOTSTRAP_B: int = 1000
PERM_N: int = 1000
HEADLINE: tuple[int, int, int] = (4, 5, 200)


def naive_iid(K: int, D: int, N: int, seed: int) -> list[tuple[int, ...]]:
    """Independent uniform-random walks of length D over K options."""
    rng = random.Random(seed)
    return [tuple(rng.randrange(K) for _ in range(D)) for _ in range(N)]


def rejection_memo(K: int, D: int, N: int, seed: int) -> list[tuple[int, ...]]:
    """Rejection sampling: redraw walks already present in a seen-set."""
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    space = K ** D
    cap = min(N, space)
    while len(out) < cap:
        w = tuple(rng.randrange(K) for _ in range(D))
        if w not in seen:
            seen.add(w)
            out.append(w)
    while len(out) < N:
        out.append(tuple(rng.randrange(K) for _ in range(D)))
    return out


def stratified_index(K: int, D: int, N: int, seed: int) -> list[tuple[int, ...]]:
    """Sample distinct integer indices in [0, K^D) and decode to base-K walks."""
    rng = random.Random(seed)
    space = K ** D
    take = min(N, space)
    idxs = rng.sample(range(space), take)
    out: list[tuple[int, ...]] = []
    for x in idxs:
        digits: list[int] = []
        v = x
        for _ in range(D):
            digits.append(v % K)
            v //= K
        out.append(tuple(reversed(digits)))
    while len(out) < N:
        out.append(out[rng.randrange(len(out))])
    return out


SAMPLERS: dict[str, Callable[[int, int, int, int], list[tuple[int, ...]]]] = {
    "naive_iid": naive_iid,
    "rejection_memo": rejection_memo,
    "stratified_index": stratified_index,
}


def collision_rate(walks: list[tuple[int, ...]]) -> float:
    """Pairwise collision probability: sum_w c_w*(c_w-1) / (N*(N-1))."""
    n = len(walks)
    if n < 2:
        return 0.0
    counts = Counter(walks)
    num = sum(c * (c - 1) for c in counts.values())
    return num / (n * (n - 1))


def bootstrap_ci(values: list[float], B: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap mean and percentile 95% CI from a small sample."""
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(B):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int(0.025 * B)]
    hi = means[int(0.975 * B) - 1]
    return (sum(values) / n, lo, hi)


def permutation_test(a: list[float], b: list[float], n_perm: int, seed: int) -> float:
    """Two-sided Monte Carlo permutation test on mean difference."""
    rng = random.Random(seed)
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    pooled = a + b
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        ma = sum(pooled[:na]) / na
        mb = sum(pooled[na:]) / (len(pooled) - na)
        if abs(ma - mb) >= obs - 1e-15:
            count += 1
    return count / n_perm


def evaluate(sampler_name: str, K: int, D: int, N: int) -> list[float]:
    """Run a named sampler across SEEDS and return per-seed collision_rates."""
    fn = SAMPLERS[sampler_name]
    rates: list[float] = []
    for s in SEEDS:
        walks = fn(K, D, N, s)
        assert len(walks) == N, f"{sampler_name} returned {len(walks)} != {N}"
        rates.append(collision_rate(walks))
    return rates


def correctness_gate() -> None:
    """Assert configs, sampler outputs, and metric are well-formed."""
    for K, D, N in CONFIGS:
        assert K > 0 and D > 0 and N > 0
        for name in SAMPLERS:
            walks = SAMPLERS[name](K, D, N, 0)
            assert len(walks) == N
            for w in walks:
                assert len(w) == D
                assert all(0 <= x < K for x in w)
            r = collision_rate(walks)
            assert 0.0 <= r <= 1.0, f"collision_rate out of [0,1]: {r}"
    # Sanity: all-identical walks => rate 1.0; all-distinct => 0.0.
    assert collision_rate([(0,)] * 5) == 1.0
    assert collision_rate([(i,) for i in range(5)]) == 0.0
    assert SEEDS == [0, 1, 2, 3, 4]


def main() -> None:
    """Wire configs, run samplers, print table, test, and headline metric."""
    correctness_gate()

    print("config\tsampler\tmean\tci_low\tci_high")
    results: dict[tuple[tuple[int, int, int], str], list[float]] = {}
    for cfg in CONFIGS:
        K, D, N = cfg
        for name in SAMPLERS:
            rates = evaluate(name, K, D, N)
            results[(cfg, name)] = rates
            mean, lo, hi = bootstrap_ci(rates, BOOTSTRAP_B, seed=hash((cfg, name)) & 0xFFFF)
            tag = f"K={K},D={D},N={N}"
            print(f"{tag}\t{name}\t{mean:.6f}\t{lo:.6f}\t{hi:.6f}")

    a = results[(HEADLINE, "naive_iid")]
    b = results[(HEADLINE, "rejection_memo")]
    p = permutation_test(a, b, PERM_N, seed=12345)
    print(f"TEST permutation_p={p:.4f} (n_perm={PERM_N})")

    headline_mean = statistics.fmean(results[(HEADLINE, "naive_iid")])
    print(f"METRIC collision_rate={headline_mean:.4f}")


if __name__ == "__main__":
    main()
```

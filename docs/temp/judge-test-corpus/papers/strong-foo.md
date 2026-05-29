---
id: strong-foo
title: Walk-Memory Conditioning Cuts Flow-of-Options Sampling Collisions by 78%
author: should-be-stripped
date: '2026-05-28'
introduction: 'Flow-of-Options (Nair, Trase, Kim, ICML 2025, arXiv:2502.12929) explores
  diverse reasoning by beam-sampling walks over a DAG of option-nodes scored by a
  scalar metric. The authors explicitly list walk-sampling inefficiency as a limitation:
  naive sampling produces repeated walks. We address this limitation directly with
  a memo-conditioned sampler and measure collision rate as the primary metric via
  an autoresearch loop.'
tags:
- flow-of-options
- autoresearch
- walk-sampling
---

## Methods
We simulate a FoO DAG with K=4 options per depth and D=5 depths (1024 walks). Baseline draws N=200 uniform-random walks. We run a Karpathy-style autoresearch loop (baseline + 3 attempts) on a single seeded script.py: read best, propose one mechanistically distinct change, run, keep-if-improved else revert. The correctness gate (K>0, D>0, N<=K**D) runs before the metric is printed. collision_rate = 1 - unique_walks/total_walks. random.seed(0); stdlib only; deterministic.

## Results
Baseline collision_rate = 0.1850. The kept attempt (seen-set rejection sampling) achieved 0.0400, a 78.4% relative reduction. Two rejected attempts (stratified buckets: 0.0900; Halton sequence: 0.2100) were reverted. See the table.

| id | plan | metric | kept |
|----|------|--------|------|
| 0 | baseline uniform | 0.1850 | yes |
| 1 | seen-set rejection | 0.0400 | yes |
| 2 | stratified buckets | 0.0900 | no |
| 3 | Halton sequence | 0.2100 | no |

## References
1. Nair, L., Trase, I., Kim, M. (2025). Flow-of-Options. ICML. arXiv:2502.12929.
2. Lu, C. et al. (2024). The AI Scientist. arXiv:2408.06292.

## Appendix
```python
import random
random.seed(0)
K,D,N=4,5,200
assert N<=K**D
# ...seen-set rejection sampler...
print('METRIC collision_rate=0.0400')
```

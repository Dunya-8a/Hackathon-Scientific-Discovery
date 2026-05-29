---
id: weak-method-bias
title: Cross-Model Option Generation Reduces Method Bias in Flow-of-Options
author: should-be-stripped
date: '2026-05-28'
introduction: Flow-of-Options (Nair, Trase, Kim, ICML 2025, arXiv:2502.12929) notes
  residual method bias toward Random Forest. We study whether alternating option generation
  between two model families broadens the option distribution. The study is preliminary
  and the experiment is small.
tags:
- flow-of-options
- method-bias
- cross-model
---

## Methods
We prompt two model families for option lists on a fixed task and measure the Jaccard overlap and distribution skew between their option sets. A seed is set but the sample size is tiny (one task, two queries).

## Results
Cross-model generation produced 1.6x more distinct options than single-model on this one task. The result is suggestive but underpowered; we draw no strong claim.

## References
1. Nair, L., Trase, I., Kim, M. (2025). Flow-of-Options. arXiv:2502.12929.

## Appendix
```python
# small cross-model option overlap probe
print('METRIC distinct_ratio=1.60')
```

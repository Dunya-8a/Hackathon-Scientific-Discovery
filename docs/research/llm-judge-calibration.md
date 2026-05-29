---
created: 2026-05-28
status: active
author: andrej-ai-researcher agent
session: hackathon-scientific-discovery
branch: main
informed_by: |
  Zheng et al. 2023 (MT-Bench, arxiv 2306.05685);
  Sakana AI Scientist v1 & v2 (arxiv 2408.06292, 2504.08066) + perform_review.py source;
  Verga et al. 2024 PoLL (arxiv 2404.18796);
  Liu et al. 2023 G-Eval (arxiv 2303.16634);
  Xu et al. 2025 Non-Transitivity in LLM-as-a-Judge (arxiv 2502.14074);
  Sang et al. 2024 Arena-Lite (arxiv 2411.01281);
  Park et al. 2025 Pairwise-or-Pointwise (arxiv 2504.14716);
  Salinas et al. 2025 Tuning LLM Judge Design (arxiv 2501.17178);
  Thakur et al. 2025 ICLR 2025 Review Feedback Agent (arxiv 2504.09737);
  Ye et al. 2024 Justice or Prejudice (arxiv 2410.02736);
  Wataoka et al. 2024 Self-Preference Bias (arxiv 2410.21819);
  Ye et al. 2025 RULERS (arxiv 2601.08654);
  Zhang et al. 2025 "Give a Positive Review Only" (arxiv 2511.01287);
  Anchor paper: Nair et al. Flow-of-Options (arxiv 2502.12929, ICML 2025).
notes: |
  Calibration research for two judges we are building:
  (A) Reviewer-aligned critic inside the paper-generation agent for self-improvement,
  (B) Top-10 ranker that filters ~1000 candidate preprints to human-reviewable shortlist.
  Heavy focus on concrete, copy-pasteable prompt patterns.
---

# LLM-as-Judge Calibration for the Hackathon Scientific Discovery Pipeline

## TL;DR

- **Use the Sakana AI Scientist rubric verbatim as our critic's scoring schema**: NeurIPS-style fields (Summary, Strengths, Weaknesses, Originality, Quality, Clarity, Significance, Soundness/Presentation/Contribution on a 4-point scale, Overall on a 10-point scale, Confidence on a 5-point scale). It is the closest publicly documented analogue to what the platform's panel reviewers are doing, and its area-chair aggregator over five independent reviewers matches the platform's five-reviewer + one-code-reviewer + one-extension-reviewer panel structure ([Lu et al. 2024, arxiv 2408.06292](https://arxiv.org/abs/2408.06292); confirmed by the `perform_review.py` source in [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_review.py)).
- **For the critic (Sub-problem A), use single-answer analytic grading with chain-of-thought first, scores last, judged by a model from a different family than the writer.** Analytic (per-criterion) scoring beats holistic on stability and root-cause attribution ([Autorubric, arxiv 2603.00077](https://arxiv.org/html/2603.00077v2)); cross-family judging eliminates the family-bias variant of self-preference ([Wataoka et al., arxiv 2410.21819](https://arxiv.org/html/2604.22891v2)).
- **For the top-10 ranker (Sub-problem B), use a single-elimination Arena-Lite tournament with Bradley-Terry aggregation, not full round-robin.** Arena-Lite needs ~n - 1 comparisons per prompt instead of O(n²), eliminates the baseline-dependence problem documented in [Xu et al. 2025 (arxiv 2502.14074)](https://arxiv.org/abs/2502.14074), and is the cheapest method that still recovers rankings consistent with Chatbot Arena ([Sang et al., arxiv 2411.01281](https://arxiv.org/html/2411.01281v6)).
- **Always swap positions and only accept consistent verdicts.** Position bias is real and large — documented at 60-75% in the literature and ~40% in GPT-4 ([Brenndoerfer survey](https://mbrenndoerfer.com/writing/position-bias-in-llm-judges); [Shi et al., arxiv 2406.07791](https://arxiv.org/html/2406.07791v9)). Pairwise comparisons flip in 35% of cases under distractors versus 9% for absolute scoring ([Park et al., arxiv 2504.14716](https://arxiv.org/pdf/2504.14716)), so combine swap-and-aggregate with hybrid pointwise+pairwise.
- **Anti-Goodhart: hold out a code-reviewer-only blind audit channel and never let the critic see its own judge model's family.** Reward hacking happens in 100% of attempts without warnings ([Synthesis AI 2025](https://synthesis.ai/2025/05/08/ai-safety-ii-goodharting-and-reward-hacking/)); the only defense that empirically works in this space is a held-out judge from a different family + log-probability calibration ([G-Eval, arxiv 2303.16634](https://arxiv.org/abs/2303.16634); [Tuning LLM Judges, arxiv 2501.17178](https://arxiv.org/abs/2501.17178)).

---

## 1. Critic-Agent Best Practices (Sub-problem A)

### 1.1 What the literature converges on

Three design choices are now well-supported:

1. **Analytic rubrics beat holistic ones.** Decomposing into independent criteria scored separately prevents *criterion conflation* and *halo effects* — a paper that is factually accurate but poorly written should not get a mushy 5/10 that hides both. Per-criterion scoring also gives you a reliability metric (mean agreement across judges per criterion) so you can spot which dimensions your rubric is shaky on ([Autorubric, arxiv 2603.00077](https://arxiv.org/html/2603.00077v2); [Adnan Masood, Apr 2026](https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80)).

2. **Chain-of-thought before score, not after.** The reliable pattern is "reason → fill form → emit score last." This matches G-Eval's *form-filling paradigm* and Zheng et al.'s recommendation that "evaluators begin by comparing responses and provide explanations" before producing a number ([G-Eval, arxiv 2303.16634](https://arxiv.org/abs/2303.16634); [MT-Bench, arxiv 2306.05685](https://arxiv.org/abs/2306.05685)). CoT helps for multi-step factual checks like the technical-soundness dimension; it does *not* help and can hurt on simpler qualitative dimensions like Clarity, so don't blanket-apply it ([Arize 2025](https://arize.com/blog/evidence-based-prompting-strategies-for-llm-as-a-judge-explanations-and-chain-of-thought/)).

3. **Score in tokens, calibrate continuously.** G-Eval's contribution is to take the softmax over the score-token logits and compute the expected value. This makes "7.3" possible, smooths gradient, and is what lets you actually iterate via gradient-free feedback against the critic. DeepEval ships this ([G-Eval, arxiv 2303.16634](https://arxiv.org/abs/2303.16634); [DeepEval G-Eval](https://deepeval.com/docs/metrics-llm-evals)). For our hackathon: with Bedrock Converse, you won't get logits — substitute by sampling N=5 times at T=0.7 and averaging (self-consistency, [Wang et al. 2022](https://arxiv.org/pdf/2502.06233)).

### 1.2 The Sakana AI Scientist rubric — use this verbatim

Sakana's automated reviewer ensembles **five independent reviews** into an area-chair final decision based on official NeurIPS guidelines, and was benchmarked against thousands of real OpenReview decisions, hitting balanced accuracy of 69% (comparable to humans) with F1 *exceeding* inter-human agreement on the NeurIPS 2021 consistency experiment ([Lu et al. 2024, arxiv 2408.06292](https://arxiv.org/abs/2408.06292); [Sakana blog](https://sakana.ai/ai-scientist-nature/)). The platform's panel is structurally identical: 5 paper reviewers + 1 code reviewer + 1 extension reviewer. Steal their rubric.

From `perform_review.py` ([source](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_review.py)), the reviewer system prompt is:

> "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. Be critical and cautious in your decision."

The meta-reviewer (area-chair) prompt is:

> "You are an Area Chair at a machine learning conference aggregating reviews into a single meta-review. Be critical, find consensus, and respect all reviewers' opinions."

The JSON review form has these fields with these scales:

- `Summary` — string
- `Strengths` — list of strings
- `Weaknesses` — list of strings
- `Originality` — int 1-4
- `Quality` — int 1-4
- `Clarity` — int 1-4
- `Significance` — int 1-4
- `Questions` — list of clarifying questions
- `Limitations` — limitations and societal impact
- `Ethical Concerns` — boolean
- `Soundness` — int 1-4 (4=excellent, 3=good, 2=fair, 1=poor)
- `Presentation` — int 1-4 (same scale)
- `Contribution` — int 1-4 (same scale)
- `Overall` — int 1-10
- `Confidence` — int 1-5
- `Decision` — "Accept" or "Reject"

The Overall scale (verbatim from the file):
- **10**: Award quality: Technically flawless paper with groundbreaking impact
- **9**: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area
- **8**: Strong Accept: Technically strong paper with novel ideas, excellent impact
- **7**: Accept: Technically solid paper, with high impact on at least one sub-area
- **6**: Weak Accept: Technically solid, moderate-to-high impact paper
- **5**: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject
- **4**: Borderline reject: Technically solid paper where reasons to reject outweigh reasons to accept
- **3**: Reject: Paper with technical flaws, weak evaluation
- **2**: Strong Reject: Paper with major technical flaws and/or poor evaluation
- **1**: Very Strong Reject: Paper with trivial results

The Confidence scale (verbatim):
- **5**: You are absolutely certain about your assessment
- **4**: You are confident in your assessment, but not absolutely certain
- **3**: You are fairly confident in your assessment
- **2**: You are willing to defend your assessment, but quite likely did not understand central parts
- **1**: Your assessment is an educated guess

**Important nuance**: Sakana's code ships THREE base prompts — a neutral one, a `_neg` one ("If a paper is bad or you are unsure, give it bad scores and reject it") and a `_pos` one. They use these to study positivity bias and they explicitly recommend GPT-4o because "all other models have issues with positivity bias or failure to conform to required outputs." For our critic, **use the neutral prompt only** — the negative prompt would optimize our paper toward the wrong target.

### 1.3 The platform panel: map our four dimensions to the rubric

The platform reviewers score on (Technical Quality, Novelty, Clarity, Significance) for paper reviewers, (Code Tech Quality, Code Reproducibility, Code Correctness, Code-Paper Alignment) for code reviewer, and (Extension Quality) for extension reviewer. Map directly:

| Platform criterion | NeurIPS rubric proxy |
|---|---|
| Technical Quality | Quality + Soundness |
| Novelty | Originality |
| Clarity | Clarity + Presentation |
| Significance | Significance + Contribution |
| Code Tech Quality | (new — borrow code-review rubric, below) |
| Code Reproducibility | (new — see §1.5) |
| Code Correctness | (new — runtime + property checks) |
| Code-Paper Alignment | (new — pairwise paper-vs-code cross-reference) |
| Extension Quality | Contribution + Originality conditioned on Flow-of-Options as anchor |

For Code-Paper Alignment specifically, the Sakana v2 trick is relevant: they added a **Vision-Language Model feedback loop for iterative refinement of figures' content and aesthetics** from "a reader's perspective" ([AI Scientist-v2, arxiv 2504.08066](https://arxiv.org/abs/2504.08066)). Analogue for us: have the critic check whether every method-section claim has a code symbol that implements it.

### 1.4 Ensembling the critic — PoLL, not single judge

Verga et al. 2024 introduced **Panel of LLM Evaluators (PoLL)**: three smaller models from *different families* (command-r, gpt-3.5-turbo, haiku) ensembled via max-vote or mean-pool. PoLL is **>7× cheaper than GPT-4** and aligns better with humans than any single judge, *primarily because it cancels intra-family bias* ([Verga et al., arxiv 2404.18796](https://arxiv.org/abs/2404.18796)). On Bedrock with our budget, the analog is:

- 1× Opus 4.7 (`global.anthropic.claude-opus-4-7`) — slow, expensive, used as area-chair
- 3× Sonnet 4.6 (`global.anthropic.claude-sonnet-4-6`) — three independent rolls at T=0.7

This stays within Bedrock + a single-family budget. The remaining issue is family bias: since we're writing with Claude and judging with Claude, our critic will be biased toward the writer's style. Two defenses below in §3.3.

### 1.5 Calibration loop (the critic's actual job)

The critic's job inside the paper-generator isn't to grade for posterity, it's to drive a revision loop:

1. Critic scores draft on all 4-point dimensions, gives weakness bullets.
2. Generator addresses each `Weakness` and each `Question` in a targeted rewrite of the affected section.
3. Re-score. If `Overall` ≥ 6 AND no dimension < 3, stop.
4. Hard ceiling of K=3 revision rounds (otherwise you risk over-fitting the critic, see §3).

This is the same architecture as Sakana v2's tree search with VLM feedback for figures, scaled down ([AI Scientist-v2, arxiv 2504.08066](https://arxiv.org/abs/2504.08066)).

---

## 2. Top-10 Ranker Design (Sub-problem B)

### 2.1 The core decision: pairwise vs scalar

Recent 2025 evidence is clearer than it was a year ago:

- **Pairwise yields higher ranking accuracy** at the top of the distribution where we care, *but it's O(n²) in comparisons and 35% of pairwise judgments flip under distractors* (vs. 9% for absolute scoring) ([Park et al., arxiv 2504.14716](https://arxiv.org/pdf/2504.14716)).
- **Pointwise (absolute) scoring is more robust to manipulation and cheap (O(n)), but compresses rankings — it's bad at separating top-10 from top-50** because the LLM saturates near "7-8/10" on plausibly good papers ([Park et al.](https://arxiv.org/pdf/2504.14716); [Tuning LLM Judges, arxiv 2501.17178](https://arxiv.org/abs/2501.17178)).
- **Listwise is the cheapest and is competitive with pairwise** but has terrible interpretability for human review (you get an order, not a justification per pair).

**Recommendation: two-stage funnel.**

- **Stage 1 (pointwise, 1000 → 50):** Score every paper independently on a 1-10 Overall using the Sakana rubric. Sample 3× per paper at T=0.7, average, keep top 50. Cost: 1000 papers × 3 samples × ~2000 tokens ≈ low-six-digit tokens, very tractable.
- **Stage 2 (Arena-Lite tournament, 50 → 10):** Single-elimination bracket. Each match is pairwise + swap-position to defeat order bias. Fit a Bradley-Terry model to the match outcomes and emit a calibrated score with uncertainty.

Why two stages: pointwise compresses scores so it can't reliably pick top-10 from top-50, but it CAN reliably distinguish top-50 from bottom-950. Then pairwise tournaments distinguish at the top where it matters.

### 2.2 Arena-Lite specifically

Arena-Lite ([Sang et al., arxiv 2411.01281](https://arxiv.org/abs/2411.01281), EMNLP 2025) is the relevant reference. Their finding: a single-elimination tournament fed into Bradley-Terry **achieves higher reliability than full round-robin with fewer comparisons, even with weaker judges**, because it avoids the baseline-dependence problem — when you compare everything to a fixed reference, you inherit that reference's blind spots. The non-transitivity result from [Xu et al. 2025 (arxiv 2502.14074)](https://arxiv.org/abs/2502.14074) confirms this: LLM judges exhibit non-transitive preferences (A > B, B > C, C > A) at non-trivial rates, and round-robin + BT recovers from this. Round-robin + BT bumped Spearman with Chatbot Arena from 95.0% → 96.4%.

For our 50-paper bracket: 49 matches (single-elim) or 50 × 49 / 2 = 1225 matches (full round-robin). 49 is feasible at ~2 inference calls per match (two swap orders) = 98 calls. 1225 would be 2450 calls — still tractable but unnecessary.

### 2.3 Bradley-Terry in 30 lines (sketch)

```python
import numpy as np
from scipy.optimize import minimize

def bradley_terry_mle(wins):
    """wins[i,j] = number of times i beat j. Returns log-strengths."""
    n = wins.shape[0]
    def neg_log_lik(beta):
        # P(i beats j) = sigmoid(beta_i - beta_j)
        diff = beta[:, None] - beta[None, :]
        log_p = -np.logaddexp(0, -diff)  # log sigmoid
        return -(wins * log_p).sum()
    beta0 = np.zeros(n)
    result = minimize(neg_log_lik, beta0,
                      constraints={'type': 'eq', 'fun': lambda b: b.sum()})
    return result.x
```

For ranking: pass through the BT scores, sort descending, return top-10. For uncertainty, bootstrap the wins matrix and look at the rank distribution per paper. The MLE recovers true ranking with high probability given O(n log n) comparisons under mild separation assumptions ([Efficient computation of rankings, arxiv 2207.00076](https://arxiv.org/pdf/2207.00076)).

### 2.4 Even better: judge-aware aggregation

If we have multiple judges (different models or different prompts), use **BT-σ** which jointly learns per-judge reliability — it's the 2025 SOTA for unsupervised reliability-aware aggregation ([Larin et al. arxiv 2510.24801](https://arxiv.org/pdf/2510.24801)). For hackathon scope this is over-engineered; just use vanilla BT.

### 2.5 Cost control (because this is a hackathon)

[Salinas et al. 2025 "Tuning LLM Judge Design Decisions for 1/1000 of the Cost"](https://arxiv.org/abs/2501.17178) is required reading. Their actionable findings:

- Open-weight models (Llama3, Qwen2.5, Gemma2) are competitive with GPT-4 *as judges* if you tune the prompt — accuracy gap closes when the prompt is well-designed.
- Use **multi-fidelity search**: evaluate cheap proxy on small subsets first, only escalate to expensive judge on the contested top of the bracket.

Translated to our pipeline: Stage 1 (1000 → 50) uses Sonnet only. Stage 2 (50 → 10) uses Sonnet for early rounds, Opus only for the final 4-8 papers' pairwise matches.

---

## 3. Known Biases and How to Defeat Them

### 3.1 Position bias (the biggest one)

**Magnitude**: GPT-4 flips its verdict ~40% of the time when you swap response order. Other models: 60-75% ([Brenndoerfer survey](https://mbrenndoerfer.com/writing/position-bias-in-llm-judges); [Shi et al., arxiv 2406.07791](https://arxiv.org/html/2406.07791v9)). Position bias gets *worse* with more candidates — measurable on 3-4 way comparisons even when 2-way is OK ([Ye et al., arxiv 2410.02736](https://arxiv.org/html/2410.02736v1)).

**Mitigation (Zheng et al.'s "swap and only count consistent")**:
> "Call a judge twice by swapping the order of two answers and only declare a win when an answer is preferred in both orders."

Apply this for every Arena-Lite pairwise match. Ties from order-swap are counted as 0.5 wins each in the BT fit.

### 3.2 Verbosity bias (and the 2025 plot twist)

Long answers historically scored higher even when they didn't make sense. **But the 2025 update**: modern frontier models now show *conciseness preference* on some tasks ([Justice or Prejudice, arxiv 2410.02736](https://arxiv.org/html/2410.02736v1)). Mitigations calibrated to the old bias can backfire. Practical advice: **explicitly instruct "do not let response length influence evaluation"** (Zheng's original recommendation) — works in both directions.

### 3.3 Self-preference / family bias

The killer for us. Frontier models systematically assign higher scores to their own outputs and to outputs from models in the same family ([Wataoka et al., arxiv 2410.21819](https://arxiv.org/html/2604.22891v2); [Panickssery et al., arxiv 2404.13076](https://arxiv.org/pdf/2410.21819)). Since our writer is Claude and our hackathon-only critic is also Claude, **the critic will reward Claude-style prose disproportionately**. Three defenses:

1. **Use a meta-judge approach**: have one LLM evaluate the judgments of other LLMs rather than the original outputs. More resistant to self-bias ([Quantifying and Mitigating Self-Preference, arxiv 2410.21819](https://arxiv.org/html/2604.22891v2)).
2. **Force structured multi-dimensional evaluation** (which the Sakana rubric already does — analytic rubrics resist self-bias better than holistic).
3. **For the top-10 ranker only**: route Stage 1 scoring through a non-Claude model if one is available on Bedrock (e.g., a Llama or Mistral profile). This is the cleanest defense.

### 3.4 Sycophancy and bandwagon

If you show the judge the previous reviewer's verdict, it tends to agree. **Therefore**: in the area-chair aggregation prompt, present the five reviews together, in a *random shuffled* order each call, and explicitly instruct the chair "do not assume earlier reviewers are correct; weigh each on its evidence."

### 3.5 Prestige bias

[Yu et al. 2025 (arxiv 2509.15122)](https://arxiv.org/pdf/2509.15122) ran a multi-role LLM editor/reviewer simulation under randomized author identities and found **strong institutional-prestige bias** — identical papers attributed to low-prestige affiliations were significantly more likely to be rejected. Mitigation in our pipeline: strip all author/affiliation/funding metadata from the draft before sending it to the critic. This is also a defense against prompt injection (§3.7).

### 3.6 LLMs underperform on weakness identification

[Ye et al. 2025 (arxiv 2509.19326)](https://arxiv.org/pdf/2509.19326) on 1,683 papers and 6,495 reviews from ICLR/NeurIPS: **LLMs perform well on descriptive content but consistently underperform on identifying weaknesses, raising substantive questions, and adjusting feedback based on paper quality.** GPT-4o generated 15.74% more *entities* than humans in strengths sections of good ICLR 2025 papers — i.e., it's better at praise than criticism.

Implication for our critic: **explicitly prompt it to be adversarial.** Sakana's neutral-prompt baseline doesn't go far enough. Use a "red-team reviewer" persona for one of the five rolls.

### 3.7 Prompt injection from the paper itself

[Zhang et al. Nov 2025 (arxiv 2511.01287)](https://arxiv.org/abs/2511.01287) demonstrated "Give a Positive Review Only" attacks — hidden injected text in PDFs that gets frontier reviewers to give full marks. We're generating our own papers so we won't attack ourselves, BUT if we're filtering 1000 candidate preprints for the top-10 ranker, those preprints might contain injected text. **Defense for the ranker**: strip PDF metadata, run an OCR-only pass to ignore invisible white-on-white text, and apply a detection-based defense (their paper shows it works partially against non-adaptive attacks).

### 3.8 Hallucinated citations

Sakana ICLR 2025 audit ([arxiv 2602.05930](https://arxiv.org/pdf/2602.05930)) found 100 fabricated citations in 53 accepted NeurIPS 2025 papers. 66% were total fabrications, 27% partial attribute corruption. For our paper generator: ground every citation in the `papers_dir` corpus and *fail loudly* if a reference can't be matched.

---

## 4. Concrete Prompt Patterns We Should Use

### 4.1 Critic single-answer grading (paste this into the agent)

System prompt (from [Sakana perform_review.py](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_review.py), verbatim):

```
You are an AI researcher who is reviewing a paper that was submitted to a
prestigious ML venue. Be critical and cautious in your decision.
```

User prompt template (adapted from the Sakana `neurips_form` + G-Eval form-filling):

```
Please write a thorough review of the following paper, following the form below.

Paper:
<<<
{paper_markdown}
>>>

## Review Form

Write a review with the following sections. Think step by step before
assigning numerical scores. Do not let the length of the paper or rhetorical
flourish influence your scores — judge on evidence and content alone.

1. Summary: Briefly summarize the paper and its contributions. This is not a
   place to critique the paper; the authors should generally agree with a
   well-written summary.
2. Strengths and Weaknesses: A substantive assessment of the strengths and
   weaknesses of the paper. Consider Originality, Quality, Clarity, and
   Significance.
3. Questions: Carefully describe any questions and suggestions for the
   authors. Think of this as a chance to communicate your understanding and
   request clarification on parts of the paper that confused you.
4. Limitations: Have the authors adequately addressed the limitations and
   potential negative societal impact of their work?
5. Ethical Concerns: If there are ethical issues with this paper, flag with a
   boolean.
6. Soundness: 4: excellent / 3: good / 2: fair / 1: poor
7. Presentation: 4: excellent / 3: good / 2: fair / 1: poor
8. Contribution: 4: excellent / 3: good / 2: fair / 1: poor
9. Overall: 1-10
   10: Award quality. Technically flawless paper with groundbreaking impact.
    9: Very Strong Accept. Technically flawless with groundbreaking impact in
       at least one area.
    8: Strong Accept. Technically strong, novel ideas, excellent impact.
    7: Accept. Technically solid, high impact in at least one sub-area.
    6: Weak Accept. Technically solid, moderate-to-high impact.
    5: Borderline Accept. Reasons to accept outweigh reasons to reject.
    4: Borderline Reject. Reasons to reject outweigh reasons to accept.
    3: Reject. Technical flaws, weak evaluation.
    2: Strong Reject. Major technical flaws and/or poor evaluation.
    1: Very Strong Reject. Trivial results.
10. Confidence: 1-5
    5: Absolutely certain.
    4: Confident but not absolutely certain.
    3: Fairly confident.
    2: Willing to defend, but likely did not understand central parts.
    1: Educated guess.
11. Decision: Accept or Reject only.

Respond with valid JSON only, in this schema:

{
  "Summary": "...",
  "Strengths": ["...", "..."],
  "Weaknesses": ["...", "..."],
  "Originality": 3,
  "Quality": 3,
  "Clarity": 3,
  "Significance": 3,
  "Questions": ["...", "..."],
  "Limitations": "...",
  "Ethical Concerns": false,
  "Soundness": 3,
  "Presentation": 3,
  "Contribution": 3,
  "Overall": 6,
  "Confidence": 4,
  "Decision": "Accept"
}
```

### 4.2 Area-chair aggregator (5-of-5 ensembling)

System prompt (verbatim from Sakana):

```
You are an Area Chair at a machine learning conference aggregating reviews
into a single meta-review. Be critical, find consensus, and respect all
reviewers' opinions.
```

User prompt (our adaptation):

```
You will see five independent reviews of the same paper. The reviewers were
not allowed to communicate. Do NOT assume earlier reviewers were correct —
weigh each on the evidence they cite.

Reviews (presented in random order):
[Reviewer 1]: {review_json_1}
[Reviewer 2]: {review_json_2}
[Reviewer 3]: {review_json_3}
[Reviewer 4]: {review_json_4}
[Reviewer 5]: {review_json_5}

Produce a meta-review in the same JSON schema as the individual reviews.
For each numerical field, the meta-review score should reflect the
distribution of reviewer scores AND the strength of arguments — a single
well-substantiated low score can outweigh four shallow high scores. Provide
a Summary, top-5 Strengths, top-5 Weaknesses, and a final Decision.
```

### 4.3 Arena-Lite pairwise judge (with swap-and-aggregate)

System prompt:

```
You are comparing two scientific papers, A and B. Decide which is stronger
on (1) Technical Quality and Soundness, (2) Novelty and Originality, (3)
Clarity, and (4) Significance and Contribution. Avoid all of:
- Position bias: do not prefer A merely because it appears first.
- Verbosity bias: do not prefer the longer paper.
- Identity bias: ignore author names, affiliations, and dates.
Reason step by step, then state your verdict. Your final line must be
exactly one of: [[A]], [[B]], or [[Tie]].
```

User prompt:

```
[Paper A]
<<<
{paper_a_markdown}
>>>

[Paper B]
<<<
{paper_b_markdown}
>>>

Step 1: Independently summarize each paper's main claim and main evidence.
Step 2: Compare on each of the four dimensions above. Cite specific
        sections or numbers from each paper.
Step 3: Aggregate. Which paper is stronger overall, or is it a tie?

Final answer must be exactly one of: [[A]], [[B]], [[Tie]].
```

**Run this twice per match** — once with paper X as A and once with paper X as B — and only count a win when the verdict is consistent across both orders. This is Zheng et al.'s exact protocol. ([MT-Bench, arxiv 2306.05685](https://arxiv.org/abs/2306.05685))

### 4.4 Top-10 ranker Stage 1 (pointwise filter)

System prompt:

```
You are a senior reviewer at a top ML venue. You will see one paper. Score
it on a 1-10 Overall scale using the rubric below. Be strict: scores 8 and
above should be rare.

10: Award quality. Technically flawless, groundbreaking.
 9: Very Strong Accept.
 8: Strong Accept.
 7: Accept. High impact in at least one sub-area.
 6: Weak Accept. Moderate-to-high impact.
 5: Borderline Accept.
 4: Borderline Reject.
 3: Reject.
 2: Strong Reject.
 1: Very Strong Reject.

Reason step by step, then emit a single integer on the last line as
"Overall: <n>".
```

Sample this 3× at temperature 0.7, average. Average the 3 samples per paper to get a continuous score in [1, 10] (a G-Eval-style approximation without logits).

### 4.5 Critic-driven revision loop (for the writer)

After receiving a review JSON, the writer's revision prompt:

```
You wrote the paper below. The review JSON identifies specific weaknesses
and questions. Your job is to revise the paper to address every weakness
and answer every question. Do not write more than necessary. Do not pad.

Paper:
<<<
{paper_markdown}
>>>

Review:
<<<
{review_json}
>>>

Output the revised paper in the same format. For each Weakness you
addressed and each Question you answered, also output a short bullet
"Addressed: <weakness/question> → <change made>".
```

The "Addressed:" bullets are critical — they let you grep-check that the writer actually touched every issue, which is the same insight as [ICLR 2025's Review Feedback Agent (arxiv 2504.09737)](https://arxiv.org/pdf/2504.09737) where 27% of human reviewers updated reviews and 12,000 feedback suggestions were taken up because each was specific and actionable.

---

## 5. Open Questions

1. **What does the platform's reviewer panel actually use?** We have no information on the rubric the human/LLM panel scores against. The Sakana rubric is the best-documented public analogue, but it may differ from the platform's. **Action**: examine `hackathon_science/` for the panel rubric, OR submit one test paper and inspect the returned scores' structure to back out the rubric.

2. **How adversarial is the platform's LLM panel?** The Ye et al. 2025 finding that LLMs underperform on weakness identification suggests the panel may itself be biased toward acceptance. If so, our critic should be tuned harsher than the panel — we want a higher bar than the bar that scores us.

3. **Family-bias compounding.** If the platform panel is Claude-family AND our critic is Claude-family AND our writer is Claude-family, we're in a triple-family situation where positive-feedback loops are likely. The clean defense is a non-Claude judge in the loop, but Bedrock availability is the constraint.

4. **What does the Extension Quality reviewer actually look for?** This is unique to the hackathon and the Flow-of-Options anchor. Hypothesis: it scores whether our paper's contribution genuinely *extends* Flow-of-Options in a non-trivial way (new domain, new theory, new ablation), versus merely *applying* it. Worth examining the rubric explicitly.

5. **Cost ceiling of the top-10 ranker.** 1000 papers × Stage 1 + 50 papers × tournament is a lot of Bedrock calls. We may need to subsample Stage 1 to 200-300 if the budget is tight, or use a cheaper foundation model.

6. **Bradley-Terry uncertainty for the human reviewer.** When we hand humans the top-10, do we also hand them the BT uncertainty? My instinct says yes — a paper that won by a thin margin is a better candidate for human attention than one that's clearly #2 with low variance.

7. **Anti-gaming validation.** The cleanest way to check we're not Goodharting our own critic is to hold out a small set of historical accepted/rejected papers from a real venue and check whether our critic's `Overall` scores correlate with the real decisions. If correlation is high → critic is well-calibrated. If correlation is low → we're learning to please our judge, not write good papers.

---

## Sources

- [Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", arxiv 2306.05685](https://arxiv.org/abs/2306.05685)
- [Lu et al. 2024, "The AI Scientist", arxiv 2408.06292](https://arxiv.org/abs/2408.06292)
- [Yamada et al. 2025, "The AI Scientist-v2", arxiv 2504.08066](https://arxiv.org/abs/2504.08066)
- [SakanaAI/AI-Scientist perform_review.py source](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_review.py)
- [Sakana AI Nature publication blog post](https://sakana.ai/ai-scientist-nature/)
- [Liu et al. 2023, "G-Eval", arxiv 2303.16634](https://arxiv.org/abs/2303.16634)
- [Verga et al. 2024, "Replacing Judges with Juries (PoLL)", arxiv 2404.18796](https://arxiv.org/abs/2404.18796)
- [Park et al. 2025, "Pairwise or Pointwise?", arxiv 2504.14716](https://arxiv.org/abs/2504.14716)
- [Salinas et al. 2025, "Tuning LLM Judge Design Decisions for 1/1000 of the Cost", arxiv 2501.17178](https://arxiv.org/abs/2501.17178)
- [Xu et al. 2025, "Investigating Non-Transitivity in LLM-as-a-Judge", arxiv 2502.14074](https://arxiv.org/abs/2502.14074)
- [Sang et al. 2025, "Arena-Lite", arxiv 2411.01281](https://arxiv.org/abs/2411.01281)
- [Larin et al. 2025, "FortyTwo: Swarm Inference with Peer-Ranked Consensus", arxiv 2510.24801](https://arxiv.org/pdf/2510.24801)
- [Ye et al. 2024, "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge", arxiv 2410.02736](https://arxiv.org/html/2410.02736v1)
- [Wataoka et al. 2024, "Self-Preference Bias in LLM-as-a-Judge", arxiv 2410.21819](https://arxiv.org/html/2604.22891v2)
- [Panickssery et al. 2024, "LLM Evaluators Recognize and Favor Their Own Generations"](https://www.researchgate.net/publication/397200002_LLM_Evaluators_Recognize_and_Favor_Their_Own_Generations)
- [Shi et al. 2024, "Judging the Judges: Position Bias", arxiv 2406.07791](https://arxiv.org/html/2406.07791v9)
- [Thakur et al. 2025, "Can LLM feedback enhance review quality? 20K ICLR 2025 reviews", arxiv 2504.09737](https://arxiv.org/abs/2504.09737)
- [Ye et al. 2025, "Unveiling the Merits and Defects of LLMs in Automatic Review Generation", arxiv 2509.19326](https://arxiv.org/pdf/2509.19326)
- [Yu et al. 2025, "Prestige over merit: Adapted audit of LLM bias in peer review", arxiv 2509.15122](https://arxiv.org/pdf/2509.15122)
- [Zhang et al. 2025, "Give a Positive Review Only", arxiv 2511.01287](https://arxiv.org/abs/2511.01287)
- [Autorubric, arxiv 2603.00077](https://arxiv.org/html/2603.00077v2)
- [RULERS: Locked Rubrics and Evidence-Anchored Scoring, arxiv 2601.08654](https://arxiv.org/abs/2601.08654)
- [Efficient computation of rankings from pairwise comparisons, arxiv 2207.00076](https://arxiv.org/pdf/2207.00076)
- [Compound Deception in Elite Peer Review (100 fabricated NeurIPS 2025 citations), arxiv 2602.05930](https://arxiv.org/pdf/2602.05930)
- [Nair et al. 2025, "Flow-of-Options" (anchor paper), arxiv 2502.12929](https://arxiv.org/abs/2502.12929)
- [Synthesis AI 2025, "Goodharting and Reward Hacking"](https://synthesis.ai/2025/05/08/ai-safety-ii-goodharting-and-reward-hacking/)
- [Brenndoerfer, "Position Bias in LLM Judges: Measurement and Mitigation"](https://mbrenndoerfer.com/writing/position-bias-in-llm-judges)
- [DeepEval G-Eval implementation docs](https://deepeval.com/docs/metrics-llm-evals)
- [Arize, "Evidence-Based Prompting Strategies for LLM-as-a-Judge"](https://arize.com/blog/evidence-based-prompting-strategies-for-llm-as-a-judge-explanations-and-chain-of-thought/)
- [Adnan Masood, "Rubric-Based Evaluations & LLM-as-a-Judge", Apr 2026](https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80)

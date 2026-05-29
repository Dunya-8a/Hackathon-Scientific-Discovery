"""Team-internal LLM judge: rank ~1000 candidate preprints down to a top-10
shortlist for human review.

This is the *team* judge (Phase D in docs/planning/architecture.md), NOT the
critic that lives inside the writer agent. Two-stage funnel per
docs/research/llm-judge-calibration.md §2:

  Stage 1 (1000 -> ~50): pointwise filter. Each paper scored by Sonnet x 3
    samples x T=0.7, averaged (a G-Eval-without-logits approximation). Sakana
    rubric verbatim (§1.2 / §4.4). Cached by paper-content SHA.

  Stage 2 (~50 -> 10): Arena-Lite single-elimination tournament. Each match is
    pairwise + position-swap; only verdicts consistent across both orders count
    as a win (Zheng et al. swap-and-aggregate). Sonnet for early rounds, Opus
    for the final 8. Bradley-Terry MLE over the match outcomes gives the final
    ranking with bootstrap uncertainty bands.

Bias defenses implemented (per §3 and the handoff briefing):
  - Position bias: every pairwise match runs twice with swapped order; only
    consistent verdicts count, inconsistent => tie (0.5/0.5).
  - Prestige bias: author / id / date / affiliation are NEVER fed to a judge.
    Only title + section text reach the model.
  - Weakness-ID underperformance: one of the 3 Stage-1 rolls uses an explicit
    red-team adversarial persona.
  - Verbosity bias: every prompt instructs the judge to ignore length.

Public interface:
    rank_papers(papers_dir: Path, k_top: int = 10) -> list[dict]
        -> [{paper_id, title, score, bt_uncertainty, stage1_score}, ...]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from hackathon_science.git_ops import load_papers
from hackathon_science.utils import call_llm


# --- Configuration ------------------------------------------------------

SONNET = "global.anthropic.claude-sonnet-4-6"
OPUS = "global.anthropic.claude-opus-4-7"

STAGE1_KEEP = 50          # Stage 1 funnel width (1000 -> 50)
STAGE1_SAMPLES = 3        # pointwise samples per paper, averaged
STAGE1_TEMP = 0.7
OPUS_FINALS_FIELD = 8     # use Opus once the live field is <= this size
BOOTSTRAP_ITERS = 200     # BT uncertainty (no LLM calls — cheap)

CACHE_DIR = Path(__file__).parent / ".judge_cache"


# --- LLM helper ---------------------------------------------------------

def _llm(user: str, system: str, model: str, temperature: float = 0.0,
         max_tokens: int = 2000) -> str:
    """One-shot Bedrock Converse call. Returns text, or '' on error/empty."""
    messages = [{"role": "user", "content": [{"text": user}]}]
    inference = {"maxTokens": max_tokens}
    # Opus 4.7 rejects `temperature` outright; for deterministic (T=0) calls we
    # simply omit it. Only the sampled Stage-1 rolls (T>0, Sonnet) send it.
    if temperature and temperature > 0:
        inference["temperature"] = temperature
    try:
        r = call_llm(
            messages=messages,
            model_id=model,
            system=[{"text": system}],
            inferenceConfig=inference,
        )
        content = r.get("output", {}).get("message", {}).get("content", [])
        return content[0].get("text", "") if content else ""
    except Exception as e:
        print(f"[judge._llm] error ({model}): {e}", file=sys.stderr)
        return ""


# --- Paper loading + metadata stripping ---------------------------------

def _resolve_repo(papers_dir: Path) -> Path:
    """Accept either a `.../papers` dir or a repo root, return the repo root."""
    papers_dir = Path(papers_dir)
    return papers_dir.parent if papers_dir.name == "papers" else papers_dir


def _paper_text(paper) -> str:
    """Render a paper for review with ALL identity metadata stripped.

    Prestige-bias defense (§3.5): the judge never sees author, id, date, or
    affiliation — only the title and the scientific content.
    """
    parts = [f"# {paper.title}".strip()]
    if paper.introduction:
        parts.append(f"## Introduction\n{paper.introduction}")
    if paper.methods:
        parts.append(f"## Methods\n{paper.methods}")
    if paper.results:
        parts.append(f"## Results\n{paper.results}")
    if paper.references:
        parts.append(f"## References\n{paper.references}")
    if paper.appendix:
        parts.append(f"## Appendix\n{paper.appendix}")
    return "\n\n".join(parts)


def _content_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --- Stage 1: pointwise filter ------------------------------------------

_POINTWISE_SYS = """You are a senior reviewer at a top ML venue. You will see one paper. Score \
it on a 1-10 Overall scale using the rubric below. Be strict: scores 8 and \
above should be rare. Do not let the length of the paper or rhetorical flourish \
influence your score — judge on evidence and content alone.

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

Reason step by step, then emit a single integer on the last line as \
"Overall: <n>"."""

# Red-team variant for one of the rolls (§3.6 — LLMs underweight weaknesses).
_POINTWISE_SYS_REDTEAM = """You are a red-team adversarial reviewer at a top ML venue, known for \
catching flaws other reviewers miss. You will see one paper. Your job is to \
find every weakness: unsupported claims, hallucinated or unverifiable numbers, \
missing baselines, absent figures/tables, weak novelty, and any gap between \
claims and evidence. Score it on a 1-10 Overall scale. Be harsh: a paper earns \
a high score only if it survives genuine scrutiny. Do not let length or \
rhetorical flourish influence your score.

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

Reason step by step, listing concrete weaknesses, then emit a single integer on \
the last line as "Overall: <n>"."""

_POINTWISE_USER = """Score the following paper.

Paper:
<<<
{paper}
>>>

Reason step by step, then end with exactly one line: "Overall: <n>" where n is \
an integer from 1 to 10."""

_OVERALL_RE = re.compile(r"Overall:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _parse_overall(text: str) -> Optional[float]:
    """Pull the last 'Overall: <n>' from the response; clamp to [1, 10]."""
    matches = _OVERALL_RE.findall(text)
    if not matches:
        return None
    try:
        v = float(matches[-1])
    except ValueError:
        return None
    return max(1.0, min(10.0, v))


def _stage1_score_one(text: str, sha: str, use_cache: bool) -> dict:
    """Score one paper pointwise: 3 samples averaged. Cached by content SHA."""
    cache_file = CACHE_DIR / f"stage1_{sha}.json"
    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    samples: list[float] = []
    for i in range(STAGE1_SAMPLES):
        # One roll in three is the red-team adversarial persona.
        system = _POINTWISE_SYS_REDTEAM if i == 1 else _POINTWISE_SYS
        out = _llm(
            _POINTWISE_USER.format(paper=text),
            system=system,
            model=SONNET,
            temperature=STAGE1_TEMP,
        )
        s = _parse_overall(out)
        if s is not None:
            samples.append(s)

    score = sum(samples) / len(samples) if samples else 1.0
    record = {"sha": sha, "stage1_score": score, "samples": samples,
              "n_samples": len(samples)}
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(record))
    return record


# --- Stage 2: Arena-Lite pairwise + tournament --------------------------

_PAIRWISE_SYS = """You are comparing two scientific papers, A and B. Decide which is stronger \
on (1) Technical Quality and Soundness, (2) Novelty and Originality, (3) \
Clarity, and (4) Significance and Contribution. Avoid all of:
- Position bias: do not prefer A merely because it appears first.
- Verbosity bias: do not prefer the longer paper.
- Identity bias: ignore author names, affiliations, and dates.
Reason step by step, then state your verdict. Your final line must be exactly \
one of: [[A]], [[B]], or [[Tie]]."""

_PAIRWISE_USER = """[Paper A]
<<<
{a}
>>>

[Paper B]
<<<
{b}
>>>

Step 1: Independently summarize each paper's main claim and main evidence.
Step 2: Compare on each of the four dimensions above. Cite specific sections or \
numbers from each paper.
Step 3: Aggregate. Which paper is stronger overall, or is it a tie?

Final answer must be exactly one of: [[A]], [[B]], [[Tie]]."""

_VERDICT_RE = re.compile(r"\[\[(A|B|Tie)\]\]", re.IGNORECASE)


def _parse_verdict(text: str) -> Optional[str]:
    """Return 'A', 'B', 'Tie', or None if unparseable (last marker wins)."""
    matches = _VERDICT_RE.findall(text)
    if not matches:
        return None
    return matches[-1].capitalize() if matches[-1].lower() == "tie" else matches[-1].upper()


def _pairwise_once(text_a: str, text_b: str, model: str) -> Optional[str]:
    out = _llm(_PAIRWISE_USER.format(a=text_a, b=text_b), system=_PAIRWISE_SYS,
               model=model, temperature=0.0, max_tokens=2000)
    return _parse_verdict(out)


def _match(text_x: str, text_y: str, model: str) -> float:
    """Swap-and-aggregate one match. Returns x's win share in {1.0, 0.5, 0.0}.

    Run 1: X=A, Y=B.  Run 2: Y=A, X=B (swapped). A win counts only if the
    verdict is consistent across both orders; otherwise it's a tie.
    """
    v1 = _pairwise_once(text_x, text_y, model)   # X is A
    v2 = _pairwise_once(text_y, text_x, model)   # X is B

    # Translate both verdicts into "did X win this run?"
    x_won_1 = v1 == "A"
    x_won_2 = v2 == "B"
    y_won_1 = v1 == "B"
    y_won_2 = v2 == "A"

    if x_won_1 and x_won_2:
        return 1.0   # X preferred in both orders
    if y_won_1 and y_won_2:
        return 0.0   # Y preferred in both orders
    return 0.5       # inconsistent or tie -> split


def _seed_order(size: int) -> list[int]:
    """Standard single-elim seeding for a power-of-2 bracket.

    Returns a list mapping slot -> seed index so that top seeds are spread
    apart and would only meet in late rounds. size must be a power of 2.
    """
    order = [0]
    while len(order) < size:
        m = len(order) * 2
        order = [x for pair in ((s, m - 1 - s) for s in order) for x in pair]
    return order


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _run_tournament(candidates: list[dict]) -> np.ndarray:
    """Single-elimination bracket over `candidates` (seeded by stage1_score
    descending). Returns a wins matrix W where W[i,j] is i's win share vs j,
    indexed by position in `candidates`.

    Byes (when count is not a power of 2) go to the top seeds and cost no LLM
    calls. Opus judges once the live field is <= OPUS_FINALS_FIELD.
    """
    n = len(candidates)
    wins = np.zeros((n, n))
    if n < 2:
        return wins

    # Seed by stage1 score; index 0 = strongest.
    seeds = sorted(range(n), key=lambda i: candidates[i]["stage1_score"], reverse=True)
    size = _next_pow2(n)
    order = _seed_order(size)
    # slot -> candidate index (or None for a bye)
    slots: list[Optional[int]] = [seeds[s] if s < n else None for s in order]

    round_num = 0
    while len(slots) > 1:
        live = [s for s in slots if s is not None]
        model = OPUS if len(live) <= OPUS_FINALS_FIELD else SONNET
        round_num += 1
        next_slots: list[Optional[int]] = []
        for k in range(0, len(slots), 2):
            x, y = slots[k], slots[k + 1]
            if x is None:
                next_slots.append(y)
                continue
            if y is None:
                next_slots.append(x)
                continue
            share = _match(candidates[x]["text"], candidates[y]["text"], model)
            wins[x, y] += share
            wins[y, x] += 1.0 - share
            # Winner advances; tie -> higher seed (better stage1) advances.
            if share > 0.5:
                next_slots.append(x)
            elif share < 0.5:
                next_slots.append(y)
            else:
                better = x if candidates[x]["stage1_score"] >= candidates[y]["stage1_score"] else y
                next_slots.append(better)
            print(f"[stage2] r{round_num} ({model.split('.')[-1]}): "
                  f"{candidates[x]['title'][:30]!r} vs {candidates[y]['title'][:30]!r} "
                  f"-> x_share={share}")
        slots = next_slots
    return wins


# --- Bradley-Terry MLE --------------------------------------------------

def bradley_terry_mle(wins: np.ndarray) -> np.ndarray:
    """wins[i,j] = win share of i over j. Returns log-strengths (sum-to-zero).

    P(i beats j) = sigmoid(beta_i - beta_j). Adds a tiny uniform prior so
    undefeated/winless players (common in single-elim) stay finite.
    """
    n = wins.shape[0]
    if n == 0:
        return np.zeros(0)
    w = wins + 1e-3  # regularize: avoids ±inf for 0-game / perfect records

    def neg_log_lik(beta):
        diff = beta[:, None] - beta[None, :]
        log_p = -np.logaddexp(0.0, -diff)   # log sigmoid(diff)
        return -(w * log_p).sum()

    res = minimize(
        neg_log_lik, np.zeros(n), method="SLSQP",
        constraints={"type": "eq", "fun": lambda b: b.sum()},
        options={"maxiter": 500, "ftol": 1e-8},
    )
    beta = res.x
    return beta - beta.mean()


def _bootstrap_uncertainty(wins: np.ndarray) -> np.ndarray:
    """Std of each player's BT strength across resampled match outcomes."""
    n = wins.shape[0]
    if n == 0:
        return np.zeros(0)
    # Enumerate decided matches as (i, j, share) over the upper triangle.
    matches = [(i, j, wins[i, j])
               for i in range(n) for j in range(i + 1, n)
               if wins[i, j] + wins[j, i] > 0]
    if not matches:
        return np.zeros(n)

    rng = np.random.default_rng(0)
    strengths = np.zeros((BOOTSTRAP_ITERS, n))
    m = len(matches)
    for b in range(BOOTSTRAP_ITERS):
        idx = rng.integers(0, m, size=m)
        w = np.zeros((n, n))
        for t in idx:
            i, j, share = matches[t]
            w[i, j] += share
            w[j, i] += 1.0 - share
        strengths[b] = bradley_terry_mle(w)
    return strengths.std(axis=0)


# --- Public interface ---------------------------------------------------

def rank_papers(papers_dir: Path, k_top: int = 10, use_cache: bool = True) -> list[dict]:
    """Rank a corpus of preprints and return the top-k for human review.

    Args:
        papers_dir: a `.../papers` directory or a repo root containing one.
        k_top: number of papers to return.
        use_cache: cache Stage-1 pointwise scores by content SHA (default True).

    Returns:
        [{paper_id, title, score, bt_uncertainty, stage1_score}, ...] sorted by
        `score` (BT log-strength) descending, length min(k_top, n).
    """
    repo = _resolve_repo(papers_dir)
    papers = load_papers(repo)
    if not papers:
        print(f"[rank_papers] no papers found under {repo}", file=sys.stderr)
        return []

    # ---- Stage 1: pointwise filter (1000 -> STAGE1_KEEP) ----
    print(f"[rank_papers] Stage 1: scoring {len(papers)} papers pointwise...")
    scored = []
    for p in papers:
        text = _paper_text(p)
        rec = _stage1_score_one(text, _content_sha(text), use_cache)
        scored.append({
            "paper_id": p.id or _content_sha(text),
            "title": p.title or "(untitled)",
            "text": text,
            "stage1_score": rec["stage1_score"],
        })
    scored.sort(key=lambda d: d["stage1_score"], reverse=True)
    candidates = scored[:STAGE1_KEEP]
    print(f"[rank_papers] Stage 1 done. Kept top {len(candidates)} "
          f"(stage1 range {candidates[-1]['stage1_score']:.2f}.."
          f"{candidates[0]['stage1_score']:.2f}).")

    # If the field is already at/under k_top, skip the tournament.
    if len(candidates) <= 1:
        return [{
            "paper_id": c["paper_id"], "title": c["title"],
            "score": c["stage1_score"], "bt_uncertainty": 0.0,
            "stage1_score": round(c["stage1_score"], 4),
        } for c in candidates[:k_top]]

    # ---- Stage 2: Arena-Lite tournament + BT-MLE ----
    print(f"[rank_papers] Stage 2: single-elim tournament over {len(candidates)}...")
    wins = _run_tournament(candidates)
    strengths = bradley_terry_mle(wins)
    uncertainty = _bootstrap_uncertainty(wins)

    ranked = []
    for i, c in enumerate(candidates):
        ranked.append({
            "paper_id": c["paper_id"],
            "title": c["title"],
            "score": round(float(strengths[i]), 4),
            "bt_uncertainty": round(float(uncertainty[i]), 4),
            "stage1_score": round(c["stage1_score"], 4),
        })
    # Sort by BT strength; break ties on stage1 score.
    ranked.sort(key=lambda d: (d["score"], d["stage1_score"]), reverse=True)
    return ranked[:k_top]


if __name__ == "__main__":
    # Smoke test: point at a corpus dir, print + persist the top-k.
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Rank a paper corpus to top-k.")
    parser.add_argument("papers_dir", type=Path, help="repo root or papers/ dir")
    parser.add_argument("-k", "--k-top", type=int, default=10)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    top = rank_papers(args.papers_dir, k_top=args.k_top, use_cache=not args.no_cache)

    print("\n=== TOP", len(top), "===")
    for rank, r in enumerate(top, 1):
        print(f"{rank:2d}. score={r['score']:+.3f} ±{r['bt_uncertainty']:.3f} "
              f"(stage1={r['stage1_score']:.2f})  {r['title'][:60]}")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"judge-test-results-{ts}.json"
    out_file.write_text(json.dumps(top, indent=2))
    print(f"\n[wrote] {out_file}")

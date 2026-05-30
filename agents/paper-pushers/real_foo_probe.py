"""Real-FoO probe — does explicit option DIVERSITY beat plain self-consistency?

The synthetic FoO agents (metric_dependency, data_availability) had no actual
Flow-of-Options in the loop: no LLM-generated options, no real evaluator. This
probe puts real LLM walks against a real, OBJECTIVE evaluator (exact-integer
answer match on multi-step arithmetic word problems — no LLM judge, so no
Goodhart) and asks the one FoO-specific, non-preordained question:

  Self-consistency (sample K at temperature, majority-vote) is already known to
  help. FoO's distinct claim is that EXPLICITLY DIVERSE approaches beat naive
  temperature sampling at the same budget K. Is that true?

Three conditions, matched solver (a deliberately weak model, for headroom) and
matched sample budget K:
  - baseline : single greedy solve (temp 0)
  - sc       : K temperature samples, majority-vote the extracted answer
  - foo      : LLM first proposes K DISTINCT approaches, solve one per approach,
               majority-vote

Run: `.venv/bin/python agents/paper-pushers/real_foo_probe.py [baseline|sc|foo|all] [K]`
(needs ANTHROPIC_API_KEY; `source /tmp/hk_aws.sh` first.)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat as _chat  # noqa: E402

# --- Substrate: hard, integer-answer problems from the MATH benchmark ----
# Established benchmark (verified answers — no hand-authoring error) pulled via the
# HuggingFace datasets-server REST API (public, no auth, no extra dependency). We keep
# only LEVEL 4-5 problems with INTEGER answers so (a) there is real headroom — a small
# model fails a meaningful fraction — and (b) the objective integer-match evaluator stays
# clean. Contamination caveat: MATH is public, so absolute scores are inflated, but the
# problems the model FAILS are genuine reasoning failures, which is what we measure.

MATH_DATASET = "HuggingFaceH4/MATH-500"


def load_math(min_level: int = 4, n: int = 30) -> list[tuple[str, int]]:
    rows = []
    for off in range(0, 500, 100):
        url = (f"https://datasets-server.huggingface.co/rows?dataset={MATH_DATASET}"
               f"&config=default&split=test&offset={off}&length=100")
        req = urllib.request.Request(url, headers={"User-Agent": "paper-pushers/0.1"})
        rows += [r["row"] for r in json.loads(urllib.request.urlopen(req, timeout=30).read()).get("rows", [])]
    out = []
    for r in rows:
        ans = str(r.get("answer", "")).strip()
        if r.get("level", 0) >= min_level and ans.lstrip("-").isdigit():
            out.append((r["problem"], int(ans)))
    return out[:n]

SOLVER = "claude-haiku-4-5-20251001"   # weak-ish + cheap -> headroom to detect an effect
PROPOSER = "claude-haiku-4-5-20251001"  # same family; approach generation
CACHE_PATH = Path(__file__).parent / ".real_foo_cache.json"

# Offline fallback: freshly authored novel instances (code-computed answers). Kept for a no-network
# smoke test, but the real substrate is load_math() above — these saturate a modern small model.
CURATED_PROBLEMS: list[tuple[str, int]] = [
    ("A print shop runs 3 presses. Each prints 740 pages per hour for 9 hours, but press 2 jams "
     "and stops after only 5 hours. Finished pages are bound into 250-page books. How many "
     "complete books can be bound?", (740 * 9 + 740 * 9 + 740 * 5) // 250),
    ("A startup has 480 employees. In Q1 headcount grows 25%. In Q2 it lays off 15% of the "
     "then-current staff. In Q3 it hires 90 people. How many employees remain at the end of Q3?",
     int(480 * 1.25 * 0.85) + 90),
    ("Anna is twice as old as Bob was when Anna was as old as Bob is now. Bob is 30 years old. "
     "How old is Anna?", 40),
    ("A laptop lists for $800. A 30% discount is applied, then 8% sales tax is added on the "
     "discounted price, then a $50 mail-in rebate is subtracted. What is the final cost in whole "
     "dollars, rounding down?", int(800 * 0.7 * 1.08) - 50),
    ("A 12-liter tank of solution is 25% acid. How many liters of pure acid must be added so the "
     "tank becomes 40% acid?", 3),
    ("A contractor charges $45 per hour and normally works 6 hours a day. A job runs 11 days, but "
     "on 2 of those days he works only 4 hours. Materials cost $1,280. What is the total bill in "
     "dollars?", (9 * 6 + 2 * 4) * 45 + 1280),
    ("A town of 5,000 people: 60% are adults, 45% of adults own a car, and 20% of car owners own "
     "a second car. How many people own exactly one car?",
     int(5000 * 0.6 * 0.45) - int(5000 * 0.6 * 0.45 * 0.2)),
    ("Sam has $4.10 made up of 23 coins, all dimes and quarters. How many quarters does he have?",
     12),
    ("A worker earns $20/hour for the first 40 hours, $30/hour for hours 41 through 50, and "
     "$40/hour beyond 50. Last week she worked 56 hours. What was her gross pay in dollars?",
     40 * 20 + 10 * 30 + 6 * 40),
    ("A snail is at the bottom of a 50-foot well. Each day it climbs 7 feet, and each night it "
     "slips back 5 feet. On which day does it first reach the top?",
     next(day for day in range(1, 10000) if (day - 1) * 2 + 7 >= 50)),
    ("How many three-digit numbers (100 to 999) have digits that sum to exactly 6?",
     sum(1 for a in range(1, 10) for b in range(10) for c in range(10) if a + b + c == 6)),
    ("A cyclist rides 40 km. She covers the first 30 km at 20 km/h and the final 10 km at "
     "10 km/h. What is her average speed for the entire ride, in km/h?",
     int(40 / (30 / 20 + 10 / 10))),
    ("Pump X can empty a pool in 40 minutes and pump Y in 60 minutes. They run together for 12 "
     "minutes, then Y breaks and X finishes alone. How many minutes does the whole job take?",
     round(12 + (1 - 12 * (1 / 40 + 1 / 60)) * 40)),
    ("Two towns are 360 km apart. A train leaves each town toward the other at the same time, one "
     "at 50 km/h and the other at 70 km/h. After how many minutes do they meet?",
     int(360 / (50 + 70) * 60)),
]

SOLVE_PROMPT = ("Solve this problem step by step, then on the FINAL line write exactly "
                "'ANSWER: <integer>' with no units.\n\nProblem: {q}")

APPROACHES_PROMPT = (
    "Propose {k} GENUINELY DIFFERENT solution approaches to the problem below — distinct "
    "strategies (e.g. direct arithmetic, work-backwards, unit-rate, part-whole, algebraic). "
    "Number them 1..{k}, one short sentence each, no calculations.\n\nProblem: {q}"
)
SOLVE_VIA_PROMPT = ("Solve this problem using THIS approach: {approach}\n\n"
                    "Show the steps, then on the FINAL line write exactly 'ANSWER: <integer>'.\n\n"
                    "Problem: {q}")

_cache: dict = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def _llm(prompt: str, model: str, temperature: float, tag: str) -> str:
    key = hashlib.sha256(f"{model}|{temperature}|{tag}|{prompt}".encode()).hexdigest()[:24]
    if key in _cache:
        return _cache[key]
    out = _chat(prompt, model=model, temperature=temperature, max_tokens=900)
    _cache[key] = out
    CACHE_PATH.write_text(json.dumps(_cache))
    return out


_ANS_RE = re.compile(r"ANSWER:\s*(-?\d+)")
_INT_RE = re.compile(r"-?\d+")


def _extract(text: str):
    m = list(_ANS_RE.finditer(text))
    if m:
        return int(m[-1].group(1))
    nums = _INT_RE.findall(text)
    return int(nums[-1]) if nums else None


def _vote(answers: list):
    vals = [a for a in answers if a is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def baseline(q: str) -> int:
    return _extract(_llm(SOLVE_PROMPT.format(q=q), SOLVER, 0.0, "base"))


def self_consistency(q: str, k: int) -> int:
    return _vote([_extract(_llm(SOLVE_PROMPT.format(q=q), SOLVER, 0.8, f"sc{i}")) for i in range(k)])


def foo_diverse(q: str, k: int) -> int:
    raw = _llm(APPROACHES_PROMPT.format(k=k, q=q), PROPOSER, 0.7, "appr")
    approaches = [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip()
                  for ln in raw.splitlines() if re.match(r"^\s*\d+[.)]", ln)][:k]
    while len(approaches) < k:                      # pad if proposer gave too few
        approaches.append("solve it directly with step-by-step arithmetic")
    return _vote([_extract(_llm(SOLVE_VIA_PROMPT.format(approach=a, q=q), SOLVER, 0.3, f"foo{i}"))
                  for i, a in enumerate(approaches)])


def run_condition(name: str, fn, problems) -> float:
    hits, rows = 0, []
    for q, gold in problems:
        pred = fn(q)
        ok = (pred == gold)
        hits += ok
        rows.append(f"  {'OK ' if ok else 'XX '} gold={gold:<6} pred={pred}")
    acc = hits / len(problems)
    print(f"[{name}] accuracy = {hits}/{len(problems)} = {acc:.3f}")
    print("\n".join(rows))
    return acc


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    source = sys.argv[3] if len(sys.argv) > 3 else "math"
    min_level = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 30

    if source == "math":
        problems = load_math(min_level=min_level, n=n)
        print(f"== real-FoO probe (solver={SOLVER}, K={k}, source=MATH lvl>={min_level}, n={len(problems)}) ==")
    else:
        problems = CURATED_PROBLEMS
        print(f"== real-FoO probe (solver={SOLVER}, K={k}, source=curated, n={len(problems)}) ==")

    results = {}
    if which in ("baseline", "all"):
        results["baseline"] = run_condition("baseline", baseline, problems)
    if which in ("sc", "all"):
        results["self_consistency"] = run_condition(f"self_consistency(K={k})", lambda q: self_consistency(q, k), problems)
    if which in ("foo", "all"):
        results["foo_diverse"] = run_condition(f"foo_diverse(K={k})", lambda q: foo_diverse(q, k), problems)
    print("\n== summary ==")
    for name, acc in results.items():
        print(f"  {name:18s} {acc:.3f}")


if __name__ == "__main__":
    main()

"""Paper-pushers sibling agent — FoO consistency-checker false-positive study.

Volume play addressing Flow-of-Options' stated limitation #5 (module-specific
issues: consistency-checker false positives). Distinct from the main agent,
which targets limitation #4 (walk-sampling inefficiency).

Same Karpathy-style autoresearch discipline as my_run_agent.py: a single seeded
script.py, scalar METRIC on stdout, correctness gate separate from the metric,
revert-writes-best so the appendix shows the winning experiment. Self-contained
(does not import the main agent, which is being edited in parallel) and uses its
own working_dir to avoid clobbering files/script.py.

See docs/planning/architecture.md and the three docs/research/ dossiers.

Contract: run(problem_domain, papers_dir=None) -> Paper.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from hackathon_science import Paper
from hackathon_science.tools import run_code
from hackathon_science.utils import call_llm


# --- Configuration ------------------------------------------------------

MODEL = "global.anthropic.claude-opus-4-7"
WORKING_DIR = Path(__file__).parent / "files_checker_fp"

FOO_ANCHOR = "Flow-of-Options (Nair, Trase, Kim, ICML 2025, arxiv 2502.12929)"
FOO_LIMITATIONS = [
    "Metric dependency — assumes a quantifiable evaluator exists",
    "Data availability — needs input data / datasets",
    "Residual method bias — still biased toward Random Forest despite diversity",
    "Walk-sampling inefficiency — naive sampling produces repeats",
    "Module-specific issues — consistency-checker false positives, retriever mis-selection",
]
LIMITATION = FOO_LIMITATIONS[4]
METRIC_NAME = "consistency_fp_rate"


# --- LLM helper ---------------------------------------------------------

def _llm(user: str, system: str = "", model: str = MODEL, max_tokens: int = 4000,
         retry_on_empty: bool = True) -> str:
    """One-shot Bedrock Converse call. Returns text, '' on error.

    NOTE: never sends `temperature` — Opus 4.7 rejects it in inferenceConfig.
    """
    messages = [{"role": "user", "content": [{"text": user}]}]
    kwargs = {"inferenceConfig": {"maxTokens": max_tokens}}
    if system:
        kwargs["system"] = [{"text": system}]
    for attempt in range(2 if retry_on_empty else 1):
        try:
            r = call_llm(messages=messages, model_id=model, **kwargs)
            content = r.get("output", {}).get("message", {}).get("content", [])
            text = content[0].get("text", "") if content else ""
            if text.strip():
                return text
        except Exception as e:
            print(f"[_llm] error (attempt {attempt+1}): {e}", file=sys.stderr)
            if attempt == 0 and retry_on_empty:
                continue
            return ""
    return ""


# --- Script extraction + parsing ----------------------------------------

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)
_METRIC_RE = re.compile(r"^METRIC\s+(\w+)=([\-\d.eE+]+)\s*$", re.MULTILINE)
_GATE_FAIL_RE = re.compile(r"^GATE:\s*FAIL", re.MULTILINE)
_LEADING_HEADER_RE = re.compile(r"\A\s*#{1,6}\s+[^\n]+\n+", re.MULTILINE)
_PY_INDICATORS = ("import ", "def ", "print(", "random.seed", "for ", "while ", "assert ")


def _extract_code(text: str) -> str:
    m = _CODE_FENCE.search(text)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    if any(ind in stripped for ind in _PY_INDICATORS):
        return stripped
    return ""


def _strip_leading_header(text: str) -> str:
    return _LEADING_HEADER_RE.sub("", text, count=1).strip()


def _parse_metric(stdout: str, name: str) -> Optional[float]:
    for m in _METRIC_RE.finditer(stdout):
        if m.group(1) == name:
            try:
                return float(m.group(2))
            except ValueError:
                return None
    return None


def _gate_ok(stdout: str) -> bool:
    if _GATE_FAIL_RE.search(stdout):
        return False
    if "Process exited with code" in stdout:
        return False
    return True


# --- Fallback baseline experiment ---------------------------------------
# A walk is a tuple of D option-indices in [0, K). Ground-truth validity: every
# adjacent option pair is compatible, |opt_i - opt_{i+1}| <= TOL. FoO's
# consistency-checker is meant to prune invalid walks; we measure how often it
# wrongly flags a *valid* walk (false positive). The baseline checker has an
# off-by-one bug (uses >= TOL instead of > TOL), so it over-flags exactly-at-
# tolerance transitions. A correctness gate requires the checker still catch a
# meaningful fraction of truly-invalid walks (recall >= 0.5), which blocks the
# degenerate "flag nothing => FP=0" reward hack.

FALLBACK_BASELINE = '''"""Baseline: false-positive rate of a prefix-pattern consistency checker on
simulated Flow-of-Options walks (Nair, Trase, Kim, arxiv 2502.12929).

Measures FoO limitation #5 (module-specific issues: consistency-checker false
positives). A walk of D option-indices in [0,K) is ground-truth-valid iff every
adjacent pair satisfies |a-b| <= TOL. The checker under test should prune
invalid walks; consistency_fp_rate = (valid walks it wrongly flags) / (valid
walks). Lower is better.
"""
import random

random.seed(0)

K = 5      # options per depth
D = 6      # primary walk length
N = 4000   # walks sampled per length
TOL = 2    # ground-truth compatibility tolerance
LENGTHS = [4, 6, 8]   # sweep over walk lengths


def is_valid(walk):
    return all(abs(walk[i] - walk[i + 1]) <= TOL for i in range(len(walk) - 1))


def checker_flags_invalid(walk):
    # Baseline heuristic (off-by-one): flags any adjacent transition with
    # |a-b| >= TOL. This wrongly flags valid transitions where |a-b| == TOL.
    return any(abs(walk[i] - walk[i + 1]) >= TOL for i in range(len(walk) - 1))


def measure(d):
    # Independent, call-order-invariant RNG per length so a given D is
    # reproducible whether read as the primary metric or in the sweep.
    rng = random.Random(1000 + d)
    walks = [tuple(rng.randrange(K) for _ in range(d)) for _ in range(N)]
    valid = [w for w in walks if is_valid(w)]
    invalid = [w for w in walks if not is_valid(w)]
    fp = sum(1 for w in valid if checker_flags_invalid(w))
    tp = sum(1 for w in invalid if checker_flags_invalid(w))
    fp_rate = fp / len(valid) if valid else 0.0
    recall = tp / len(invalid) if invalid else 1.0
    return fp_rate, recall, len(valid), len(invalid)


# Correctness gate (runs BEFORE the metric).
fp_primary, recall_primary, n_valid, n_invalid = measure(D)
assert K > 0 and D > 0 and N > 0, "GATE: FAIL invalid params"
assert n_valid > 0, "GATE: FAIL no valid walks to measure FP against"
assert recall_primary >= 0.5, "GATE: FAIL checker recall too low (degenerate checker)"

print(f"CONFIG: K={K}, D={D}, N={N}, TOL={TOL}, seed=0")
print(f"INPUT: synthetic FoO DAG; primary D={D}: {n_valid} valid / {n_invalid} invalid walks")
for d in LENGTHS:
    fpr, rec, nv, ni = measure(d)
    print(f"SWEEP: D={d} fp_rate={fpr:.4f} recall={rec:.4f} valid={nv} invalid={ni}")
print(f"recall={recall_primary:.4f} (gate>=0.5)")
print(f"METRIC consistency_fp_rate={fp_primary:.4f}")
'''


# --- Autoresearch loop --------------------------------------------------

BASELINE_PROMPT = """Design a minimal, self-contained Python script that measures one stated limitation of {anchor}.

Limitation under study: "{limitation}"
Metric name: {metric_name} (direction: minimize)

The experiment: simulate Flow-of-Options walks (tuples of D option-indices in [0,K)). A walk is
ground-truth-VALID iff every adjacent option pair is compatible: |opt_i - opt_(i+1)| <= TOL.
A consistency-checker is supposed to prune INVALID walks. {metric_name} = (truly-VALID walks the
checker wrongly flags as invalid) / (truly-valid walks). Lower is better.

HARD CONSTRAINTS on the script:
1. Only stdlib (random) — do NOT import numpy or any third-party library.
2. random.seed(0) at the top. Use an INDEPENDENT, deterministic RNG per walk length (e.g.
   random.Random(1000+d)) so a given D yields identical numbers as the primary metric and in the
   sweep — do NOT draw all lengths from one shared stream (that makes the same D disagree by call order).
3. Print, in order: "CONFIG: ...", "INPUT: ...", then per-walk-length "SWEEP: D=<d> fp_rate=<f> recall=<f> ..." lines.
4. Correctness gate BEFORE the metric: assert K>0,D>0,N>0; assert there are valid walks; assert the
   checker's recall on truly-invalid walks is >= 0.5 (this blocks the "flag nothing" reward hack).
   On any failure the assert exits non-zero.
5. Print exactly ONE final line: "METRIC {metric_name}=<float:.4f>".
6. Finish in under 5 seconds.
7. Use K=5, D=6, N=4000, TOL=2 for the baseline, sweeping D over [4, 6, 8].

Output ONLY the Python code. No markdown fences, no commentary.
"""

PROPOSE_PROMPT = """You are iterating on a Python script in an autoresearch loop.

Goal: reduce {metric_name} (lower is better) — the false-positive rate of a Flow-of-Options
consistency-checker — WITHOUT breaking the recall gate (the checker must still catch >= 50% of
truly-invalid walks). The editable surface is the `checker_flags_invalid` heuristic.

Current best script.py:
```python
{best_code}
```

Current best {metric_name}: {best_metric:.4f}

Recent attempts (newest last):
{history_tail}

PRIOR MECHANISMS (do NOT repeat anything substantively equivalent):
{prior_mechanisms}

Propose ONE focused improvement to the checker heuristic that lowers false positives while keeping
recall >= 0.5. Examples of valid, distinct moves:
  - Fix the off-by-one boundary (>= TOL vs > TOL) so exactly-at-tolerance transitions are not flagged
  - Require two consecutive suspicious transitions before flagging (reduce single-edge over-flagging)
  - Use a margin band: only flag transitions clearly beyond tolerance (|a-b| >= TOL+1)
  - Aggregate evidence across the walk (flag only if a fraction of transitions are suspicious)

Do NOT weaken the ground-truth is_valid definition. Do NOT remove the gate. Do NOT change the metric
definition. Keep stdlib-only, seeded, CONFIG/INPUT/SWEEP lines, single final METRIC line, under 5s.

Output format (exactly):

PLAN: <one sentence: the change AND why it is mechanistically distinct from prior attempts>

```python
<full new script.py here>
```
"""


def autoresearch_loop(k_attempts: int = 3) -> dict:
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    print("[autoresearch] designing baseline...")
    llm_baseline = _extract_code(_llm(BASELINE_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=LIMITATION, metric_name=METRIC_NAME,
    )))
    baseline_code = llm_baseline if llm_baseline.strip() else FALLBACK_BASELINE

    out = run_code(baseline_code, filename="script.py", working_dir=str(WORKING_DIR))
    m = _parse_metric(out, METRIC_NAME)
    ok = _gate_ok(out) and m is not None
    log.append({"id": 0, "kind": "baseline",
                "plan": "baseline (LLM-designed)" if llm_baseline.strip() else "baseline (fallback)",
                "metric": m, "ok": ok, "stdout_tail": out[-500:]})
    print(f"[autoresearch] baseline ok={ok} metric={m}")

    if not ok:
        print("[autoresearch] baseline failed, using fallback")
        out = run_code(FALLBACK_BASELINE, filename="script.py", working_dir=str(WORKING_DIR))
        m = _parse_metric(out, METRIC_NAME)
        ok = _gate_ok(out) and m is not None
        baseline_code = FALLBACK_BASELINE
        log.append({"id": 0, "kind": "baseline-fallback", "plan": "fallback baseline",
                    "metric": m, "ok": ok, "stdout_tail": out[-500:]})

    best = (m, baseline_code) if ok else None

    for i in range(1, k_attempts + 1):
        if best is None:
            print("[autoresearch] no working baseline, skipping attempts")
            break
        history_tail = "\n".join(
            f'{{"id": {e["id"]}, "plan": {e["plan"]!r}, "metric": {e["metric"]}, '
            f'"ok": {e["ok"]}, "kept": {e.get("kept", "—")!r}}}'
            for e in log[-3:]
        )
        prior_mechanisms = "\n".join(
            f"- (attempt {e['id']}, metric={e['metric']}) {e['plan']}"
            for e in log if e.get("plan") not in (None, "")
        ) or "(none)"
        print(f"[autoresearch] attempt {i}/{k_attempts}...")
        response = _llm(PROPOSE_PROMPT.format(
            metric_name=METRIC_NAME, best_code=best[1], best_metric=best[0],
            history_tail=history_tail, prior_mechanisms=prior_mechanisms,
        ))
        plan_line = next((l for l in response.splitlines() if l.strip().startswith("PLAN:")), "PLAN: <no plan>")
        plan = plan_line.split(":", 1)[1].strip()
        new_code = _extract_code(response)

        if not new_code or new_code == best[1]:
            log.append({"id": i, "kind": "attempt", "plan": plan, "metric": None,
                        "ok": False, "kept": False, "stdout_tail": "[no new code]"})
            continue

        out = run_code(new_code, filename="script.py", working_dir=str(WORKING_DIR))
        m = _parse_metric(out, METRIC_NAME)
        ok = _gate_ok(out) and m is not None
        keep = ok and m < best[0]
        log.append({"id": i, "kind": "attempt", "plan": plan, "metric": m,
                    "ok": ok, "kept": keep, "stdout_tail": out[-500:]})
        print(f"[autoresearch]   metric={m} ok={ok} keep={keep}")
        if keep:
            best = (m, new_code)
        else:
            run_code(best[1], filename="script.py", working_dir=str(WORKING_DIR))

    if best is not None:
        (WORKING_DIR / "script.py").write_text(best[1])

    return {"log": log, "best": best, "metric_name": METRIC_NAME}


# --- Paper composition --------------------------------------------------

INTRO_PROMPT = """Write the Introduction section of a research paper that extends {anchor}.

The paper investigates one of FoO's stated limitations: "{limitation}".

Required content (in order):
1. State the anchor paper explicitly with arxiv ID: "{anchor}".
2. Summarize FoO's central claim in one sentence (DAG of option-nodes with beam-sampled walks scored
   by a scalar metric, with a consistency-checker that prunes inconsistent walks and case-based reasoning).
3. Identify the specific limitation we address: the consistency-checker's tendency to produce FALSE
   POSITIVES — wrongly pruning valid walks. Use the phrase "consistency-checker false positives".
4. State our contribution in one sentence: an autoresearch-driven empirical study of {metric_name}
   (false-positive rate) across walk lengths, with iterative refinement of the checker heuristic.
5. Outline the paper structure (1 sentence).

Do NOT start with a header. Begin directly with prose. ~250 words. Markdown. Precise, restrained, no hype.
"""

METHODS_PROMPT = """Write the Methods section of a research paper.

Context:
- Anchor paper: {anchor}
- Limitation addressed: "{limitation}" (consistency-checker false positives)
- Approach: Karpathy-style autoresearch loop on a single seeded Python script (cite: Karpathy 2026,
  "autoresearch"). Baseline + K=3 proposals. Single scalar metric ({metric_name}, lower is better).
  Correctness gate (checker recall >= 0.5) runs before the metric; failed gates trigger a revert.
- Discipline: section-by-section composition (cite: Lu et al. 2024, "The AI Scientist", arxiv 2408.06292).

CRITICAL: the EXACT script that produced the results is below. Any constant you cite (K, D, N, TOL,
recall threshold) MUST match it verbatim. Do NOT invent values.

```python
{best_script}
```

Required content (in order):
1. Walk + ground-truth-validity model (|opt_i - opt_(i+1)| <= TOL), citing exact K, D, N, TOL.
2. Definition of {metric_name} = (valid walks the checker wrongly flags) / (valid walks).
3. The recall gate and WHY it exists (blocks the degenerate "flag nothing => FP=0" reward hack).
4. The autoresearch loop: read best -> propose checker change -> run -> measure -> keep/revert.
5. Honest scope: we study the consistency-checker module in isolation on synthetic walks; we do not
   claim a general improvement to FoO.

Do NOT start with a header. Begin directly with prose. ~300 words. Markdown.
"""

RESULTS_PROMPT = """Write the Results section of a research paper.

EXPERIMENT LOG (newest last):
{log_text}

Best {metric_name}: {best_metric}

Required content:
1. Report baseline {metric_name} and best achieved {metric_name}. Both numbers MUST come from the log.
2. Report the improvement as absolute delta and percent change (computed from the log).
3. List each KEPT attempt with its plan and metric.
4. Acknowledge attempts that did NOT improve and what they tried.
5. Include a small markdown table summarizing all attempts (id, plan, metric, kept).
6. If the per-length SWEEP lines are present in the log tail, summarize how FP rate varies with walk
   length D. Use ONLY numbers that appear in the log. Do NOT invent numbers.

Do NOT start with a header. Begin directly with prose. ~250 words. Markdown.
"""

TITLE_PROMPT = """Propose a single short paper title for a study that extends "{anchor}" by addressing
"consistency-checker false positives" via an autoresearch loop measuring {metric_name}.

Requirements: mentions Flow-of-Options or FoO; mentions "consistency checker" or "false positive";
under 14 words; no quotes, no subtitle. Output the title only — single line.
"""

REFERENCES = """1. Nair, L., Trase, I., & Kim, M. (2025). Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*. arXiv:2502.12929. https://arxiv.org/abs/2502.12929

2. Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv:2408.06292. https://arxiv.org/abs/2408.06292

3. Karpathy, A. (2026). autoresearch: minimal agent loop for autonomous LLM experimentation. https://github.com/karpathy/autoresearch

4. Yamada, Y., et al. (2025). The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. arXiv:2504.08066.
"""


def _format_log_for_results(log: list[dict]) -> str:
    out = []
    for e in log:
        out.append(
            f'{{"id": {e["id"]}, "kind": "{e["kind"]}", "plan": {e.get("plan", "")!r}, '
            f'"metric": {e["metric"]}, "ok": {e["ok"]}, "kept": {e.get("kept", "—")}}}'
        )
        tail = e.get("stdout_tail", "")
        sweeps = [l for l in tail.splitlines() if l.startswith("SWEEP:")]
        if sweeps:
            out.extend(f"    {s}" for s in sweeps)
    return "\n".join(out)


def compose_paper(problem_domain: str, result: dict) -> Paper:
    log = result["log"]
    best_metric = result["best"][0] if result["best"] else None
    best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "N/A (baseline failed)"
    log_text = _format_log_for_results(log)

    print("[compose] writing intro...")
    intro = _strip_leading_header(_llm(INTRO_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=LIMITATION, metric_name=METRIC_NAME,
    ))) or f"This paper extends {FOO_ANCHOR} by studying consistency-checker false positives."

    print("[compose] writing methods...")
    best_script = result["best"][1] if result["best"] else "(no successful run — see appendix)"
    methods = _strip_leading_header(_llm(METHODS_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=LIMITATION, metric_name=METRIC_NAME, best_script=best_script,
    ))) or "We simulate FoO walks, apply a consistency-checker, and autoresearch its false-positive rate."

    print("[compose] writing results...")
    results = _strip_leading_header(_llm(RESULTS_PROMPT.format(
        log_text=log_text, metric_name=METRIC_NAME, best_metric=best_metric_str,
    ))) or f"Best {METRIC_NAME}: {best_metric_str}. See appendix for the experiment script."

    print("[compose] writing title...")
    title_raw = _llm(TITLE_PROMPT.format(anchor=FOO_ANCHOR, metric_name=METRIC_NAME)).strip()
    title = title_raw.strip('"').strip("'").splitlines()[0] if title_raw else \
        "Reducing Consistency-Checker False Positives in Flow-of-Options"

    best_code = result["best"][1] if result["best"] else ""
    appendix = f"# Code\n\n```python\n{best_code}\n```" if best_code else ""

    return Paper(
        title=title,
        introduction=intro,
        methods=methods,
        results=results,
        references=REFERENCES,
        appendix=appendix,
        tags=["flow-of-options", "autoresearch", "consistency-checker", "false-positives"],
    )


# --- run() entrypoint ---------------------------------------------------

def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    print(f"[run] problem_domain: {problem_domain!r}")
    print(f"[run] FoO anchor: {FOO_ANCHOR}")
    print(f"[run] limitation: {LIMITATION}")
    print(f"[run] metric: {METRIC_NAME} (minimize)")

    result = autoresearch_loop(k_attempts=3)
    print(f"[run] autoresearch done. log entries: {len(result['log'])}")
    if result["best"]:
        print(f"[run] best {METRIC_NAME}: {result['best'][0]:.4f}")
    else:
        print("[run] no successful experiment — paper will note this")

    paper = compose_paper(problem_domain, result)
    print(f"[run] paper drafted. title: {paper.title!r}")
    return paper

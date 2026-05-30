"""Paper-pushers sibling agent — FoO data-availability study.

Volume play addressing Flow-of-Options' stated limitation #1 (data
availability: the evaluator needs input data to score walks). Under a low-data
regime the evaluator's per-walk quality estimates are noisy, so FoO selects the
wrong walk; we measure selection accuracy vs an oracle as n_train shrinks, and
autoresearch a selection rule that copes with scarce, heteroscedastic data.

Same Karpathy-style autoresearch discipline as my_run_agent.py: a single seeded
script.py, scalar METRIC on stdout, a correctness gate separate from the metric,
revert-writes-best so the appendix shows the winning experiment. Self-contained
for the experiment loop, but pulls gather_literature()/_lit_block/
_augment_references/_results_table_md/archive_paper from the main agent so the
paper is literature-grounded and locally archived like the rest of the fleet.

Contract: run(problem_domain, papers_dir=None) -> Paper.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

from hackathon_science import Paper
from hackathon_science.tools import run_code

# Provider-routed LLM helper (Anthropic / OpenAI / Bedrock). See llm.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat as _chat, STRONG_MODEL  # noqa: E402

# Shared helpers from the main agent: literature grounding + persistent archive.
from my_run_agent import (  # noqa: E402
    gather_literature,
    _lit_block,
    _augment_references,
    _results_table_md,
    archive_paper,
)


# --- Configuration ------------------------------------------------------

MODEL = STRONG_MODEL   # provider-routed; override via LLM_STRONG_MODEL env (see llm.py)
WORKING_DIR = Path(__file__).parent / "files_data_avail"

FOO_ANCHOR = "Flow-of-Options (Nair, Trase, Kim, ICML 2025, arxiv 2502.12929)"
FOO_LIMITATIONS = [
    "Metric dependency — assumes a quantifiable evaluator exists",
    "Data availability — needs input data / datasets",
    "Residual method bias — still biased toward Random Forest despite diversity",
    "Walk-sampling inefficiency — naive sampling produces repeats",
    "Module-specific issues — consistency-checker false positives, retriever mis-selection",
]
LIMITATION = FOO_LIMITATIONS[1]
METRIC_NAME = "low_data_walk_accuracy"
MINIMIZE = False   # accuracy of the chosen walk vs oracle — higher is better

# Seed for the synthetic ground-truth world (true means + per-walk noise scales).
# Override via DATA_SEED to test whether the null result is seed-robust. Note:
# this hard-pins the fallback baseline; the LLM-designed baseline receives the
# seed via the prompt and usually honors it, but is only soft-pinned (it may
# still vary other free choices).
DATA_SEED = int(os.environ.get("DATA_SEED", "7"))


# --- LLM helper ---------------------------------------------------------

def _llm(user: str, system: str = "", model: str = MODEL, max_tokens: int = 4000,
         retry_on_empty: bool = True) -> str:
    """Provider-routed call (see llm.chat). Returns text or '' on error.

    No temperature sent — llm.chat only forwards temperature when > 0, and
    Opus 4.7 on Bedrock rejects it outright.
    """
    for _ in range(2 if retry_on_empty else 1):
        text = _chat(user, system=system, model=model, max_tokens=max_tokens)
        if text.strip():
            return text
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


def _better(candidate: float, incumbent: float) -> bool:
    return candidate < incumbent if MINIMIZE else candidate > incumbent


# --- Fallback baseline experiment ---------------------------------------
# W candidate walks have fixed true mean qualities and heteroscedastic noise
# scales. The oracle best = argmax true mean. The evaluator sees only n_train
# samples per walk (drawn from each walk's Normal(mean, sigma)) and selects a
# walk from them. low_data_walk_accuracy = fraction of seeded trials whose
# selected walk equals the oracle (higher = better). The correctness gate
# requires accuracy at the largest n_train >= 0.90 — a degenerate / data-
# ignoring selector scores ~1/W there and fails, so the gate is load-bearing.

FALLBACK_BASELINE = '''"""Baseline: Flow-of-Options walk-selection accuracy under limited data
(Nair, Trase, Kim, arxiv 2502.12929).

Measures FoO limitation #1 (data availability): the evaluator needs data to
score walks. W candidate walks have fixed true means and heteroscedastic noise
scales; the oracle best = argmax true mean. The evaluator sees only n_train
samples per walk and selects from them. low_data_walk_accuracy = fraction of
seeded trials whose selected walk equals the oracle. Higher is better.
"""
import random

random.seed(0)

W = 12                          # candidate walks
N_TRAINS = [10, 50, 200, 1000]  # samples-per-walk sweep
PRIMARY_NTRAIN = 50             # n_train for the headline metric
TRIALS = 300                    # seeded selection trials per n_train


def world():
    # Fixed ground truth: true means + heteroscedastic per-walk noise scales.
    rng = random.Random(7)
    means = [rng.uniform(0.0, 1.0) for _ in range(W)]
    sigmas = [rng.uniform(0.5, 2.0) for _ in range(W)]
    oracle = max(range(W), key=lambda i: means[i])
    return means, sigmas, oracle


def samples(mean, sigma, n, rng):
    return [rng.gauss(mean, sigma) for _ in range(n)]


def select(means, sigmas, n_train, rng):
    # Baseline selector: argmax of the empirical mean (ignores estimate noise).
    est = []
    for i in range(len(means)):
        xs = samples(means[i], sigmas[i], n_train, rng)
        est.append(sum(xs) / len(xs))
    return max(range(len(means)), key=lambda i: est[i])


def accuracy(n_train):
    means, sigmas, oracle = world()
    rng = random.Random(3000 + n_train)
    hits = sum(1 for _ in range(TRIALS) if select(means, sigmas, n_train, rng) == oracle)
    return hits / TRIALS


# Correctness gate (runs BEFORE the metric).
assert PRIMARY_NTRAIN >= 5, "GATE: FAIL n_train must be >= 5"
acc_full = accuracy(max(N_TRAINS))
# anti-hack: given abundant data the selector must actually find the oracle.
# A degenerate / data-ignoring selector scores ~1/W here and fails this gate.
assert acc_full >= 0.90, "GATE: FAIL selector fails even with abundant data (not using the data)"
acc_primary = accuracy(PRIMARY_NTRAIN)

means, sigmas, oracle = world()
print(f"CONFIG: W={W}, primary_n_train={PRIMARY_NTRAIN}, trials={TRIALS}, seed=0")
print(f"INPUT: {W} walks, heteroscedastic sigma in [0.5,2.0]; oracle walk={oracle} mean={means[oracle]:.4f}")
for nt in N_TRAINS:
    a = accuracy(nt)
    print(f"SWEEP: n_train={nt} low_data_walk_accuracy={a:.4f}")
print(f"acc@maxdata={acc_full:.4f} (gate>=0.90)")
print(f"METRIC low_data_walk_accuracy={acc_primary:.4f}")
'''


# --- Autoresearch loop --------------------------------------------------

BASELINE_PROMPT = """Design a minimal, self-contained Python script that measures one stated limitation of {anchor}.

Limitation under study: "{limitation}"
Metric name: {metric_name} (direction: MAXIMIZE — higher accuracy is better)

The experiment: W candidate Flow-of-Options walks have FIXED true mean qualities and HETEROSCEDASTIC
per-walk noise scales. The oracle best walk = argmax true mean. The evaluator sees only n_train samples
per walk (drawn from each walk's Normal(mean, sigma)) and selects a walk from them. {metric_name} =
fraction of seeded trials whose selected walk equals the oracle. Higher is better.

HARD CONSTRAINTS on the script (write clean, reviewable code — a reviewer will grade Code Tech Quality):
1. Only stdlib (random) — do NOT import numpy or any third-party library.
2. Module docstring at the top, then random.seed(0). Use typed helper functions (world, samples, select,
   accuracy) and a `def main()` style flow; no top-level spaghetti.
3. Seed the ground-truth world (true means + per-walk noise scales) with random.Random({data_seed}) so
   the underlying problem is reproducible. Use a SEPARATE, INDEPENDENT, deterministic RNG per n_train
   (e.g. random.Random(3000+n_train)) so a given n_train yields identical numbers as the primary metric
   and in the sweep.
4. Print, in order: "CONFIG: ...", "INPUT: ...", then per-n_train "SWEEP: n_train=<d> {metric_name}=<f>" lines.
5. Correctness gate BEFORE the metric: assert PRIMARY_NTRAIN >= 5; assert accuracy at the LARGEST n_train
   >= 0.90 (a degenerate / data-ignoring selector scores ~1/W and fails this — that is the point).
   On any failure the assert exits non-zero.
6. Print exactly ONE final line: "METRIC {metric_name}=<float:.4f>".
7. Finish in under 5 seconds.
8. Use W=12, PRIMARY_NTRAIN=50, TRIALS=300 for the baseline, sweeping n_train over [10, 50, 200, 1000].

Output ONLY the Python code. No markdown fences, no commentary.
"""

PROPOSE_PROMPT = """You are iterating on a Python script in an autoresearch loop.

Goal: INCREASE {metric_name} (higher accuracy is better) — the fraction of trials where the selected
walk equals the oracle best — at the SAME limited n_train, WITHOUT breaking the gate (accuracy at the
largest n_train must stay >= 0.90). The editable surface is the SELECTION RULE (how you turn the
n_train samples per walk into a chosen walk). You may NOT change the ground-truth world (means, sigmas),
n_train, the metric definition, or the gate.

Current best script.py:
```python
{best_code}
```

Current best {metric_name}: {best_metric:.4f}

Recent attempts (newest last):
{history_tail}

PRIOR MECHANISMS (do NOT repeat anything substantively equivalent):
{prior_mechanisms}

Propose ONE focused improvement to the selection rule that raises low-data accuracy while keeping
acc@maxdata >= 0.90. The noise is HETEROSCEDASTIC, so a walk with a flukishly high empirical mean from
a noisy few samples can beat the true best — exploit that. Examples of valid, distinct moves:
  - Uncertainty-penalized (lower-confidence-bound) selection: argmax of mean - c * std/sqrt(n_train)
  - Trim outlier samples per walk before averaging (robust mean) to resist heavy noise draws
  - Shrink each walk's estimate toward the grand mean by a per-walk factor that grows with its sample
    variance (empirical-Bayes; pulls in unreliable high-variance walks more than reliable ones)
  - Use the median per-walk instead of the mean to down-weight extreme draws

This is the honest scientific point about data availability: a noise-aware selector extracts more
signal from the same scarce data. Do NOT make the selector ignore the data. Keep stdlib-only, seeded,
CONFIG/INPUT/SWEEP lines, single final METRIC line, under 5s.

Output format (exactly):

PLAN: <one sentence: the change AND why it is mechanistically distinct from prior attempts>

```python
<full new script.py here>
```
"""


def autoresearch_loop(k_attempts: int = 3) -> dict:
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    # Honor DATA_SEED in the fallback too (the LLM baseline gets it via the prompt).
    fallback_code = FALLBACK_BASELINE.replace("random.Random(7)", f"random.Random({DATA_SEED})")

    print(f"[autoresearch] designing baseline... (world seed={DATA_SEED})")
    llm_baseline = _extract_code(_llm(BASELINE_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=LIMITATION, metric_name=METRIC_NAME, data_seed=DATA_SEED,
    )))
    baseline_code = llm_baseline if llm_baseline.strip() else fallback_code

    out = run_code(baseline_code, filename="script.py", working_dir=str(WORKING_DIR))
    m = _parse_metric(out, METRIC_NAME)
    ok = _gate_ok(out) and m is not None
    log.append({"id": 0, "kind": "baseline",
                "plan": "baseline (LLM-designed)" if llm_baseline.strip() else "baseline (fallback)",
                "metric": m, "ok": ok, "kept": ok, "stdout_tail": out[-500:]})
    print(f"[autoresearch] baseline ok={ok} metric={m}")

    if not ok:
        print("[autoresearch] baseline failed, using fallback")
        out = run_code(fallback_code, filename="script.py", working_dir=str(WORKING_DIR))
        m = _parse_metric(out, METRIC_NAME)
        ok = _gate_ok(out) and m is not None
        baseline_code = fallback_code
        log.append({"id": 0, "kind": "baseline-fallback", "plan": "fallback baseline",
                    "metric": m, "ok": ok, "kept": ok, "stdout_tail": out[-500:]})

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
        keep = ok and _better(m, best[0])
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

Retrieved literature you may cite as [L1], [L2], ... (use only those relevant; do not invent others):
{literature}

Required content (in order):
1. State the anchor paper explicitly with arxiv ID: "{anchor}".
2. Summarize FoO's central claim in one sentence (DAG of option-nodes with beam-sampled walks scored by
   a scalar metric, plus a consistency-checker and case-based reasoning).
3. Identify the specific limitation we address: FoO's evaluator needs DATA to score walks; under a
   LOW-DATA regime its per-walk estimates are noisy and it selects the wrong walk. Use the phrase
   "data availability".
4. State our contribution in one sentence: an autoresearch-driven empirical study of {metric_name}
   (accuracy of the selected walk vs an oracle as n_train shrinks) that TESTS whether noise-aware
   selection rules improve low-data accuracy over naive argmax. Do NOT assert any rule succeeded — the
   Results section reports the actual outcome (which may be a null result).
5. Outline the paper structure (1 sentence).

HARD CONSTRAINT: the study is a SINGLE synthetic simulation (fixed walks with true means + per-walk
Gaussian noise, an evaluator that sees n_train samples per walk). Do NOT claim any other experiments,
datasets, real tasks, or "LLM-judge benchmarks" — none exist. The structure sentence must reflect that
the evaluation is one synthetic simulation.

Do NOT start with a header. Begin directly with prose. ~250 words. Markdown. Precise, restrained, no hype.
"""

METHODS_PROMPT = """Write the Methods section of a research paper.

Context:
- Anchor paper: {anchor}
- Limitation addressed: "{limitation}" (data availability — low-data selection)
- Approach: Karpathy-style autoresearch loop on a single seeded Python script (cite: Karpathy 2026,
  "autoresearch"). Baseline + K=3 proposals. Single scalar metric ({metric_name}, HIGHER is better).
  Correctness gate (accuracy at the largest n_train >= 0.90) runs before the metric; failed gates revert.
- Discipline: section-by-section composition (cite: Lu et al. 2024, "The AI Scientist", arxiv 2408.06292).

Retrieved literature you may cite as [L1], [L2], ...:
{literature}

CRITICAL: the EXACT script that produced the results is below. Any constant you cite (W, n_train sweep,
TRIALS, accuracy threshold) MUST match it verbatim. Describe ONLY what the script implements — do not
invent methodology the code does not contain.

```python
{best_script}
```

Required content (in order):
1. Walk model: fixed true means + heteroscedastic per-walk noise; oracle = argmax true mean. The
   evaluator sees only n_train samples per walk. Cite exact W, TRIALS, and the n_train sweep.
2. Definition of {metric_name} = fraction of seeded trials whose selected walk equals the oracle.
3. The accuracy-at-abundant-data gate and WHY it exists (a degenerate / data-ignoring selector scores
   ~1/W and fails it).
4. The autoresearch loop: read best -> propose a selection-rule change -> run -> measure -> keep/revert.
5. Honest scope: synthetic walks with Gaussian per-walk noise in isolation; we do not claim a general
   improvement to FoO.

Do NOT start with a header. Begin directly with prose. ~300 words. Markdown.
"""

RESULTS_PROMPT = """Write the Results section of a research paper.

EXPERIMENT LOG (newest last):
{log_text}

Best {metric_name}: {best_metric} (higher is better)

Required content:
1. Report baseline {metric_name} and best achieved {metric_name}. Both numbers MUST come from the log.
2. Report the improvement as absolute delta and percent change (computed from the log).
3. List each KEPT attempt with its plan and metric.
4. Acknowledge attempts that did NOT improve and what they tried.
5. If the per-n_train SWEEP lines are present in the log tail, summarize how accuracy rises with
   n_train (the low-data deficit). Use ONLY numbers that appear in the log. Do NOT invent numbers.

Do NOT start with a header. Begin directly with prose. ~250 words. Markdown.
"""

TITLE_PROMPT = """Propose a single short paper title for a study that extends "{anchor}" by addressing
"data availability" (walk-selection accuracy under low data) via an autoresearch loop measuring
{metric_name}.

Requirements: mentions Flow-of-Options or FoO; mentions "low-data" or "data" or "scarce"; under 14
words; no quotes, no subtitle. Output the title only — single line.
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


def compose_paper(problem_domain: str, result: dict, literature: Optional[list[dict]] = None) -> Paper:
    log = result["log"]
    best_metric = result["best"][0] if result["best"] else None
    best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "N/A (baseline failed)"
    log_text = _format_log_for_results(log)
    literature = literature or []
    lit_text = _lit_block(literature)

    print("[compose] writing intro...")
    intro = _strip_leading_header(_llm(INTRO_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=LIMITATION, metric_name=METRIC_NAME, literature=lit_text,
    ))) or f"This paper extends {FOO_ANCHOR} by studying walk-selection accuracy under limited data."

    print("[compose] writing methods...")
    best_script = result["best"][1] if result["best"] else "(no successful run — see appendix)"
    methods = _strip_leading_header(_llm(METHODS_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=LIMITATION, metric_name=METRIC_NAME,
        best_script=best_script, literature=lit_text,
    ))) or "We simulate FoO walks scored from limited data and autoresearch a noise-aware selection rule."

    print("[compose] writing results...")
    results_prose = _strip_leading_header(_llm(RESULTS_PROMPT.format(
        log_text=log_text, metric_name=METRIC_NAME, best_metric=best_metric_str,
    ))) or f"Best {METRIC_NAME}: {best_metric_str}. See appendix for the experiment script."
    results = f"{results_prose}\n\n{_results_table_md(log)}"

    print("[compose] writing title...")
    title_raw = _llm(TITLE_PROMPT.format(anchor=FOO_ANCHOR, metric_name=METRIC_NAME)).strip()
    title = title_raw.strip('"').strip("'").splitlines()[0] if title_raw else \
        "Data Availability in Flow-of-Options: Walk-Selection Accuracy Under Scarce Data"

    best_code = result["best"][1] if result["best"] else ""
    appendix = f"# Code\n\n```python\n{best_code}\n```" if best_code else ""

    return Paper(
        title=title,
        introduction=intro,
        methods=methods,
        results=results,
        references=_augment_references(REFERENCES, literature),
        appendix=appendix,
        tags=["flow-of-options", "autoresearch", "data-availability", "low-data"],
    )


# --- run() entrypoint ---------------------------------------------------

def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    print(f"[run] problem_domain: {problem_domain!r}")
    print(f"[run] FoO anchor: {FOO_ANCHOR}")
    print(f"[run] limitation: {LIMITATION}")
    print(f"[run] metric: {METRIC_NAME} ({'minimize' if MINIMIZE else 'maximize'})")

    result = autoresearch_loop(k_attempts=3)
    print(f"[run] autoresearch done. log entries: {len(result['log'])}")
    if result["best"]:
        print(f"[run] best {METRIC_NAME}: {result['best'][0]:.4f}")
    else:
        print("[run] no successful experiment — paper will note this")

    print("[run] gathering literature via arXiv + OpenAlex...")
    literature = gather_literature(max_total=8)

    paper = compose_paper(problem_domain, result, literature=literature)
    print(f"[run] paper drafted. title: {paper.title!r}")

    archive_path = archive_paper(paper)
    if archive_path:
        print(f"[run] archived to {archive_path}")

    return paper

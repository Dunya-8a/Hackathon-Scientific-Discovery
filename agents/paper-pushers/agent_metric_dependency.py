"""Paper-pushers sibling agent — FoO metric-dependency study.

Volume play addressing Flow-of-Options' stated limitation #0 (metric
dependency: FoO assumes a quantifiable evaluator exists). When that evaluator
is noisy, the argmax "best walk" becomes unstable; we measure how unstable, and
autoresearch a selection procedure that denoises it.

Same Karpathy-style autoresearch discipline as my_run_agent.py: a single seeded
script.py, scalar METRIC on stdout, a correctness gate separate from the metric,
revert-writes-best so the appendix shows the winning experiment. Self-contained
for the experiment loop, but pulls gather_literature()/_lit_block/
_augment_references/_results_table_md/archive_paper from the main agent so the
paper is literature-grounded and locally archived like the rest of the fleet.

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
WORKING_DIR = Path(__file__).parent / "files_metric_dep"

FOO_ANCHOR = "Flow-of-Options (Nair, Trase, Kim, ICML 2025, arxiv 2502.12929)"
FOO_LIMITATIONS = [
    "Metric dependency — assumes a quantifiable evaluator exists",
    "Data availability — needs input data / datasets",
    "Residual method bias — still biased toward Random Forest despite diversity",
    "Walk-sampling inefficiency — naive sampling produces repeats",
    "Module-specific issues — consistency-checker false positives, retriever mis-selection",
]
LIMITATION = FOO_LIMITATIONS[0]
METRIC_NAME = "best_walk_variance"
MINIMIZE = True   # lower variance of the selected walk's true quality is better


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
# W candidate walks have fixed latent qualities. The evaluator returns
# true_quality + Gaussian(0, noise_std). FoO selects argmax. Over N seeded
# replays the TRUE quality of the selected walk varies; best_walk_variance =
# Var over replays of that true quality (lower = noise-immune). A correctness
# gate requires the selection still tracks quality (mean selected / oracle >=
# 0.80), which blocks the degenerate "always pick a constant walk => variance 0
# but ignores the evaluator" reward hack.

FALLBACK_BASELINE = '''"""Baseline: variance of Flow-of-Options best-walk selection under a noisy
scalar evaluator (Nair, Trase, Kim, arxiv 2502.12929).

Measures FoO limitation #0 (metric dependency: FoO assumes a quantifiable
evaluator). W candidate walks have fixed latent qualities; the evaluator
returns true_quality + Gaussian(0, noise_std). FoO selects argmax. Over N
seeded replays the TRUE quality of the selected walk varies; best_walk_variance
= Var over replays of that true quality. Lower is better (0 = noise-immune).
"""
import random

random.seed(0)

W = 20                              # candidate walks
N = 50                              # seeded replays of the noisy evaluation
NOISE_STDS = [0.0, 0.05, 0.1, 0.2]  # evaluator noise sweep
PRIMARY_NOISE = 0.1                 # noise level for the headline metric
REPS = 1                            # noisy reads averaged per walk before argmax (baseline: 1)


def latent_qualities(w):
    rng = random.Random(7)
    return [rng.random() for _ in range(w)]


def evaluate(true_q, noise_std, rng, reps):
    # Average `reps` independent noisy reads (baseline reps=1 -> single read).
    return sum(true_q + rng.gauss(0.0, noise_std) for _ in range(reps)) / reps


def select_best(quals, noise_std, rng, reps):
    noisy = [evaluate(q, noise_std, rng, reps) for q in quals]
    return max(range(len(quals)), key=lambda i: noisy[i])


def _mean(xs):
    return sum(xs) / len(xs)


def _variance(xs):
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def measure(noise_std, reps=REPS):
    # Independent, deterministic RNG per noise level so a given noise_std yields
    # identical numbers as the primary metric and in the sweep.
    quals = latent_qualities(W)
    oracle = max(quals)
    rng = random.Random(2000 + int(round(noise_std * 1000)))
    chosen_true = [quals[select_best(quals, noise_std, rng, reps)] for _ in range(N)]
    var = _variance(chosen_true)
    quality = _mean(chosen_true) / oracle if oracle else 0.0
    return var, quality


# Correctness gate (runs BEFORE the metric).
assert PRIMARY_NOISE >= 0, "GATE: FAIL noise_std must be >= 0"
assert N >= 10, "GATE: FAIL need N >= 10 replays"
var_primary, quality_primary = measure(PRIMARY_NOISE)
# anti-hack: selection must still track quality. Blocks the degenerate
# 'always pick a constant walk' which yields variance 0 but ignores the
# evaluator (mean selected quality collapses toward the population mean).
assert quality_primary >= 0.80, "GATE: FAIL selection quality too low (selector ignoring metric)"

print(f"CONFIG: W={W}, N={N}, primary_noise={PRIMARY_NOISE}, reps={REPS}, seed=0")
print(f"INPUT: {W} candidate walks, fixed latent qualities; oracle best={max(latent_qualities(W)):.4f}")
for ns in NOISE_STDS:
    v, qy = measure(ns)
    print(f"SWEEP: noise_std={ns} best_walk_variance={v:.6f} sel_quality={qy:.4f}")
print(f"sel_quality={quality_primary:.4f} (gate>=0.80)")
print(f"METRIC best_walk_variance={var_primary:.6f}")
'''


# --- Autoresearch loop --------------------------------------------------

BASELINE_PROMPT = """Design a minimal, self-contained Python script that measures one stated limitation of {anchor}.

Limitation under study: "{limitation}"
Metric name: {metric_name} (direction: minimize)

The experiment: W candidate Flow-of-Options walks have FIXED latent qualities. A noisy evaluator returns
true_quality + Gaussian(0, noise_std). FoO selects the argmax walk. Over N seeded replays of the noisy
evaluation, the TRUE quality of the selected walk varies. {metric_name} = variance over replays of that
true quality. Lower is better (0 = the evaluator's noise never changes the choice).

HARD CONSTRAINTS on the script (write clean, reviewable code — a reviewer will grade Code Tech Quality):
1. Only stdlib (random) — do NOT import numpy or any third-party library.
2. Module docstring at the top, then random.seed(0). Use typed helper functions (latent_qualities,
   evaluate, select_best, measure) and a `def main()` style flow; no top-level spaghetti.
3. Use an INDEPENDENT, deterministic RNG per noise level (e.g. random.Random(2000+int(round(ns*1000))))
   so a given noise_std yields identical numbers as the primary metric and in the sweep.
4. Print, in order: "CONFIG: ...", "INPUT: ...", then per-noise "SWEEP: noise_std=<f> {metric_name}=<f> sel_quality=<f>" lines.
5. Correctness gate BEFORE the metric: assert PRIMARY_NOISE >= 0; assert N >= 10; assert the selection
   still tracks quality (mean true-quality of selected walk / oracle best >= 0.80). This blocks the
   degenerate "always pick a constant walk => variance 0 but ignores the evaluator" reward hack.
   On any failure the assert exits non-zero.
6. Print exactly ONE final line: "METRIC {metric_name}=<float:.6f>".
7. Finish in under 5 seconds.
8. Use W=20, N=50, PRIMARY_NOISE=0.1, REPS=1 for the baseline, sweeping noise_std over [0.0, 0.05, 0.1, 0.2].

Output ONLY the Python code. No markdown fences, no commentary.
"""

PROPOSE_PROMPT = """You are iterating on a Python script in an autoresearch loop.

Goal: reduce {metric_name} (lower is better) — the variance of the Flow-of-Options best-walk selection
under a noisy evaluator — WITHOUT breaking the quality gate (mean true-quality of the selected walk /
oracle best must stay >= 0.80). The editable surface is the EVALUATION / SELECTION procedure (how many
noisy reads you take per walk and how you aggregate them before argmax). You may NOT change the latent
qualities, the noise model, the metric definition, or the gate.

Current best script.py:
```python
{best_code}
```

Current best {metric_name}: {best_metric:.6f}

Recent attempts (newest last):
{history_tail}

PRIOR MECHANISMS (do NOT repeat anything substantively equivalent):
{prior_mechanisms}

Propose ONE focused improvement that lowers selection variance while keeping sel_quality >= 0.80.
Examples of valid, distinct moves:
  - Average REPS independent noisy reads per walk before argmax (repeated measurement denoises)
  - Use a robust aggregator across REPS reads (median instead of mean) to resist outlier draws
  - Two-stage screen-then-refine: cheap single read to shortlist top-k walks, then more reads on those
  - Best-of-window: re-evaluate the current argmax a few times and keep it only if it stays on top

Note that any of these costs more evaluator queries — that honest trade-off (more measurements buy
stability) IS the scientific point about metric dependency; it is NOT a reward hack. Do NOT make the
selector ignore the evaluator. Keep stdlib-only, seeded, CONFIG/INPUT/SWEEP lines, single final METRIC
line, under 5s.

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
                "metric": m, "ok": ok, "kept": ok, "stdout_tail": out[-500:]})
    print(f"[autoresearch] baseline ok={ok} metric={m}")

    if not ok:
        print("[autoresearch] baseline failed, using fallback")
        out = run_code(FALLBACK_BASELINE, filename="script.py", working_dir=str(WORKING_DIR))
        m = _parse_metric(out, METRIC_NAME)
        ok = _gate_ok(out) and m is not None
        baseline_code = FALLBACK_BASELINE
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
   a SCALAR metric, plus a consistency-checker and case-based reasoning).
3. Identify the specific limitation we address: FoO assumes a QUANTIFIABLE, reliable evaluator; under a
   NOISY evaluator the argmax "best walk" becomes unstable. Use the phrase "metric dependency".
4. State our contribution in one sentence: an autoresearch-driven empirical study of {metric_name}
   (variance of the selected walk's true quality across seeded replays of a noisy evaluator) that TESTS
   whether iteratively refining the selection procedure reduces it. Do NOT assert the refinement
   succeeded — the Results section reports the actual outcome (which may be a null result).
5. Outline the paper structure (1 sentence).

HARD CONSTRAINT: the study is a SINGLE synthetic simulation (a fixed set of walks with latent qualities
scored by a Gaussian-noise evaluator). Do NOT claim any other experiments, datasets, real tasks, or
"LLM-judge benchmarks" — none exist. The structure sentence must reflect that the evaluation is one
synthetic simulation.

Do NOT start with a header. Begin directly with prose. ~250 words. Markdown. Precise, restrained, no hype.
"""

METHODS_PROMPT = """Write the Methods section of a research paper.

Context:
- Anchor paper: {anchor}
- Limitation addressed: "{limitation}" (metric dependency — noisy evaluator)
- Approach: Karpathy-style autoresearch loop on a single seeded Python script (cite: Karpathy 2026,
  "autoresearch"). Baseline + K=3 proposals. Single scalar metric ({metric_name}, lower is better).
  Correctness gate (selection quality vs oracle >= 0.80) runs before the metric; failed gates revert.
- Discipline: section-by-section composition (cite: Lu et al. 2024, "The AI Scientist", arxiv 2408.06292).

Retrieved literature you may cite as [L1], [L2], ...:
{literature}

CRITICAL: the EXACT script that produced the results is below. Any constant you cite (W, N, noise levels,
quality threshold) MUST match it verbatim. Describe ONLY what the script implements — do not invent
methodology the code does not contain.

```python
{best_script}
```

Required content (in order):
1. Walk + latent-quality model and the noisy evaluator (true_quality + Gaussian(0, noise_std)), citing
   exact W, N, PRIMARY_NOISE, and the noise sweep.
2. Definition of {metric_name} = variance over N replays of the true quality of the selected walk.
3. The selection-quality gate and WHY it exists (blocks the degenerate "pick a constant walk => variance
   0 but ignores the evaluator" reward hack).
4. The autoresearch loop: read best -> propose a selection-procedure change -> run -> measure -> keep/revert.
5. Honest scope: synthetic walks and a Gaussian-noise evaluator in isolation; we do not claim a general
   improvement to FoO.

Do NOT start with a header. Begin directly with prose. ~300 words. Markdown.
"""

RESULTS_PROMPT = """Write the Results section of a research paper.

EXPERIMENT LOG (newest last):
{log_text}

Best {metric_name}: {best_metric} (lower is better)

Required content:
1. Report baseline {metric_name} and best achieved {metric_name}. Both numbers MUST come from the log.
2. Report the improvement as absolute delta and percent change (computed from the log).
3. List each KEPT attempt with its plan and metric.
4. Acknowledge attempts that did NOT improve and what they tried.
5. If the per-noise SWEEP lines are present in the log tail, summarize how the variance grows with
   noise_std (and that it is 0 at noise_std=0). Use ONLY numbers that appear in the log. Do NOT invent numbers.

Do NOT start with a header. Begin directly with prose. ~250 words. Markdown.
"""

TITLE_PROMPT = """Propose a single short paper title for a study that extends "{anchor}" by addressing
"metric dependency" (selection instability under a noisy evaluator) via an autoresearch loop measuring
{metric_name}.

Requirements: mentions Flow-of-Options or FoO; mentions "noisy" or "metric" or "evaluator"; under 14
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
    best_metric_str = f"{best_metric:.6f}" if best_metric is not None else "N/A (baseline failed)"
    log_text = _format_log_for_results(log)
    literature = literature or []
    lit_text = _lit_block(literature)

    print("[compose] writing intro...")
    intro = _strip_leading_header(_llm(INTRO_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=LIMITATION, metric_name=METRIC_NAME, literature=lit_text,
    ))) or f"This paper extends {FOO_ANCHOR} by studying metric dependency under a noisy evaluator."

    print("[compose] writing methods...")
    best_script = result["best"][1] if result["best"] else "(no successful run — see appendix)"
    methods = _strip_leading_header(_llm(METHODS_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=LIMITATION, metric_name=METRIC_NAME,
        best_script=best_script, literature=lit_text,
    ))) or "We simulate FoO walks scored by a noisy evaluator and autoresearch the selection procedure's stability."

    print("[compose] writing results...")
    results_prose = _strip_leading_header(_llm(RESULTS_PROMPT.format(
        log_text=log_text, metric_name=METRIC_NAME, best_metric=best_metric_str,
    ))) or f"Best {METRIC_NAME}: {best_metric_str}. See appendix for the experiment script."
    results = f"{results_prose}\n\n{_results_table_md(log)}"

    print("[compose] writing title...")
    title_raw = _llm(TITLE_PROMPT.format(anchor=FOO_ANCHOR, metric_name=METRIC_NAME)).strip()
    title = title_raw.strip('"').strip("'").splitlines()[0] if title_raw else \
        "Metric Dependency in Flow-of-Options: Selection Stability Under a Noisy Evaluator"

    best_code = result["best"][1] if result["best"] else ""
    appendix = f"# Code\n\n```python\n{best_code}\n```" if best_code else ""

    return Paper(
        title=title,
        introduction=intro,
        methods=methods,
        results=results,
        references=_augment_references(REFERENCES, literature),
        appendix=appendix,
        tags=["flow-of-options", "autoresearch", "metric-dependency", "noisy-evaluator"],
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
        print(f"[run] best {METRIC_NAME}: {result['best'][0]:.6f}")
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

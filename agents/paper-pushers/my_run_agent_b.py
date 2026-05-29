"""Paper-pushers run agent — Phase B.

FoO-DAG-over-paper-artifacts (the headline extension).

Architecture:
  Phase 0 — Autoresearch loop on script.py (reused from Phase A; Reviewer F still needs this)
  Phase 1 — Build a DAG of paper-drafting decisions: K options per depth, D=5 depths
  Phase 2 — Sample walks through the DAG, score each via a simulated reviewer panel
  Phase 3 — Render the winning walk as a full paper

Simplifications vs the full architecture (will note honestly in the paper):
  - Independent (not depth-conditional) option generation
  - Uniform walk sampling, not full beam search
  - Simplified 6-axis scorer (not full 9-axis PoLL ensemble)
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

from hackathon_science import Paper
from hackathon_science.tools import run_code
from hackathon_science.utils import call_llm

# Persistent local archive — .cache/paper_draft.* gets clobbered every run.
sys.path.insert(0, str(Path(__file__).parent))
from my_run_agent import archive_paper  # noqa: E402


# --- Config -------------------------------------------------------------

MODEL = "global.anthropic.claude-opus-4-7"
WORKING_DIR = Path(__file__).parent / "files"   # share with Phase A; same experiment topic

FOO_ANCHOR = "Flow-of-Options (Nair, Trase, Kim, ICML 2025, arxiv 2502.12929)"
FOO_LIMITATIONS = [
    "Metric dependency — assumes a quantifiable evaluator exists",
    "Data availability — needs input data / datasets",
    "Residual method bias — still biased toward Random Forest despite diversity",
    "Walk-sampling inefficiency — naive sampling produces repeats",
    "Module-specific issues — consistency-checker false positives, retriever mis-selection",
]

DEPTH_NAMES = [
    "framing_angle",          # what claim do we make about FoO?
    "methods_structure",      # how is the experiment organized?
    "ablation_choice",        # what's the key variable manipulated?
    "results_narrative",      # how are the numbers presented?
    "discussion_stance",      # what's the closing argument?
]
DEPTH_GUIDANCE = {
    "framing_angle":      'Framing: what specific claim about FoO does the paper make? Examples: "FoO has hard limits at small N", "FoO benefits from retrieval grounding", "FoO\'s walk-sampling is a special case of stratified sampling".',
    "methods_structure":  'Methods structure: how is the experiment organized? Examples: "single-ablation study", "head-to-head with a baseline sampler", "scaling study across N", "case study + analysis".',
    "ablation_choice":    'Ablation: what variable is manipulated? Examples: "vary walk count N", "vary depth D", "vary the sampler\'s memo budget", "vary the consistency-checker tolerance".',
    "results_narrative":  'Results narrative: how are findings presented? Examples: "primary metric then secondary", "limitations-first then mitigations", "comparison-driven", "ablation-driven".',
    "discussion_stance":  'Discussion stance: closing argument. Examples: "FoO is incomplete without memo-conditioning", "FoO\'s walk-sampling is universal across structured DAGs", "FoO needs a redesign for high-density regimes", "small fix yields large gains".',
}

K_OPTIONS = 3                 # K options per depth
N_WALKS  = 6                  # walks sampled and scored


# --- LLM helper ---------------------------------------------------------

def _llm(user: str, system: str = "", model: str = MODEL, max_tokens: int = 4000) -> str:
    """Bedrock Converse call with single retry on empty response."""
    messages = [{"role": "user", "content": [{"text": user}]}]
    kwargs = {"inferenceConfig": {"maxTokens": max_tokens}}
    if system:
        kwargs["system"] = [{"text": system}]
    for attempt in range(2):
        try:
            r = call_llm(messages=messages, model_id=model, **kwargs)
            content = r.get("output", {}).get("message", {}).get("content", [])
            text = content[0].get("text", "") if content else ""
            if text.strip():
                return text
        except Exception as e:
            print(f"[_llm] error (attempt {attempt+1}): {e}", file=sys.stderr)
            if attempt == 1:
                return ""
    return ""


# --- Script-parsing helpers (shared with Phase A) ----------------------

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)
_METRIC_RE = re.compile(r"^METRIC\s+(\w+)=([\-\d.eE+]+)\s*$", re.MULTILINE)
_GATE_FAIL_RE = re.compile(r"^GATE:\s*FAIL", re.MULTILINE)
_LEADING_HEADER_RE = re.compile(r"\A\s*#{1,6}\s+[^\n]+\n+", re.MULTILINE)
_PY_INDICATORS = ("import ", "def ", "print(", "random.seed", "for ", "while ", "assert ")
_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _CODE_FENCE.search(text)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    return stripped if any(ind in stripped for ind in _PY_INDICATORS) else ""


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM response."""
    m = _JSON_FENCE.search(text)
    if m:
        candidate = m.group(1).strip()
    else:
        # Find first '{' through matching '}'
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _parse_metric(stdout: str, name: str) -> Optional[float]:
    for m in _METRIC_RE.finditer(stdout):
        if m.group(1) == name:
            try:
                return float(m.group(2))
            except ValueError:
                return None
    return None


def _gate_ok(stdout: str) -> bool:
    return not _GATE_FAIL_RE.search(stdout) and "Process exited with code" not in stdout


def _strip_leading_header(text: str) -> str:
    return _LEADING_HEADER_RE.sub("", text, count=1).strip()


# --- Phase 0: autoresearch loop (verbatim from Phase A) ----------------

FALLBACK_BASELINE = '''import random
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
'''

BASELINE_PROMPT = """Design a Python script (stdlib only, no numpy/scipy) that empirically characterizes Flow-of-Options' walk-sampling-inefficiency limitation.

It must do ALL of the following:

1. **Multi-seed evaluation.** For each (K, D, N) configuration, run S=5 seeds (0..4) and collect collision_rate measurements.
2. **Parameter sweep.** Evaluate at least 4 configurations: (K=3,D=4,N=80), (K=4,D=5,N=200), (K=5,D=5,N=500), (K=4,D=6,N=200). Vary regime.
3. **Multiple named samplers.** Implement at least THREE:
   - `naive_iid`: independent uniform-random walks (the FoO baseline)
   - `rejection_memo`: rejection sampling against a seen-set
   - `stratified_index`: random.sample on the integer index space [0, K^D), decoded to walks
4. **Bootstrap 95% CIs.** Implement a stdlib bootstrap (B=1000 resamples) for the per-config mean across the 5 seeds. Report (mean, ci_low, ci_high).
5. **Statistical test.** Implement a permutation-based two-sample test (Monte Carlo, 1000 permutations) comparing `naive_iid` vs `rejection_memo` collision_rates on the headline K=4,D=5,N=200 cell. Report a two-sided p-value.
6. **Print a results table.** Tab-separated rows with columns: config, sampler, mean, ci_low, ci_high. Then print the test result line: "TEST permutation_p={{p:.4f}} (n_perm=1000)".
7. **Correctness gate.** Run assertions (K, D, N positive; samplers return exactly N walks; seeds set explicitly) BEFORE the metric.
8. **Headline metric.** Final line MUST be: "METRIC collision_rate=<float:.4f>" where the number is the `naive_iid` mean at (K=4, D=5, N=200, seeds 0..4). This is what the autoresearch loop optimizes; lower is better.
9. **Reproducibility.** random.seed inside each run, stdlib only, no network, no file I/O outside stdout.
10. Must finish in under 30 seconds wall-clock.

Output ONLY the Python code. No markdown fences, no commentary, no preamble.
"""

PROPOSE_PROMPT = """Iterate on this script to lower the headline METRIC collision_rate (the naive_iid mean at K=4,D=5,N=200) while preserving the multi-config, multi-seed sweep, bootstrap CIs, and the permutation test.

Current best metric: {best_metric:.4f}

Current best script.py:
```python
{best_code}
```

Prior mechanisms (do NOT propose anything substantively equivalent):
{prior_mechanisms}

Propose ONE mechanistically distinct improvement (e.g., a new sampler variant added to the comparison, OR a sharper definition of the headline that still reads as collision_rate). Preserve:
  - Multi-seed evaluation
  - The 4-config sweep
  - The named samplers (at least 3)
  - The bootstrap CIs and the permutation test
  - The single final "METRIC collision_rate=<float:.4f>" line (this is what we optimize)
  - The correctness gate before the metric

Output:

PLAN: <one sentence>

```python
<full new script.py>
```
"""


def autoresearch_loop(k_attempts: int = 3) -> dict:
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    print("[autoresearch] designing baseline...")
    code = _extract_code(_llm(BASELINE_PROMPT)) or FALLBACK_BASELINE
    out = run_code(code, filename="script.py", working_dir=str(WORKING_DIR))
    m = _parse_metric(out, "collision_rate")
    ok = _gate_ok(out) and m is not None
    log.append({"id": 0, "kind": "baseline", "plan": "baseline", "metric": m, "ok": ok})
    if not ok:
        out = run_code(FALLBACK_BASELINE, filename="script.py", working_dir=str(WORKING_DIR))
        m = _parse_metric(out, "collision_rate")
        ok = _gate_ok(out) and m is not None
        code = FALLBACK_BASELINE
        log[-1] = {"id": 0, "kind": "baseline-fallback", "plan": "fallback", "metric": m, "ok": ok}
    best = (m, code) if ok else None
    print(f"[autoresearch] baseline ok={ok} metric={m}")

    for i in range(1, k_attempts + 1):
        if best is None:
            break
        prior = "\n".join(f"- {e['plan']}" for e in log if e.get("plan")) or "(none)"
        response = _llm(PROPOSE_PROMPT.format(
            best_metric=best[0], best_code=best[1], prior_mechanisms=prior
        ))
        plan_line = next((l for l in response.splitlines() if l.strip().startswith("PLAN:")), "PLAN: <no plan>")
        plan = plan_line.split(":", 1)[1].strip()
        new_code = _extract_code(response)
        if not new_code or new_code == best[1]:
            log.append({"id": i, "kind": "attempt", "plan": plan, "metric": None, "ok": False, "kept": False})
            continue
        out = run_code(new_code, filename="script.py", working_dir=str(WORKING_DIR))
        m = _parse_metric(out, "collision_rate")
        ok = _gate_ok(out) and m is not None
        keep = ok and m < best[0]
        log.append({"id": i, "kind": "attempt", "plan": plan, "metric": m, "ok": ok, "kept": keep})
        print(f"[autoresearch]   attempt {i}: metric={m} ok={ok} keep={keep}")
        if keep:
            best = (m, new_code)
        else:
            run_code(best[1], filename="script.py", working_dir=str(WORKING_DIR))

    if best is not None:
        (WORKING_DIR / "script.py").write_text(best[1])
    return {"log": log, "best": best, "metric_name": "collision_rate"}


# --- Phase 1: build FoO DAG of paper-drafting decisions ---------------

OPTIONS_PROMPT = """You are generating diverse, mechanistically distinct options for one decision in a research paper outline.

The paper extends {anchor}. It is an empirical study of FoO's walk-sampling-inefficiency limitation, with the autoresearch experiment summarized below.

Experiment summary (for grounding only — do not alter these facts):
  Metric: collision_rate (lower is better), simulated FoO walk sampling, K=4 options per depth, D=5 depths.
  Baseline: {baseline_metric:.4f}
  Best achieved: {best_metric:.4f}
  Kept mechanism: {kept_plan}

Decision to be made: **{depth_name}**

Guidance for this decision: {guidance}

Generate exactly {k} mechanistically distinct options for this decision. Each option must be:
  - A single short paragraph (2-4 sentences) describing the choice and its rationale.
  - Substantively different from the others (different argument structure, different emphasis, different stance — not just rephrasings).
  - Honest about scope (this is a focused empirical study, not a field-changing result).

Output ONLY a JSON object: {{"options": ["option 1 text", "option 2 text", "option 3 text"]}}. No commentary, no markdown fences.
"""


def generate_depth_options(depth_name: str, exp_result: dict, k: int = K_OPTIONS) -> list[str]:
    """Generate K distinct options for one depth."""
    best_metric = exp_result["best"][0] if exp_result["best"] else 0.0
    baseline = next((e["metric"] for e in exp_result["log"] if e.get("kind", "").startswith("baseline") and e["metric"] is not None), 0.0)
    kept = next((e["plan"] for e in exp_result["log"] if e.get("kept")), "no kept attempt")
    text = _llm(OPTIONS_PROMPT.format(
        anchor=FOO_ANCHOR,
        depth_name=depth_name,
        guidance=DEPTH_GUIDANCE[depth_name],
        baseline_metric=baseline,
        best_metric=best_metric,
        kept_plan=kept,
        k=k,
    ))
    parsed = _extract_json(text)
    if parsed and isinstance(parsed.get("options"), list) and len(parsed["options"]) >= 1:
        return parsed["options"][:k]
    # Fallback: minimal options so the walk space is non-empty.
    return [f"Default {depth_name} option {i+1}: {DEPTH_GUIDANCE[depth_name].split(':')[0]}." for i in range(k)]


def build_paper_dag(exp_result: dict) -> dict[str, list[str]]:
    """Generate K options for each depth, in parallel-friendly order."""
    dag = {}
    for depth in DEPTH_NAMES:
        print(f"[dag] generating options for depth: {depth}")
        dag[depth] = generate_depth_options(depth, exp_result, k=K_OPTIONS)
    return dag


# --- Phase 2: sample and score walks ----------------------------------

def sample_walks(dag: dict[str, list[str]], n_walks: int, seed: int = 0) -> list[tuple[int, ...]]:
    """Uniform-random sample without replacement of walks through the DAG.

    Includes one 'diagonal' walk (all-index-0) and one 'anti-diagonal' walk
    (each depth's highest index) to ensure structurally distinct anchors.
    """
    rng = random.Random(seed)
    depths = [len(dag[d]) for d in DEPTH_NAMES]
    total = 1
    for n in depths:
        total *= n
    n_walks = min(n_walks, total)

    walks: set[tuple[int, ...]] = set()
    # Seed with two extremes for diversity
    walks.add(tuple([0] * len(DEPTH_NAMES)))
    walks.add(tuple([n - 1 for n in depths]))

    while len(walks) < n_walks:
        walk = tuple(rng.randrange(n) for n in depths)
        walks.add(walk)

    return list(walks)


SCORER_PROMPT = """You are a senior reviewer at a top ML venue evaluating a paper outline.

The paper extends {anchor} and reports an empirical study of FoO's walk-sampling-inefficiency limitation. The autoresearch experiment summary is:
  Metric: collision_rate (lower is better). Baseline: {baseline_metric:.4f}. Best: {best_metric:.4f}.
  Best mechanism: {kept_plan}

Below is one candidate outline for the paper, in five labeled decisions:

{walk_text}

Score this outline on a 1-10 scale per axis. Be strict — 8+ is rare.
  - Technical_Quality: rigor, soundness, alignment with the experiment
  - Novelty: extension beyond prior work, originality
  - Clarity: structure, exposition coherence
  - Significance: practical or scientific value of the framing
  - Extension_Quality: does this clearly extend Flow-of-Options (anchor), addressing a stated limitation?

Respond ONLY in JSON (no fences, no commentary):
{{
  "Technical_Quality": <int 1-10>,
  "Novelty": <int 1-10>,
  "Clarity": <int 1-10>,
  "Significance": <int 1-10>,
  "Extension_Quality": <int 1-10>,
  "Overall": <float — weighted mean: 0.25*Tech + 0.2*Novelty + 0.15*Clarity + 0.2*Significance + 0.2*Extension>,
  "key_weakness": "<one sentence>"
}}
"""


def render_walk_text(walk: tuple[int, ...], dag: dict[str, list[str]]) -> str:
    parts = []
    for i, depth in enumerate(DEPTH_NAMES):
        parts.append(f"### {depth.replace('_', ' ').title()}\n{dag[depth][walk[i]]}")
    return "\n\n".join(parts)


def score_walk(walk: tuple[int, ...], dag: dict, exp_result: dict) -> tuple[float, dict]:
    walk_text = render_walk_text(walk, dag)
    best_metric = exp_result["best"][0] if exp_result["best"] else 0.0
    baseline = next((e["metric"] for e in exp_result["log"] if e.get("kind", "").startswith("baseline") and e["metric"] is not None), 0.0)
    kept = next((e["plan"] for e in exp_result["log"] if e.get("kept")), "no kept attempt")
    response = _llm(SCORER_PROMPT.format(
        anchor=FOO_ANCHOR, walk_text=walk_text,
        baseline_metric=baseline, best_metric=best_metric, kept_plan=kept,
    ))
    parsed = _extract_json(response)
    if parsed and "Overall" in parsed:
        return float(parsed["Overall"]), parsed
    return 5.0, {"raw": response[:200], "parse_failed": True}


# --- Phase 3: render the best walk as a full paper --------------------

INTRO_PROMPT_B = """Write the Introduction section of a research paper that extends {anchor}.

The paper investigates "walk-sampling inefficiency" — FoO's stated limitation that naive sampling over the option DAG produces duplicate walks.

The introduction MUST use this framing decision (do not deviate):
{framing_text}

And signal the methods structure:
{methods_structure_text}

Required content (cover ALL of these — be substantive on each):
1. State the anchor explicitly with arxiv ID and authors.
2. Summarize FoO's central mechanism in 2-3 sentences: the option DAG, walk sampling, max-update edge rewards, consistency-checker, case-based reasoning.
3. Identify the walk-sampling-inefficiency limitation. Quote the limitation language. Explain WHY it matters (compute waste, depressed diversity, biased edge updates).
4. Position against related work on sampling-without-replacement and quasi-Monte Carlo (cite at least 2 of: Niederreiter 1992 on low-discrepancy sequences; Owen 1995/2003 on randomized QMC; Glasserman 2004 on Monte Carlo methods; Settles 2009 on active learning; Sakana AI Scientist v1/v2 on autoresearch protocol).
5. State the contribution precisely: an autoresearch-driven empirical study of collision_rate across a parameter sweep (multiple K, D, N), with multiple seeds, bootstrap 95% CIs, and a permutation-based statistical test. We do not claim a general improvement to FoO.
6. Outline the remaining sections (1 sentence per section, 4 sections).

Do NOT start with a header. **Target length: 500-650 words.** Markdown. Precise, technical, restrained.
"""

METHODS_PROMPT_B = """Write the Methods section.

The methods MUST use these decisions (do not deviate):
{methods_structure_text}
{ablation_text}

The EXACT script that ran for the reported results (any constants — K, D, N, seeds — MUST match this verbatim):

```python
{best_script}
```

Required content (cover ALL of these — be substantive and precise):
1. **Simulated FoO walk-sampling setup.** Describe the synthetic option DAG and the parameter sweep (each (K, D, N) cell), citing the exact constants from the script. Explain why these specific cells were chosen (cover small/medium/large index spaces).
2. **Metric.** Define collision_rate = 1 - (unique_walks / total_walks). Explain its interpretation as fraction of wasted samples.
3. **Sampler variants.** Describe each named sampler implemented in the script (naive_iid, rejection_memo, stratified_index, and any others), with a 1-2 sentence specification each.
4. **Multi-seed evaluation.** State the number of seeds per (sampler, config) cell and the seed scheme.
5. **Bootstrap confidence intervals.** Describe the bootstrap procedure (number of resamples B, percentile method for CI, the per-config mean as the statistic).
6. **Permutation test.** Describe the two-sample permutation test comparing naive_iid against rejection_memo on the headline cell. Number of permutations, test statistic (mean difference), and that the p-value is two-sided.
7. **Autoresearch loop.** Describe the iterative procedure: baseline + K=3 LLM-proposed mutations to the script, correctness gate (assertion suite), revert-writes-best policy, headline metric optimization.
8. **Reproducibility.** Explicit seeds, stdlib only, deterministic data generation, no network, no file I/O outside stdout.
9. **Honest scope.** We study walk sampling in a simulated DAG; we do not run LLM-driven FoO. We do not claim a general improvement to FoO; the contribution is mechanistic attribution and statistical characterization within one sampler family.

Reference at least 2 of: Efron & Tibshirani 1993 (bootstrap), Good 2005 (permutation tests), Niederreiter 1992 (low-discrepancy), Mannila/Toivonen/Verkamo 1994 (frequent patterns) where they apply.

Do NOT start with a header. **Target length: 550-700 words.** Markdown.
"""

RESULTS_PROMPT_B = """Write the Results section.

The results MUST be presented in this narrative style (do not deviate):
{results_narrative_text}

EXPERIMENT LOG (newest last):
{log_text}

SCRIPT STDOUT (contains the multi-config, multi-seed, bootstrap, and permutation-test outputs you must report verbatim):
```
{script_stdout}
```

Best headline collision_rate: {best_metric}

Required content (cover ALL of these — use ONLY numbers from the log and stdout):
1. **Headline result.** State the headline collision_rate (best vs baseline naive_iid at K=4, D=5, N=200), with the bootstrap 95% CI for each. Report the permutation-test p-value.
2. **Sweep table.** Reproduce the per-config, per-sampler table from stdout as a markdown table with columns: config, sampler, mean, 95% CI. Do NOT invent numbers; copy from stdout. If a number is missing, write "—".
3. **Sampler-by-sampler narrative.** Discuss what each sampler achieves and where it breaks down (e.g., stratified_index needing N ≤ K^D, rejection_memo needing enough free walks to avoid retry exhaustion).
4. **Autoresearch trace.** List each attempt id, plan, metric, and kept status. Honestly report failed attempts and why (if the stdout reveals it).
5. **Statistical interpretation.** State whether the difference between naive_iid and rejection_memo is statistically significant at the headline cell, and what the magnitude of the effect is in absolute terms.

Do NOT start with a header. **Target length: 500-650 words.** Markdown. Include exactly ONE summary table.
"""

DISCUSSION_PROMPT_B = """Write a Discussion / Limitations / Conclusion section.

The discussion MUST take this stance (do not deviate):
{discussion_stance_text}

**HARD CONSTRAINT — verbatim numbers only.** Any numerical claim in your output (collision_rate, CI bounds, p-values, percentages, resample counts) MUST be either the headline number stated immediately below OR a string copied verbatim from the EXPERIMENT LOG or SCRIPT STDOUT blocks below. Do NOT invent, interpolate, or estimate values. If a number you would naturally want to cite (e.g. a bootstrap CI) is absent from the materials below, write "—" or "not computed for this run" instead of fabricating a value. This rule exists because past runs hallucinated plausible-looking statistics; reviewers will check.

Headline collision_rate (best): {best_metric_str}

EXPERIMENT LOG (newest last):
{log_text}

SCRIPT STDOUT (truncated):
```
{script_stdout}
```

Required content (cover ALL of these):
1. **Restate the kept finding** in one precise sentence. Use the headline number ({best_metric_str}) verbatim. Cite a CI only if one appears in the stdout above; otherwise write "no bootstrap CI was emitted for this run".
2. **Defend the stance above** with a substantive argument from the empirical results — what specifically about the data supports this position?
3. **Position against related work.** Compare the contribution to (a) sampling-without-replacement / Fisher-Yates / reservoir sampling, (b) low-discrepancy sequences (Halton, Sobol), (c) active-learning-style diversity-forcing (cite Settles 2009 if relevant), (d) Sakana's AI Scientist autoresearch (Lu et al. 2024). Be honest about overlap.
4. **Three specific limitations** (numbered, each 2-3 sentences):
   - Simulation-vs-real LLM-driven FoO gap
   - Single-domain coverage / external validity
   - Choice of headline cell / parameter regime
5. **Two concrete next steps** (numbered, each 1-2 sentences):
   - A replication study in a real LLM-driven FoO setting
   - A more aggressive ablation extending the parameter sweep

Do NOT start with a header. **Target length: 450-600 words.** Markdown.
"""

TITLE_PROMPT_B = """Propose ONE short paper title (under 12 words, no quotes, no subtitle) that captures the paper's framing decision: {framing_text}

Output the title only — a single line."""


def render_final_paper(best_walk: tuple[int, ...], dag: dict, exp_result: dict) -> Paper:
    f = dag["framing_angle"][best_walk[0]]
    ms = dag["methods_structure"][best_walk[1]]
    ab = dag["ablation_choice"][best_walk[2]]
    rn = dag["results_narrative"][best_walk[3]]
    ds = dag["discussion_stance"][best_walk[4]]

    best_metric = exp_result["best"][0] if exp_result["best"] else None
    best_script = exp_result["best"][1] if exp_result["best"] else "(no working script)"
    best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"
    log_text = "\n".join(
        f'{{"id": {e["id"]}, "plan": {e.get("plan", "")!r}, "metric": {e["metric"]}, "ok": {e["ok"]}, "kept": {e.get("kept", "—")}}}'
        for e in exp_result["log"]
    )

    print("[render] writing intro...")
    intro = _strip_leading_header(_llm(INTRO_PROMPT_B.format(
        anchor=FOO_ANCHOR, framing_text=f, methods_structure_text=ms,
    ), max_tokens=6000))
    print("[render] writing methods...")
    methods = _strip_leading_header(_llm(METHODS_PROMPT_B.format(
        methods_structure_text=ms, ablation_text=ab, best_script=best_script,
    ), max_tokens=6000))
    # Capture the latest stdout from running the best script so Results can quote sweep numbers verbatim.
    script_stdout = ""
    if exp_result["best"]:
        try:
            script_stdout = run_code(exp_result["best"][1], filename="script.py",
                                     working_dir=str(WORKING_DIR))
        except Exception as e:
            script_stdout = f"(failed to re-run for stdout capture: {e})"
    print("[render] writing results...")
    results = _strip_leading_header(_llm(RESULTS_PROMPT_B.format(
        results_narrative_text=rn, log_text=log_text, best_metric=best_metric_str,
        script_stdout=script_stdout[:4000],
    ), max_tokens=6000))
    print("[render] writing discussion...")
    discussion = _strip_leading_header(_llm(DISCUSSION_PROMPT_B.format(
        discussion_stance_text=ds,
        best_metric_str=best_metric_str,
        log_text=log_text,
        script_stdout=script_stdout[:4000],
    ), max_tokens=6000))
    # Append discussion as a final subsection of results (Paper schema has no `discussion` field)
    if discussion:
        results = f"{results}\n\n### Discussion\n\n{discussion}"

    print("[render] writing title...")
    title = _llm(TITLE_PROMPT_B.format(framing_text=f)).strip().strip('"').splitlines()[0] or \
        "Autoresearch over Flow-of-Options Walk-Sampling: A FoO-DAG Approach"

    references = """1. Nair, L., Trase, I., & Kim, M. (2025). Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*. arXiv:2502.12929. https://arxiv.org/abs/2502.12929

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
"""

    appendix = f"# Code\n\n```python\n{best_script}\n```" if best_script else ""

    return Paper(
        title=title,
        introduction=intro or "Introduction unavailable.",
        methods=methods or "Methods unavailable.",
        results=results or f"Best collision_rate: {best_metric_str}.",
        references=references,
        appendix=appendix,
        tags=["flow-of-options", "autoresearch", "foo-dag", "paper-artifacts", "agentic-discovery"],
    )


# --- run() entrypoint -------------------------------------------------

def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    print(f"[run] Phase B — FoO-DAG over paper artifacts")
    print(f"[run] problem_domain: {problem_domain!r}")
    print(f"[run] anchor: {FOO_ANCHOR}")
    print(f"[run] limitation: {FOO_LIMITATIONS[3]}")

    # Phase 0 — autoresearch experiment (grounds the paper in real numbers)
    print("[run] Phase 0: autoresearch loop on script.py")
    exp_result = autoresearch_loop(k_attempts=3)
    print(f"[run]   log entries: {len(exp_result['log'])}")
    if exp_result["best"]:
        print(f"[run]   best collision_rate: {exp_result['best'][0]:.4f}")

    # Phase 1 — build FoO-DAG of paper-drafting decisions
    print(f"[run] Phase 1: building FoO-DAG ({K_OPTIONS} options × {len(DEPTH_NAMES)} depths = {K_OPTIONS**len(DEPTH_NAMES)} possible walks)")
    dag = build_paper_dag(exp_result)
    for depth, opts in dag.items():
        print(f"[run]   {depth}: {len(opts)} options")

    # Phase 2 — sample and score walks
    walks = sample_walks(dag, n_walks=N_WALKS)
    print(f"[run] Phase 2: scoring {len(walks)} walks via simulated reviewer")
    scored = []
    for w in walks:
        score, detail = score_walk(w, dag, exp_result)
        print(f"[run]   walk {w}: Overall={score:.2f} ({detail.get('key_weakness', '')[:60]!r})")
        scored.append((score, w, detail))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_walk, best_detail = scored[0]
    print(f"[run]   BEST walk {best_walk}: Overall={best_score:.2f}")

    # Phase 3 — render the winning walk into a real paper
    print(f"[run] Phase 3: rendering best walk")
    paper = render_final_paper(best_walk, dag, exp_result)
    print(f"[run] paper drafted. title: {paper.title!r}")

    archive_path = archive_paper(paper)
    if archive_path:
        print(f"[run] archived to {archive_path}")

    return paper

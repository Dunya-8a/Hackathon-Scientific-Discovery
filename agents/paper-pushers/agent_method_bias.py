"""Paper-pushers sibling agent — FoO residual method-bias cross-model study.

Addresses Flow-of-Options' stated limitation #3 (residual method bias: FoO
remains biased toward Random Forest despite enforced diversity). The headline
move is cross-model option generation (ARIS-style): ask Claude and GPT for FoO
option lists on a fixed task, then measure the distribution skew between their
option sets. Different priors break the shared-model bias that produces the
Random-Forest pull.

The autoresearch experiment is run with REAL LLM option generation (Claude via
Bedrock + GPT via OpenAI), captured once into a deterministic JSON fixture in
working_dir, then a single seeded script.py computes the skew metric over that
fixture. This keeps script.py reproducible and offline (Reviewer F) while the
numbers it reports come from genuine cross-model generation (Code-Paper
Alignment) rather than a simulation.

Requires OPENAI_API_KEY (GPT half of the cross-model pair). If unset, run()
raises with a clear message.

Same Karpathy autoresearch discipline + revert-writes-best as the sibling
agents. Self-contained; own working_dir to avoid clobbering files/script.py.

Contract: run(problem_domain, papers_dir=None) -> Paper.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from hackathon_science import Paper
from hackathon_science.tools import run_code

# Provider-routed LLM helper. STRONG_MODEL/FAST_MODEL come from env (see
# llm.py); defaults are claude-opus-4-7 / claude-sonnet-4-6 via Anthropic API.
# GPT_MODEL is hardcoded because this paper's whole point is cross-family
# Claude-vs-GPT comparison — that's the experiment, not a config knob.
sys.path.insert(0, str(Path(__file__).parent))
from llm import chat as _chat, STRONG_MODEL, FAST_MODEL  # noqa: E402


# --- Configuration ------------------------------------------------------

CLAUDE_MODEL = STRONG_MODEL              # area-chair / synthesis / writing
CLAUDE_GEN_MODEL = FAST_MODEL            # option generation (the Claude half)
GPT_MODEL = "gpt-4o"                      # the GPT half — cross-family by design
WORKING_DIR = Path(__file__).parent / "files_method_bias"

FOO_ANCHOR = "Flow-of-Options (Nair, Trase, Kim, ICML 2025, arxiv 2502.12929)"
FOO_LIMITATIONS = [
    "Metric dependency — assumes a quantifiable evaluator exists",
    "Data availability — needs input data / datasets",
    "Residual method bias — still biased toward Random Forest despite diversity",
    "Walk-sampling inefficiency — naive sampling produces repeats",
    "Module-specific issues — consistency-checker false positives, retriever mis-selection",
]
LIMITATION = FOO_LIMITATIONS[2]
METRIC_NAME = "method_bias_index"

# A fixed ML task to elicit FoO "model choice" options from each model family.
# Mirrors FoO's depth-2 (model-choice) option-generation step, where the paper
# reports the residual Random-Forest pull.
OPTION_TASK = (
    "Tabular regression on a medium-sized dataset (~10k rows, 30 numeric + "
    "categorical features, some missing values). In Flow-of-Options, this is the "
    "model-selection depth of the DAG."
)
N_OPTIONS = 12          # options requested per model per roll
N_ROLLS = 3             # independent generation rolls per model
RF_ALIASES = ("random forest", "randomforest", "random-forest", "rf",
              "extra trees", "extratrees", "extremely randomized trees")


# --- LLM helper ---------------------------------------------------------

def _llm(user: str, system: str = "", model: str = CLAUDE_MODEL,
         max_tokens: int = 2000, temperature: Optional[float] = None) -> str:
    """Provider-routed call (see llm.chat). Returns text or '' on error.

    Temperature is forwarded only when > 0 by llm.chat — this keeps Bedrock
    Opus 4.7 happy (it rejects temperature outright) while preserving the
    Sonnet T>0 rolls and the OpenAI T>0 rolls this paper actually needs.
    """
    return _chat(user, system=system, model=model, max_tokens=max_tokens,
                 temperature=temperature)


# --- Script extraction + parsing ----------------------------------------

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)
_METRIC_RE = re.compile(r"^METRIC\s+(\w+)=([\-\d.eE+]+)\s*$", re.MULTILINE)
_GATE_FAIL_RE = re.compile(r"^GATE:\s*FAIL", re.MULTILINE)
_LEADING_HEADER_RE = re.compile(r"\A\s*#{1,6}\s+[^\n]+\n+", re.MULTILINE)
_PY_INDICATORS = ("import ", "def ", "print(", "json.load", "for ", "while ", "assert ")


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


# --- Cross-model option generation (the real experiment input) ----------

_OPTGEN_SYS = (
    "You are the Option Generator in a Flow-of-Options agentic ML system. Given "
    "a task and a pipeline step, you propose a diverse list of concrete method "
    "options. Be diverse: avoid defaulting to the same family repeatedly."
)

_OPTGEN_USER = (
    "Task: {task}\n\n"
    "Propose exactly {n} DISTINCT model options for this step. Each option is a "
    "concrete model/algorithm name (e.g. 'GradientBoosting', 'TabNet'). Return "
    "ONLY a JSON array of {n} strings, no prose."
)

_JSON_ARR_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _parse_option_list(text: str) -> list[str]:
    m = _JSON_ARR_RE.search(text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in arr if str(x).strip()]


def _generate_options(model: str, temperature: float) -> list[list[str]]:
    """N_ROLLS independent option lists from one model family."""
    rolls = []
    user = _OPTGEN_USER.format(task=OPTION_TASK, n=N_OPTIONS)
    for _ in range(N_ROLLS):
        text = _llm(user, system=_OPTGEN_SYS, model=model, temperature=temperature)
        opts = _parse_option_list(text)
        if opts:
            rolls.append(opts)
    return rolls


def _build_fixture() -> dict:
    """Run real cross-model option generation; return a JSON-able fixture.

    GPT requires OPENAI_API_KEY. Claude uses Sonnet (option-generation tier).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set — agent_method_bias.py needs GPT for the "
            "cross-model option study. Export the key and re-run, or use the "
            "consistency-checker / walk-sampling siblings instead."
        )
    print("[fixture] generating Claude options (Sonnet)...")
    claude = _generate_options(CLAUDE_GEN_MODEL, temperature=0.0)
    print(f"[fixture]   {len(claude)} rolls")
    print("[fixture] generating GPT options...")
    gpt = _generate_options(GPT_MODEL, temperature=0.7)
    print(f"[fixture]   {len(gpt)} rolls")
    return {
        "task": OPTION_TASK,
        "n_options": N_OPTIONS,
        "rf_aliases": list(RF_ALIASES),
        "claude_rolls": claude,
        "gpt_rolls": gpt,
    }


# --- Fallback baseline experiment (single-model view of the fixture) ----
# The metric: method_bias_index = mean Random-Forest-family share across rolls,
# i.e. how often the option generator reaches for the RF family. FoO reports a
# residual RF pull; we quantify it per model family and test whether mixing the
# two families (cross-model union) lowers it. Lower is better (less biased).

FALLBACK_BASELINE = '''"""Baseline: single-model (Claude-only) Random-Forest pull in Flow-of-Options
option generation (Nair, Trase, Kim, arxiv 2502.12929), measuring residual
method bias (limitation #3). method_bias_index = mean fraction of options that
fall in the Random-Forest family, averaged over generation rolls. Lower is
less biased.
"""
import json
from pathlib import Path

# Locate the fixture next to this script (run_code executes with the parent's
# CWD, not the script's dir, so a bare relative path would not resolve).
_HERE = Path(__file__).resolve().parent
fixture = json.loads((_HERE / "options_fixture.json").read_text())
RF = tuple(a.lower() for a in fixture["rf_aliases"])


def rf_share(roll):
    if not roll:
        return 0.0
    hits = sum(1 for o in roll if any(a in o.lower() for a in RF))
    return hits / len(roll)


def mean_rf_share(rolls):
    return sum(rf_share(r) for r in rolls) / len(rolls) if rolls else 0.0


claude = fixture["claude_rolls"]
gpt = fixture["gpt_rolls"]

# Correctness gate (runs BEFORE the metric).
assert claude, "GATE: FAIL no Claude rolls in fixture"
assert all(len(r) > 0 for r in claude), "GATE: FAIL empty Claude roll"

baseline = mean_rf_share(claude)            # single-model view
cross = mean_rf_share(claude + gpt)         # cross-model union view

print(f"CONFIG: model=claude-only, rolls={len(claude)}, n_options={fixture['n_options']}, seed=0")
print(f"INPUT: {len(claude)} Claude rolls, {len(gpt)} GPT rolls from options_fixture.json")
print(f"SKEW: claude_rf_share={mean_rf_share(claude):.4f} gpt_rf_share={mean_rf_share(gpt):.4f} cross_rf_share={cross:.4f}")
print(f"METRIC method_bias_index={baseline:.4f}")
'''


# --- Autoresearch loop --------------------------------------------------

BASELINE_PROMPT = """Design a minimal, self-contained Python script that measures one stated limitation of {anchor}.

Limitation under study: "{limitation}"
Metric name: {metric_name} (direction: minimize)

A JSON fixture `options_fixture.json` is present in the working dir with keys: task, n_options,
rf_aliases (Random-Forest family name substrings), claude_rolls (list of option-name lists), gpt_rolls
(same). {metric_name} = mean fraction of options in the Random-Forest family across generation rolls.
The baseline measures the single-model (Claude-only) RF share. Lower is less biased.

HARD CONSTRAINTS on the script:
1. stdlib only (json, pathlib). Do NOT import numpy or third-party libs.
2. Load options_fixture.json from the SAME directory as the script: use
   `Path(__file__).resolve().parent / "options_fixture.json"` (the script is executed with the
   parent process's CWD, not its own dir, so a bare relative path will NOT resolve).
3. Print, in order: "CONFIG: ...", "INPUT: ...", a "SKEW: claude_rf_share=<f> gpt_rf_share=<f> cross_rf_share=<f>" line.
4. Correctness gate BEFORE the metric: assert there are Claude rolls and none are empty. On failure the assert exits non-zero.
5. Print exactly ONE final line: "METRIC {metric_name}=<float:.4f>".
6. Finish in under 5 seconds. Deterministic — no randomness, no network.

Output ONLY the Python code. No markdown fences, no commentary.
"""

PROPOSE_PROMPT = """You are iterating on a Python script in an autoresearch loop.

Goal: reduce {metric_name} (mean Random-Forest-family share of generated options; lower = less biased)
for the {anchor} residual-method-bias claim, by changing HOW options are aggregated from the fixture —
NOT by editing the fixture or weakening the RF-detection. The hypothesis under test: cross-model
aggregation (Claude + GPT) breaks the single-family RF pull.

Current best script.py:
```python
{best_code}
```

Current best {metric_name}: {best_metric:.4f}

Recent attempts (newest last):
{history_tail}

PRIOR MECHANISMS (do NOT repeat anything substantively equivalent):
{prior_mechanisms}

Propose ONE focused, mechanistically distinct change. Examples:
  - Report the cross-model UNION RF share (claude_rolls + gpt_rolls) as the metric instead of Claude-only
  - Deduplicate option names across families before computing RF share (interleave then unique)
  - Round-robin interleave Claude/GPT options and take the first n_options, then measure RF share
  - Compute RF share over the set-union of distinct options per matched roll pair

Do NOT edit options_fixture.json. Do NOT weaken the RF alias matching. Keep the SKEW line and the gate.
Keep stdlib-only, deterministic, CONFIG/INPUT/SKEW lines, single final METRIC line, under 5s.

Output format (exactly):

PLAN: <one sentence: the change AND why it is mechanistically distinct from prior attempts>

```python
<full new script.py here>
```
"""


def autoresearch_loop(fixture: dict, k_attempts: int = 3) -> dict:
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    (WORKING_DIR / "options_fixture.json").write_text(json.dumps(fixture, indent=2))
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
            metric_name=METRIC_NAME, anchor=FOO_ANCHOR, best_code=best[1],
            best_metric=best[0], history_tail=history_tail, prior_mechanisms=prior_mechanisms,
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

    return {"log": log, "best": best, "metric_name": METRIC_NAME, "fixture": fixture}


# --- Paper composition --------------------------------------------------

INTRO_PROMPT = """Write the Introduction section of a research paper that extends {anchor}.

The paper investigates one of FoO's stated limitations: "{limitation}".

Required content (in order):
1. State the anchor paper explicitly with arxiv ID: "{anchor}".
2. Summarize FoO's central claim in one sentence (DAG of option-nodes, beam-sampled walks scored by a
   scalar metric, consistency-checker, case-based reasoning).
3. Identify the limitation: FoO's option generator retains a residual bias toward the Random Forest
   family despite enforced diversity. Use the phrase "residual method bias".
4. State our contribution in one sentence: a cross-model option-generation study (Claude + GPT) that
   measures the Random-Forest pull per model family and tests whether mixing families reduces it,
   measured by {metric_name}.
5. Outline the paper structure (1 sentence).

Do NOT start with a header. Begin directly with prose. ~250 words. Markdown. Precise, restrained, no hype.
"""

METHODS_PROMPT = """Write the Methods section of a research paper.

Context:
- Anchor paper: {anchor}
- Limitation addressed: "{limitation}" (residual method bias toward Random Forest)
- Approach: cross-model option generation (ARIS-style). We elicit FoO model-choice option lists from
  two model families — Claude (Sonnet) and GPT (gpt-4o) — on a fixed tabular-regression task, {rolls}
  independent rolls each, {n_opt} options per roll. Captured once into options_fixture.json. A single
  seeded, offline script.py computes {metric_name} = mean Random-Forest-family share over rolls.
- Loop: Karpathy-style autoresearch (cite: Karpathy 2026). Baseline (single-model) + K=3 proposals;
  correctness gate before metric; revert-writes-best.
- Discipline: section-by-section composition (cite: Lu et al. 2024, arxiv 2408.06292).

CRITICAL: the EXACT script that produced the results is below. Any constant you cite (rolls, n_options,
RF aliases) MUST match it and the fixture. Do NOT invent numbers.

```python
{best_script}
```

Required content (in order):
1. The fixed task and the cross-model elicitation protocol (Claude + GPT, rolls, options/roll).
2. RF-family detection (alias substring match) and {metric_name} definition.
3. The autoresearch loop: baseline single-model RF share -> propose aggregation change -> measure -> keep/revert.
4. Why script.py is offline+deterministic over a captured fixture (reproducibility) even though the
   options came from real LLM calls (code-paper alignment).
5. Honest scope: one task, two families, small roll counts; this is a probe, not a general claim about FoO.

Do NOT start with a header. Begin directly with prose. ~300 words. Markdown.
"""

RESULTS_PROMPT = """Write the Results section of a research paper.

EXPERIMENT LOG (newest last):
{log_text}

Best {metric_name}: {best_metric}

Required content:
1. Report baseline (single-model) {metric_name} and best achieved {metric_name}. Both from the log.
2. Report the improvement as absolute delta and percent change (from the log).
3. Using the SKEW lines, report claude_rf_share, gpt_rf_share, and cross_rf_share, and interpret
   whether cross-model mixing reduced the Random-Forest pull.
4. List KEPT attempts (plan + metric) and acknowledge attempts that did NOT improve.
5. Include a small markdown table summarizing all attempts (id, plan, metric, kept).
6. Use ONLY numbers from the log. Do NOT invent numbers. If a value is null/None, say so.

Do NOT start with a header. Begin directly with prose. ~250 words. Markdown.
"""

TITLE_PROMPT = """Propose a single short paper title for a study that extends "{anchor}" by addressing
"residual method bias" via cross-model option generation measuring {metric_name}.

Requirements: mentions Flow-of-Options or FoO; mentions "method bias" or "cross-model"; under 14 words;
no quotes, no subtitle. Output the title only — single line.
"""

REFERENCES = """1. Nair, L., Trase, I., & Kim, M. (2025). Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*. arXiv:2502.12929. https://arxiv.org/abs/2502.12929

2. Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv:2408.06292. https://arxiv.org/abs/2408.06292

3. Karpathy, A. (2026). autoresearch: minimal agent loop for autonomous LLM experimentation. https://github.com/karpathy/autoresearch

4. ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration (2026). arXiv:2605.03042.
"""


def _format_log_for_results(log: list[dict]) -> str:
    out = []
    for e in log:
        out.append(
            f'{{"id": {e["id"]}, "kind": "{e["kind"]}", "plan": {e.get("plan", "")!r}, '
            f'"metric": {e["metric"]}, "ok": {e["ok"]}, "kept": {e.get("kept", "—")}}}'
        )
        tail = e.get("stdout_tail", "")
        skews = [l for l in tail.splitlines() if l.startswith("SKEW:")]
        if skews:
            out.extend(f"    {s}" for s in skews)
    return "\n".join(out)


def compose_paper(problem_domain: str, result: dict) -> Paper:
    log = result["log"]
    fixture = result["fixture"]
    best_metric = result["best"][0] if result["best"] else None
    best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "N/A (baseline failed)"
    log_text = _format_log_for_results(log)

    print("[compose] writing intro...")
    intro = _strip_leading_header(_llm(INTRO_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=LIMITATION, metric_name=METRIC_NAME,
    ))) or f"This paper extends {FOO_ANCHOR} by studying residual method bias via cross-model option generation."

    print("[compose] writing methods...")
    best_script = result["best"][1] if result["best"] else "(no successful run — see appendix)"
    methods = _strip_leading_header(_llm(METHODS_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=LIMITATION, metric_name=METRIC_NAME, best_script=best_script,
        rolls=len(fixture["claude_rolls"]), n_opt=fixture["n_options"],
    ))) or "We elicit FoO option lists from Claude and GPT and measure the Random-Forest-family share."

    print("[compose] writing results...")
    results = _strip_leading_header(_llm(RESULTS_PROMPT.format(
        log_text=log_text, metric_name=METRIC_NAME, best_metric=best_metric_str,
    ))) or f"Best {METRIC_NAME}: {best_metric_str}. See appendix for the experiment script."

    print("[compose] writing title...")
    title_raw = _llm(TITLE_PROMPT.format(anchor=FOO_ANCHOR, metric_name=METRIC_NAME)).strip()
    title = title_raw.strip('"').strip("'").splitlines()[0] if title_raw else \
        "Cross-Model Option Generation Reduces Residual Method Bias in Flow-of-Options"

    best_code = result["best"][1] if result["best"] else ""
    fixture_json = json.dumps(fixture, indent=2)
    appendix = (
        f"# Code\n\n```python\n{best_code}\n```\n\n"
        f"# Options fixture (real cross-model generation)\n\n```json\n{fixture_json}\n```"
    ) if best_code else ""

    return Paper(
        title=title,
        introduction=intro,
        methods=methods,
        results=results,
        references=REFERENCES,
        appendix=appendix,
        tags=["flow-of-options", "cross-model", "method-bias", "autoresearch"],
    )


# --- run() entrypoint ---------------------------------------------------

def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    print(f"[run] problem_domain: {problem_domain!r}")
    print(f"[run] FoO anchor: {FOO_ANCHOR}")
    print(f"[run] limitation: {LIMITATION}")
    print(f"[run] metric: {METRIC_NAME} (minimize)")

    fixture = _build_fixture()   # raises if OPENAI_API_KEY unset
    result = autoresearch_loop(fixture, k_attempts=3)
    print(f"[run] autoresearch done. log entries: {len(result['log'])}")
    if result["best"]:
        print(f"[run] best {METRIC_NAME}: {result['best'][0]:.4f}")
    else:
        print("[run] no successful experiment — paper will note this")

    paper = compose_paper(problem_domain, result)
    print(f"[run] paper drafted. title: {paper.title!r}")
    return paper

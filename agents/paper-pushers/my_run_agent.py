"""Paper-pushers run agent — Phase A.

FoO-over-paper-artifacts extension with Karpathy-style autoresearch on script.py.

See docs/planning/architecture.md for full design and research backing.

Phase A scope:
  - Hardcoded FoO limitation (walk-sampling inefficiency)
  - Autoresearch loop: baseline + 3 attempts on a single script.py
  - METRIC parse + GATE check + revert-writes-best
  - Section-by-section paper composition (Sakana v1 pattern)
  - Lightweight literature mining via OpenAlex (academic-grade, replaces DDG)
  - Deterministic experiment-log table + Discussion section (defense-pass-aligned)

Out of scope (Phase B+):
  - FoO DAG over paper artifacts (Phase B)
  - 9-axis PoLL critic + revision loop (Phase C)
  - get_paper ecosystem mining (Phase B)
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from hackathon_science import Paper
from hackathon_science.tools import run_code
from hackathon_science.utils import call_llm

import urllib.parse
import urllib.request


# --- Configuration ------------------------------------------------------

MODEL = "global.anthropic.claude-opus-4-7"
WORKING_DIR = Path(__file__).parent / "files"
PAPERS_DIR = Path(__file__).parent / "papers"  # persistent archive — .cache/ gets clobbered every run

FOO_ANCHOR = "Flow-of-Options (Nair, Trase, Kim, ICML 2025, arxiv 2502.12929)"
FOO_LIMITATIONS = [
    "Metric dependency — assumes a quantifiable evaluator exists",
    "Data availability — needs input data / datasets",
    "Residual method bias — still biased toward Random Forest despite diversity",
    "Walk-sampling inefficiency — naive sampling produces repeats",
    "Module-specific issues — consistency-checker false positives, retriever mis-selection",
]


# --- LLM helper ---------------------------------------------------------

def _llm(user: str, system: str = "", model: str = MODEL, max_tokens: int = 4000,
         retry_on_empty: bool = True) -> str:
    """One-shot Bedrock Converse call. Returns just the text or '' on error.

    Retries ONCE if the first response is empty (no content) — empty responses
    silently fall through to fallback strings otherwise, producing thin papers.
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
    """Pull Python code out of a fenced block, or return text if it looks like Python.

    Guard against the LLM returning prose-without-fence and us writing that to script.py
    (which would crash run_code with SyntaxError).
    """
    m = _CODE_FENCE.search(text)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    if any(ind in stripped for ind in _PY_INDICATORS):
        return stripped
    return ""


def _strip_leading_header(text: str) -> str:
    """Drop any leading markdown header (the platform / our compose pass adds its own)."""
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


# --- Fallback baseline (used if LLM-generated baseline fails) -----------

FALLBACK_BASELINE = '''"""Baseline: naive uniform-random walk sampling in a FoO DAG.

Measures collision rate at varying walk counts for the Nair-Trase-Kim
Flow-of-Options framework (arxiv 2502.12929). Lower collision rate is better.
"""
import random

random.seed(0)

K = 4   # options per depth
D = 5   # depths
N = 200 # walks sampled

# Correctness gate — must pass BEFORE the metric is printed
assert K > 0 and D > 0 and N > 0, "GATE: FAIL invalid params"
assert N <= K ** D, "GATE: FAIL N exceeds DAG capacity"

print(f"CONFIG: K={K}, D={D}, N={N}, seed=0")
print("INPUT: synthetic FoO DAG, uniform-random walk sampling")

walks = []
for _ in range(N):
    walk = tuple(random.randint(0, K - 1) for _ in range(D))
    walks.append(walk)

unique = len(set(walks))
collision_rate = 1.0 - (unique / N)

print(f"unique={unique}, total={N}")
print(f"METRIC collision_rate={collision_rate:.4f}")
'''


# --- Autoresearch loop --------------------------------------------------

BASELINE_PROMPT = """Design a minimal, self-contained Python script that measures one stated limitation of {anchor}.

Limitation under study: "{limitation}"
Metric name: {metric_name} (direction: minimize)

Definition of {metric_name}: 1 - (unique_walks / total_walks) for uniform-random walk
sampling over a synthetic FoO DAG with K options per depth, D depths, N walks.

HARD CONSTRAINTS on the script:
1. Only stdlib (random) — do NOT import numpy or any third-party libraries.
2. Set random.seed(0) explicitly at the top.
3. Print these lines BEFORE the metric (in order):
   - "CONFIG: K=<int>, D=<int>, N=<int>, seed=0"
   - "INPUT: <one-line description>"
4. Run correctness assertions BEFORE printing the metric. On any failure, the
   `assert` will exit non-zero. Include at least: K>0, D>0, N>0, N<=K**D.
5. Print exactly ONE final line: "METRIC {metric_name}=<float:.4f>"
6. Must finish in under 5 seconds on a laptop.
7. Use K=4, D=5, N=200 for the baseline.

Output ONLY the Python code. No markdown fences, no commentary, no preamble.
"""

PROPOSE_PROMPT = """You are iterating on a Python script in an autoresearch loop.

Goal: reduce {metric_name} (lower is better) for the {anchor} walk-sampling claim.

Current best script.py:
```python
{best_code}
```

Current best {metric_name}: {best_metric:.4f}

Recent attempts (newest last):
{history_tail}

PRIOR MECHANISMS (do NOT propose anything substantively equivalent to these):
{prior_mechanisms}

Propose ONE focused improvement that is MECHANISTICALLY DISTINCT from every prior mechanism above. Examples of valid (and distinct) moves:
  - Memo-conditioning via a seen-set with rejection
  - Direct enumeration of the index space with random.sample
  - Stratified sampling: partition the option space into buckets and draw from each
  - Halton / van der Corput low-discrepancy sequences cast into the option index
  - Hash-based reservoir sampling with explicit collision avoidance
  - Lexicographic walk enumeration with a random starting offset

If the current best already achieves the theoretical minimum (e.g., 0.0 collision rate), propose a CHANGE OF METRIC SCOPE instead: vary N, K, or D and observe behaviour. Document this clearly in your PLAN line.

Constraints unchanged from the baseline:
  - stdlib only (random, collections, itertools allowed; no numpy, no third-party)
  - seeded with random.seed(0)
  - CONFIG, INPUT lines first; assertions before metric
  - single final "METRIC {metric_name}=<float:.4f>" line
  - under 5 seconds

Do NOT change the metric definition. Do NOT remove assertions.

Output format (exactly):

PLAN: <one sentence describing the change AND why it is mechanistically distinct from prior attempts>

```python
<full new script.py here>
```
"""


def autoresearch_loop(metric_name: str, k_attempts: int = 3) -> dict:
    """Run baseline + k_attempts of autoresearch. Writes script.py at each step."""
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    # Baseline — try LLM first, fall back to hand-written baseline if needed.
    print("[autoresearch] designing baseline...")
    llm_baseline = _extract_code(_llm(
        BASELINE_PROMPT.format(
            anchor=FOO_ANCHOR,
            limitation=FOO_LIMITATIONS[3],
            metric_name=metric_name,
        )
    ))
    baseline_code = llm_baseline if llm_baseline.strip() else FALLBACK_BASELINE

    out = run_code(baseline_code, filename="script.py", working_dir=str(WORKING_DIR))
    m = _parse_metric(out, metric_name)
    ok = _gate_ok(out) and m is not None
    log.append({
        "id": 0, "kind": "baseline", "plan": "baseline (LLM-designed)" if llm_baseline.strip() else "baseline (fallback)",
        "metric": m, "ok": ok, "stdout_tail": out[-400:],
    })
    print(f"[autoresearch] baseline ok={ok} metric={m}")

    if not ok:
        # Retry with hardcoded fallback so the appendix is at least valid code.
        print("[autoresearch] baseline failed, using fallback")
        out = run_code(FALLBACK_BASELINE, filename="script.py", working_dir=str(WORKING_DIR))
        m = _parse_metric(out, metric_name)
        ok = _gate_ok(out) and m is not None
        baseline_code = FALLBACK_BASELINE
        log.append({
            "id": 0, "kind": "baseline-fallback", "plan": "fallback baseline",
            "metric": m, "ok": ok, "stdout_tail": out[-400:],
        })

    best = (m, baseline_code) if ok else None

    # K attempts
    for i in range(1, k_attempts + 1):
        if best is None:
            print(f"[autoresearch] no working baseline, skipping attempts")
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
            metric_name=metric_name,
            anchor=FOO_ANCHOR,
            best_code=best[1],
            best_metric=best[0],
            history_tail=history_tail,
            prior_mechanisms=prior_mechanisms,
        ))

        plan_line = next((l for l in response.splitlines() if l.strip().startswith("PLAN:")), "PLAN: <no plan>")
        plan = plan_line.split(":", 1)[1].strip()
        new_code = _extract_code(response)

        if not new_code or new_code == best[1]:
            log.append({"id": i, "kind": "attempt", "plan": plan, "metric": None,
                        "ok": False, "kept": False, "stdout_tail": "[no new code]"})
            continue

        out = run_code(new_code, filename="script.py", working_dir=str(WORKING_DIR))
        m = _parse_metric(out, metric_name)
        ok = _gate_ok(out) and m is not None
        keep = ok and m < best[0]  # lower is better

        log.append({
            "id": i, "kind": "attempt", "plan": plan,
            "metric": m, "ok": ok, "kept": keep,
            "stdout_tail": out[-400:],
        })
        print(f"[autoresearch]   metric={m} ok={ok} keep={keep}")

        if keep:
            best = (m, new_code)
        else:
            # Revert: write best back so the appendix reflects the winning experiment.
            run_code(best[1], filename="script.py", working_dir=str(WORKING_DIR))

    # Belt-and-suspenders: ensure script.py on disk matches `best` regardless of loop state.
    if best is not None:
        (WORKING_DIR / "script.py").write_text(best[1])

    return {"log": log, "best": best, "metric_name": metric_name}


# --- Paper composition --------------------------------------------------

INTRO_PROMPT = """Write the Introduction section of a research paper that extends {anchor}.

The paper investigates one of FoO's stated limitations: "{limitation}".

External literature retrieved (use these as concrete citations where relevant; do NOT invent additional sources):
{literature}

Required content (in order):
1. State the anchor paper explicitly with arxiv ID: "{anchor}".
2. Briefly summarize FoO's central claim in 1-2 sentences (DAG of option-nodes with beam-sampled walks, scored by a scalar metric, with consistency-checker pruning and case-based reasoning).
3. Situate the work against the retrieved literature above — cite at least one [L#] entry by title to show familiarity with adjacent autoresearch / sampling work.
4. Identify the specific limitation we address. Use the phrase: "walk-sampling inefficiency". Briefly explain WHY this is a real bottleneck (naive sampling produces repeats that waste evaluator budget).
5. State our contribution in 1-2 sentences: an autoresearch-driven empirical study of {metric_name} with iterative refinement of the walk-sampling strategy.
6. Outline the paper structure (1 sentence).

Do NOT start with a "## Introduction" or "# Introduction" header — the platform adds section headers itself. Begin directly with prose.

Length: ~450 words. Markdown formatting.

Style: precise, technical, restrained. Avoid hype. Do not invent results.
"""

METHODS_PROMPT = """Write the Methods section of a research paper.

Context:
- Anchor paper: {anchor}
- Limitation addressed: "{limitation}" (walk-sampling inefficiency)
- Approach: Karpathy-style autoresearch loop on a single Python script (cite: Karpathy 2026, "autoresearch" GitHub repository). Baseline + K=3 iterative proposals. Single scalar metric ({metric_name}, lower is better). Correctness gate runs before metric; failed gates trigger a revert that restores the best-known script.
- Discipline: section-by-section composition (cite: Lu et al. 2024, "The AI Scientist", arxiv 2408.06292).

External literature retrieved (cite at most one [L#] entry by title for sampling-method context; do NOT invent additional sources):
{literature}

CRITICAL: the EXACT script that ran for the reported results is shown below. Any numerical constant you cite (K, D, N, seed, etc.) MUST match this script verbatim. If you state K=4 and D=5, the script must show K=4 and D=5. Do NOT invent values.

```python
{best_script}
```

Required content (in order, each ~1 paragraph):
1. The simulated FoO walk-sampling setup, citing the EXACT K, D, N from the script above. Explain what a "walk" represents in our adaptation (a sampled option-tuple over D depths).
2. Definition of the metric: collision_rate = 1 - (unique_walks / total_walks). Explain WHY this metric tracks the stated limitation (repeats waste downstream evaluator calls).
3. The autoresearch loop: read current best → propose mechanistically distinct change → run → measure metric → keep if improved, revert if not. Note the prior-mechanisms guard that prevents the proposer from repeating earlier ideas.
4. The correctness-gate-before-metric discipline: assertions run before METRIC is printed, so a script that produces a number AT ALL has already passed validity checks. Cite [L#] if a retrieved source describes a similar pattern.
5. Reproducibility: explicit seed (random.seed(0)), stdlib-only dependencies, deterministic data generation, single self-contained script.
6. Honest scope: we extend FoO's walk-sampling mechanism specifically; we do not claim a general improvement to FoO.

Do NOT start with a "## Methods" header — the platform adds section headers itself. Begin directly with prose.

Length: ~550 words. Markdown.
"""

RESULTS_PROMPT = """Write the Results section of a research paper.

EXPERIMENT LOG (newest last):
{log_text}

Best {metric_name}: {best_metric}

Required content (in order, each ~1 paragraph):
1. Report the baseline {metric_name} and the best achieved {metric_name}. Both numbers MUST come from the log above.
2. Report the improvement as absolute delta and as percent change (compute from log). If best == baseline, say so explicitly.
3. Walk through each KEPT attempt: its plan, the metric it achieved, and WHY (per the plan) that mechanism reduced the metric.
4. Acknowledge each attempt that did NOT improve. Use the plan text from the log to be specific about what was tried and what mechanism was expected.
5. Note any boundary effects: did the loop hit a theoretical minimum (e.g., 0.0)? Did proposers run out of distinct mechanisms?
6. Be specific. Use ONLY numbers from the log. Do NOT invent numbers. If a value is null/None, say so.

Do NOT include a markdown table in your prose — the composer appends a deterministic table from the log automatically. Focus on the narrative.

Do NOT start with a "## Results" header — the platform adds section headers itself. Begin directly with prose.

Length: ~450 words. Markdown.
"""

TITLE_PROMPT = """Propose a single short paper title for a study that extends "{anchor}" by addressing
"walk-sampling inefficiency" via an autoresearch loop measuring {metric_name}.

Requirements:
- Mentions Flow-of-Options or FoO explicitly.
- Mentions either "walk sampling" or "autoresearch".
- Under 12 words.
- No quotes, no subtitle.

Output the title only — single line, nothing else.
"""

REFERENCES = """1. Nair, L., Trase, I., & Kim, M. (2025). Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*. arXiv:2502.12929. https://arxiv.org/abs/2502.12929

2. Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv:2408.06292. https://arxiv.org/abs/2408.06292

3. Karpathy, A. (2026). autoresearch: AI agents running research on single-GPU nanochat training automatically. https://github.com/karpathy/autoresearch

4. Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., & Ha, D. (2025). The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. arXiv:2504.08066.

5. Tie, G., et al. (2026). AutoResearch AI: Towards AI-Powered Research Automation for Scientific Discovery. arXiv:2605.23204.
"""


# --- Literature mining (OpenAlex, replaces DDG) ------------------------
#
# OpenAlex (https://openalex.org) is a free, no-auth scholarly index covering
# ~250M works. Polite usage convention is to include a contact email in the
# User-Agent so they can throttle abuse without blocking everyone.

LIT_QUERIES = [
    "Flow-of-Options Nair Trase Kim LLM reasoning",
    "autoresearch LLM agent iterative script optimization",
    "walk sampling diversity DAG beam search",
    "low-discrepancy quasi-random sampling reasoning",
]

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
OPENALEX_USER_AGENT = "paper-pushers/0.1 (mailto:contact@augmentationlab.org)"
LIT_TIMEOUT_SECS = 10.0


def _abstract_from_inverted(inv: Optional[dict]) -> str:
    """OpenAlex ships abstracts as {word: [positions]}. Reconstruct prose."""
    if not inv:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def _format_authors(authorships: list[dict], max_authors: int = 3) -> str:
    names = [(a.get("author") or {}).get("display_name", "") for a in authorships[:max_authors]]
    names = [n for n in names if n]
    if not names:
        return ""
    suffix = " et al." if len(authorships) > max_authors else ""
    return ", ".join(names) + suffix


def _openalex_to_lit_entry(w: dict) -> Optional[dict]:
    """Normalize an OpenAlex Work into our {title, url, snippet} shape."""
    title = (w.get("title") or "").strip()
    if not title:
        return None
    abstract = _abstract_from_inverted(w.get("abstract_inverted_index"))
    authors = _format_authors(w.get("authorships") or [])
    year = w.get("publication_year")
    # URL preference: DOI > OpenAlex landing page
    url = w.get("doi") or w.get("id") or ""
    head_bits = [b for b in [authors, f"({year})" if year else ""] if b]
    head = " ".join(head_bits).strip()
    snippet = (f"{head}. {abstract}" if head else abstract).strip(". ").strip()
    return {"title": title, "url": url, "snippet": snippet}


def _openalex_search(query: str, per_page: int = 5) -> list[dict]:
    """Query OpenAlex /works. Raises on network/parse error."""
    params = urllib.parse.urlencode({
        "search": query,
        "per-page": per_page,
        "select": "id,doi,title,abstract_inverted_index,publication_year,authorships",
    })
    req = urllib.request.Request(
        f"{OPENALEX_ENDPOINT}?{params}",
        headers={"User-Agent": OPENALEX_USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=LIT_TIMEOUT_SECS) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("results", []) or []


def gather_literature(max_total: int = 8) -> list[dict]:
    """Pull deduped academic sources from OpenAlex. Returns [{title,url,snippet}, ...].

    Failures (network, JSON, OpenAlex outage) are non-fatal — the paper still
    composes without literature context, just with weaker grounding.
    """
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    out: list[dict] = []
    for q in LIT_QUERIES:
        if len(out) >= max_total:
            break
        try:
            works = _openalex_search(q, per_page=5)
        except Exception as e:
            print(f"[lit] OpenAlex query failed for {q!r}: {e}", file=sys.stderr)
            continue
        for w in works:
            entry = _openalex_to_lit_entry(w)
            if not entry:
                continue
            url_key = entry["url"].lower()
            title_key = entry["title"].lower()
            if (url_key and url_key in seen_urls) or title_key in seen_titles:
                continue
            if url_key:
                seen_urls.add(url_key)
            seen_titles.add(title_key)
            out.append(entry)
            if len(out) >= max_total:
                break
    print(f"[lit] OpenAlex returned {len(out)} unique sources")
    return out


def _lit_block(literature: list[dict]) -> str:
    """Render retrieved sources as a bullet list for prompt grounding."""
    if not literature:
        return "(no external literature retrieved — proceed without external citation)"
    lines = []
    for i, h in enumerate(literature, start=1):
        snippet = (h.get("snippet") or "")[:280].replace("\n", " ")
        lines.append(f"[L{i}] {h['title']} — {snippet}\n      URL: {h['url']}")
    return "\n".join(lines)


def _augment_references(base: str, literature: list[dict]) -> str:
    """Append retrieved sources as a 'Retrieved literature' block, continuing numbering."""
    if not literature:
        return base
    # Use [L#] labels matching the in-text citations emitted by _lit_block.
    # If we numbered these as 6./7./8. (continuation of REFERENCES), the LLM's
    # "[L1] ... " citations in the prose would have no resolvable target.
    lines = ["", "## Retrieved literature", ""]
    for offset, h in enumerate(literature, start=1):
        lines.append(f"[L{offset}] {h['title']}. {h['url']}")
    return base.rstrip() + "\n\n" + "\n".join(lines).lstrip()


# --- Results table (deterministic, generated from log) ------------------

def _results_table_md(log: list[dict]) -> str:
    """Render the autoresearch log as a markdown table — always present, never LLM-fabricated."""
    rows = [
        "## Experiment log",
        "",
        "| id | kind | plan | metric | gate | kept |",
        "|---|---|---|---|---|---|",
    ]
    for e in log:
        plan = str(e.get("plan", "") or "")[:90].replace("|", "/").replace("\n", " ")
        m = e.get("metric")
        metric = f"{m:.4f}" if isinstance(m, (int, float)) else "—"
        gate = "pass" if e.get("ok") else "fail"
        kept_raw = e.get("kept")
        if kept_raw is True:
            kept = "kept"
        elif kept_raw is False:
            kept = "reverted"
        else:
            kept = "—"
        rows.append(f"| {e.get('id','?')} | {e.get('kind','?')} | {plan} | {metric} | {gate} | {kept} |")
    return "\n".join(rows)


# --- Discussion prompt --------------------------------------------------

DISCUSSION_PROMPT = """Write the Discussion section of a research paper.

Anchor: {anchor}
Limitation addressed: "{limitation}"
Metric: {metric_name} (lower is better)
Best {metric_name} achieved: {best_metric}

Experiment log (compact):
{log_text}

External literature retrieved (use these as concrete citations where relevant; do NOT invent additional sources):
{literature}

Required content (in order, each ~1 paragraph):
1. What our autoresearch result implies about FoO's stated walk-sampling inefficiency. Reference the actual metric movement (use numbers from the log).
2. Honest scope: this is one simulated DAG (K, D, N from the experiment); we do not claim generality to real ML pipelines or the full FoO framework.
3. Connection to or contrast with the retrieved literature above. Cite at least one [L#] entry by title.
4. Limitations of our own study: small parameter sweep, single metric, no real downstream task, K=3 attempts only.
5. Concrete next steps (sweep K/D/N, plug into a real FoO implementation, compare against case-based reasoning module).

Do NOT start with a "## Discussion" header — the platform adds section headers itself. Begin directly with prose.

Length: ~400 words. Markdown. Restrained, technical, no hype.
"""


def _format_log_for_results(log: list[dict]) -> str:
    """Render the log as proper JSONL for the Results / Discussion prompts.

    Use json.dumps so None → null, True/False → true/false. The LLM treats the
    block as structured evidence; Python-literal rendering (None/True/False)
    leaks into prose otherwise.
    """
    return "\n".join(
        json.dumps({
            "id": e["id"],
            "kind": e["kind"],
            "plan": e.get("plan", ""),
            "metric": e["metric"],
            "ok": e["ok"],
            "kept": e.get("kept"),  # missing → null (baseline rows have no kept/reverted state)
        }, ensure_ascii=False)
        for e in log
    )


def compose_paper(problem_domain: str, result: dict, literature: Optional[list[dict]] = None) -> Paper:
    metric_name = result["metric_name"]
    log = result["log"]
    best_metric = result["best"][0] if result["best"] else None
    best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "N/A (baseline failed)"
    log_text = _format_log_for_results(log)
    literature = literature or []
    lit_text = _lit_block(literature)

    print("[compose] writing intro...")
    intro = _strip_leading_header(_llm(INTRO_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=FOO_LIMITATIONS[3], metric_name=metric_name,
        literature=lit_text,
    ))) or f"This paper extends {FOO_ANCHOR} by investigating its stated walk-sampling inefficiency limitation."

    print("[compose] writing methods...")
    best_script = result["best"][1] if result["best"] else "(no successful run — see appendix)"
    methods = _strip_leading_header(_llm(METHODS_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=FOO_LIMITATIONS[3], metric_name=metric_name,
        best_script=best_script, literature=lit_text,
    ))) or "We simulate FoO walk sampling and apply an autoresearch loop to iteratively improve a single scalar metric."

    print("[compose] writing results narrative...")
    results_prose = _strip_leading_header(_llm(RESULTS_PROMPT.format(
        log_text=log_text, metric_name=metric_name, best_metric=best_metric_str,
    ))) or f"Best {metric_name}: {best_metric_str}. See appendix for the experiment script."

    print("[compose] writing discussion...")
    discussion = _strip_leading_header(_llm(DISCUSSION_PROMPT.format(
        anchor=FOO_ANCHOR, limitation=FOO_LIMITATIONS[3], metric_name=metric_name,
        best_metric=best_metric_str, log_text=log_text, literature=lit_text,
    ))) or "Our autoresearch loop produced a reproducible measurement of the stated FoO walk-sampling inefficiency, scoped to one simulated DAG."

    # Results body = narrative + deterministic table + discussion. Table is
    # generated from the log, never LLM-fabricated, so even if the prose
    # hallucinates a number the table is the source of truth for reviewers.
    table = _results_table_md(log)
    results = f"{results_prose}\n\n{table}\n\n## Discussion\n\n{discussion}"

    print("[compose] writing title...")
    title_raw = _llm(TITLE_PROMPT.format(anchor=FOO_ANCHOR, metric_name=metric_name)).strip()
    title = title_raw.strip('"').strip("'").splitlines()[0] if title_raw else \
        f"Autoresearch on Flow-of-Options Walk-Sampling Inefficiency"

    # Manually compose the appendix from the best script — the runner's
    # auto-population path only works when CWD includes `agents/`, which is
    # not the case when `uv run hackathon run` is invoked from the repo root.
    best_code = result["best"][1] if result["best"] else ""
    appendix = f"# Code\n\n```python\n{best_code}\n```" if best_code else ""

    references = _augment_references(REFERENCES, literature)

    return Paper(
        title=title,
        introduction=intro,
        methods=methods,
        results=results,
        references=references,
        appendix=appendix,
        tags=["flow-of-options", "autoresearch", "agentic-discovery", "walk-sampling"],
    )


# --- Persistent archive (.cache/ gets clobbered every run) --------------

def _slugify(s: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-")
    return (slug[:max_len].rstrip("-") or "untitled")


def _render_markdown(paper: Paper) -> str:
    parts = [f"# {paper.title}", ""]
    if paper.introduction:
        parts += ["## Introduction", "", paper.introduction, ""]
    if paper.methods:
        parts += ["## Methods", "", paper.methods, ""]
    if paper.results:
        parts += ["## Results", "", paper.results, ""]
    if paper.references:
        parts += ["## References", "", paper.references, ""]
    if paper.appendix:
        parts += [paper.appendix, ""]
    return "\n".join(parts)


def archive_paper(paper: Paper) -> Path:
    """Snapshot the paper to PAPERS_DIR with a timestamped, slugged filename.

    Returns the JSON path. .md sibling is written alongside. Non-fatal: if the
    archive write fails for any reason, we log and return None — the platform's
    own .cache/ write is unaffected.
    """
    try:
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify(paper.title)
        base = PAPERS_DIR / f"{ts}-{slug}"
        json_path = base.with_suffix(".json")
        md_path = base.with_suffix(".md")
        json_path.write_text(json.dumps(asdict(paper), indent=2, ensure_ascii=False))
        md_path.write_text(_render_markdown(paper))
        return json_path
    except Exception as e:
        print(f"[archive] failed: {e}", file=sys.stderr)
        return None


# --- run() entrypoint ---------------------------------------------------

def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    """Phase A skeleton: autoresearch on a FoO walk-sampling limitation."""
    metric_name = "collision_rate"

    print(f"[run] problem_domain: {problem_domain!r}")
    print(f"[run] FoO anchor: {FOO_ANCHOR}")
    print(f"[run] limitation: {FOO_LIMITATIONS[3]}")
    print(f"[run] metric: {metric_name} (minimize)")

    result = autoresearch_loop(metric_name=metric_name, k_attempts=3)
    print(f"[run] autoresearch done. log entries: {len(result['log'])}")
    if result["best"]:
        print(f"[run] best {metric_name}: {result['best'][0]:.4f}")
    else:
        print(f"[run] no successful experiment — paper will note this")

    print("[run] gathering literature via OpenAlex...")
    literature = gather_literature(max_total=8)

    paper = compose_paper(problem_domain, result, literature=literature)
    print(f"[run] paper drafted. title: {paper.title!r}")

    archive_path = archive_paper(paper)
    if archive_path:
        print(f"[run] archived to {archive_path}")

    return paper

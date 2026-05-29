---
created: 2026-05-28
status: active
author: feature-dev:code-reviewer agent (via main session)
session: hackathon-paper-pushers
branch: main
informed_by: agents/paper-pushers/my_run_agent.py (primary), hackathon_science/models.py, hackathon_science/tools.py
notes: Bug-only review (no style/architecture). Conducted just before first end-to-end Bedrock run after the literature-mining + Discussion + table layered edit.
---

# Code review: agents/paper-pushers/my_run_agent.py

## Summary

- **1 confirmed bug** (Finding 1, confidence 85) — wrong-data, applied on every successful run with ≥1 retrieved source.
- **1 secondary issue** (Finding 2, confidence 75) — below crash threshold, noted for completeness.
- No crash paths, no Paper-contract violations, no fragile `format()` calls. `search_web` failure handling is correct.

---

## Finding 1 — In-text `[L#]` citations have no matching bibliography entries

**Confidence:** 85
**Locations:** `_lit_block` (lines 471–473); `_augment_references` (lines 484–485); INTRO_PROMPT (~line 340); METHODS_PROMPT (~line 361); DISCUSSION_PROMPT (~line 530).

**What's wrong:** `_lit_block` labels retrieved sources `[L1]`, `[L2]`, … and all three composition prompts instruct the LLM to "cite at least one `[L#]` entry by title". The LLM writes `[L1]`, `[L2]`, … into prose. But `_augment_references` appends those same sources to the bibliography under continuation numbering (`6.`, `7.`, `8.`, …). No `[L#]` entry ever appears in `Paper.references`. Every paper with any retrieved literature ships with dangling, unresolvable citations.

**Why it's a bug, not a preference:** the two functions were written independently and never reconciled. The mismatch is structural and guaranteed.

**Fix (one-line):** change `_augment_references` to emit `[L#]` labels:
```python
for offset, h in enumerate(literature, start=1):
    lines.append(f"[L{offset}] {h['title']}. {h['url']}")
```

---

## Finding 2 — `_format_log_for_results` renders Python `None`/`True`/`False` into JSONL-ish log text

**Confidence:** 75 (below the report threshold; included for completeness)
**Location:** `_format_log_for_results` (lines 550–552).

**What's wrong:** the formatter uses `{e["metric"]}` etc., which renders Python literals (`None`, `True`, `False`) instead of JSON (`null`, `true`, `false`). The string is fed verbatim to RESULTS_PROMPT and DISCUSSION_PROMPT as structured evidence. RESULTS_PROMPT line ~396 already says "If a value is null/None, say so" so the impact is bounded; not a crash path.

**Fix (optional):** map `None → "null"`, booleans → `"true"`/`"false"` before format.

---

## Investigated and cleared (no bugs)

- `result["best"]` None handling — all access guarded; fallback strings satisfy the Paper contract in every path.
- `PROPOSE_PROMPT.format(best_code=…)` and `METHODS_PROMPT.format(best_script=…)` — `str.format()` only scans the template, not substituted values; braces in user code are safe.
- `best_metric == 0.0` at format sites — `0.0 is not None` and `f"{0.0:.4f}"` both behave correctly.
- `_gate_ok` "Process exited with code" string match — exact match against `tools.py:174`.
- `_augment_references` regex `^\s*\d+\.\s` — only counts the 5 reference-number lines, not mid-line arXiv IDs.
- `_extract_code` prose heuristic — even if prose-containing-"for " escapes, the gate check catches the non-zero exit.
- `gather_literature` / `search_web` failure path — fully non-fatal; `literature=[]` propagates safely through all formatters.
- `_results_table_md` key access on `baseline`, `baseline-fallback`, `attempt` entries — all required keys set; optional keys use `.get()`.
- `history_tail` `e["plan"]` — all three entry kinds set `plan` explicitly.
- Paper-contract on total baseline failure — all four required fields have non-empty fallback strings.
- `PROPOSE_PROMPT`'s `<float:.4f>` in constraint text — outside braces; format parser treats as literal.

---
created: 2026-05-29
status: active
author: code-reviewer agent
session: main
branch: post-hackathon-working-state
informed_by: agents/paper-pushers/llm.py, judge.py, critic.py, my_run_agent.py, my_run_agent_b.py, my_run_agent_c.py, hackathon_science/utils.py, git diff HEAD~1
notes: Review of post-hackathon refactor that moves paper-pushers agents from AWS Bedrock to the direct Anthropic API via a new shared llm.py helper. Covers provider routing, temperature/system handling correctness, behavioral regressions, import safety, and dead imports.
---

# Code Review: Anthropic API wiring (post-hackathon refactor)

## Summary

The refactor introduces `llm.py` as a shared routing layer that dispatches to Anthropic SDK, OpenAI, or Bedrock based on model-id prefix, then replaces all per-file `call_llm` Bedrock calls in the four consumer files. The routing logic and per-provider argument handling are mostly correct. Two findings stand out: one dead import in `critic.py` (`Any`) and one pre-existing T=0 characteristic that is not a regression but is worth documenting. The import path mechanics work correctly for all three invocation modes.

---

## Findings

### 1. `Any` import left behind in `critic.py` (Nitpick)

**Location:** `agents/paper-pushers/critic.py`, line 18: `from typing import Any, Optional`

The old `_llm_call` used `inf_cfg: dict[str, Any]` as a type annotation. The new implementation drops that annotation; `Any` is not referenced anywhere else in the file. `Optional` is still used on line 34 (`temperature: Optional[float]`). The `Any` import is dead code.

**Fix:** Change `from typing import Any, Optional` to `from typing import Optional`.

---

### 2. T=0 → API-default behavior for judge pairwise calls (Suggestion — not a regression)

**Location:** `agents/paper-pushers/llm.py`, lines 54 and 73; `agents/paper-pushers/judge.py`, line 250

`llm.chat` omits `temperature` when `temperature is not None and temperature > 0` is False. `judge._pairwise_once` calls `_llm(..., temperature=0.0)`. Since `0.0 > 0` is False, temperature is silently omitted, and the Anthropic SDK uses its API default of 1.0 (stochastic), not 0.0 (deterministic).

**Is this a regression?** No. The old Bedrock `judge._llm` used the same guard: `if temperature and temperature > 0` (where `0.0` is falsy in Python). The condition evaluates identically, and Bedrock also omitted T for T=0 calls. Behavior is preserved across the rewrite.

**Should it be fixed anyway?** The comment in the old code was explicit: *"Opus 4.7 rejects temperature outright; for deterministic (T=0) calls we simply omit it."* That rationale was Bedrock-specific. On the Anthropic SDK, `temperature=0` is perfectly valid and would actually deliver deterministic output. If the intent is deterministic pairwise verdicts, the condition in `llm.chat` could be changed to `temperature is not None` (send T=0 through to Anthropic, omit only for Bedrock). But this is a deliberate tradeoff in the existing helper design, not a bug introduced by this diff.

---

### 3. Provider routing — correctness assessment (Correct)

**Location:** `agents/paper-pushers/llm.py`, lines 45–77

All three branches are correct:

- **Anthropic path (`claude-*`):** `system` sent as a string kwarg; `temperature` sent only when `> 0`. Both match the Anthropic SDK's `messages.create` contract. `max_tokens` is a required top-level kwarg — present. Response extraction via `b.text for b in resp.content if getattr(b, "type", "") == "text"` handles the `ContentBlock` list correctly.

- **OpenAI path (`gpt-/o1-/o3-`):** `system` is prepended as `{"role": "system", "content": [{"text": system}]}` in the messages list. `utils._call_openai_llm` converts content lists to plain strings and preserves the `role` field, so the system message survives the conversion as `{"role": "system", "content": "<text>"}` — which is valid for OpenAI's API. `temperature` is placed as a top-level kwarg (not nested in `inferenceConfig`); `utils._call_openai_llm` pops only `max_tokens` and `inferenceConfig` from kwargs before spreading the rest into `request_params`, so `temperature` passes through cleanly.

- **Bedrock path (anything else):** `system` sent as `[{"text": system}]` (Converse format); `temperature` nested inside `inferenceConfig` — omitted when T=0 to avoid the Opus 4.7 rejection. `utils._call_bedrock_llm` passes `inferenceConfig` through to the Converse call directly.

---

### 4. Behavioral regressions vs original helpers (Correct — with one nuance)

**`judge._llm`:** Old code: `if temperature and temperature > 0` to omit T. New code delegates to `llm.chat` which uses `temperature is not None and temperature > 0`. Both omit T=0 (see Finding 2). No functional regression.

**`critic._llm_call`:** Old code had a Bedrock-specific guard: `if temperature is not None and model == SONNET: inf_cfg["temperature"] = temperature` — this deliberately prevented temperature from being sent to Opus. New code sends temperature to `llm.chat` unconditionally; `llm.chat` then applies the T>0 guard uniformly.

For the two Opus call sites in `critic.py` (`panel_review` area chair and `revise_weakest_section`), neither passes a `temperature` argument, so `temperature` defaults to `None` at `_llm_call`. `llm.chat` receives `temperature=None`, the condition `temperature is not None and temperature > 0` is False, and T is omitted. The net behavior for Opus calls is unchanged.

For Sonnet calls at T=0.7 and T=0.5, the old code sent them (model == SONNET was True); the new code sends them too (T > 0). No regression.

**`my_run_agent._llm`:** Old code had explicit `try/except` around each attempt with per-attempt error printing. New code relies on `llm.chat`'s internal `except Exception` block which prints once. The retry-on-empty logic is functionally identical: both loop up to 2 times and return `""` if both attempts produce empty text.

**`my_run_agent_b._llm`:** Same as above. Subtle difference: old code only retried on exception (not on empty) in attempt 0 inside the exception handler. However, it also naturally looped on empty (the `if text.strip(): return text` path fell through without returning, so the loop continued). Net behavior is the same: up to 2 attempts, returns first non-empty, returns `""` if both empty.

---

### 5. Import / path safety (Correct)

**`sys.path.insert(0, ...)`:** All four consumer files do `sys.path.insert(0, str(Path(__file__).resolve().parent))` before importing from `llm`. `my_run_agent_b.py` uses `Path(__file__).parent` without `.resolve()` — functionally identical because `importlib.util.spec_from_file_location` in `runner.py` always provides an absolute path, making `resolve()` a no-op.

**Runner (importlib) path:** `hackathon_science/runner.py` uses `spec_from_file_location` with the full absolute path. The loaded module's `__file__` will be absolute, so `Path(__file__).resolve().parent` resolves to the `paper-pushers` directory. `from llm import ...` will find `llm.py` there.

**`uv run hackathon run` path:** Same as above — the runner is always invoked, same mechanism.

**`my_run_agent_c.py` import chain:** `_c` imports `my_run_agent_b`, which imports `my_run_agent`, which imports `llm`. It also imports `critic`, which independently imports `llm`. `llm.py` has only `os` and `sys` at module level; `anthropic` and `hackathon_science.utils.call_llm` are lazy inline imports. No circular import risk.

---

### 6. Bedrock fallback when LLM_STRONG_MODEL is set to a `global.anthropic.*` ID (Correct)

If the user sets `LLM_STRONG_MODEL=global.anthropic.claude-opus-4-7`, the model ID does not start with `claude-` or any OpenAI prefix, so it falls into the Bedrock branch. `llm.chat` will call `call_llm` with the Bedrock-correct argument shapes: `system=[{"text": system}]`, T nested in `inferenceConfig` and omitted when T=0. This matches the Opus 4.7 requirement documented in the helper.

---

### 7. OpenAI fallback when LLM_STRONG_MODEL is set to a `gpt-*` ID (Correct)

`temperature` is placed as a top-level kwarg, not inside `inferenceConfig`. `utils._call_openai_llm` pops `inferenceConfig` (discards it) and spreads remaining `kwargs` into `request_params`. Temperature therefore reaches `client.chat.completions.create()` as a top-level parameter, which is correct.

---

## Cleanup Report

- **Files reviewed:** 6 (llm.py, judge.py, critic.py, my_run_agent.py, my_run_agent_b.py, my_run_agent_c.py, plus hackathon_science/utils.py for contract verification)
- **Issues found:** 1 dead import (nitpick), 1 pre-existing T=0 characteristic (suggestion)
- **Changes made:** None — review-only per instructions
- **Compilation status:** Not run (no build step in this project type)
- **Deferred items:** None

---

## Positive Observations

- The `llm.chat` docstring is precise and accurately describes all three provider differences. The inline comments explaining why each branch does what it does are load-bearing — they will prevent future developers from "simplifying" away the Opus T-rejection workaround.
- Lazy imports for `anthropic` and `call_llm` inside `chat()` rather than at module level is the right call: it avoids importing a heavyweight SDK on every invocation of a file that might never use the Anthropic path, and it eliminates the risk of `ImportError` at module load time for environments that only have one SDK installed.
- The `_ANTHROPIC` module-level singleton with a lazy init pattern avoids creating a new client per call without requiring module-level side effects.
- The `from __future__ import annotations` in `llm.py` is appropriate given the `model: str | None` union syntax on Python 3.9.

---

## Recommendations (priority order)

1. **Fix the dead `Any` import in `critic.py`** — one-line change, no risk.
2. **Consider documenting the T=0 behavior in `llm.chat`** — add a comment at the guard condition explaining that T=0 is intentionally omitted even on the Anthropic path (to keep Bedrock Opus compatibility when the same guard is used for all providers). This prevents a future reviewer from "fixing" it and inadvertently breaking the Bedrock path.
3. No other actionable items. The refactor is behaviorally correct for the primary use case (Anthropic direct API) and the fallback paths (Bedrock, OpenAI).

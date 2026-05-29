"""Shared provider-aware LLM helper for the paper-pushers agents.

One `chat()` entry point, routed by model-id prefix:
  claude-*       -> Anthropic API (direct SDK; stable key, no Bedrock STS)
  gpt-/o1-/o3-   -> OpenAI            via hackathon_science.utils.call_llm
  anything else  -> AWS Bedrock       via call_llm (e.g. global.anthropic.* profiles)

Default models come from env, so a whole run can switch backend with no code
edits:
  LLM_STRONG_MODEL  (default claude-opus-4-7)    — area chair / finals / writing
  LLM_FAST_MODEL    (default claude-sonnet-4-6)   — reviewers / drafting / Stage-1

Background: during the hackathon every agent called AWS Bedrock with
`global.anthropic.*` inference profiles provisioned by the platform. Those STS
creds are gone post-hackathon, so the default here is the direct Anthropic API
(ANTHROPIC_API_KEY). To go back to Bedrock with live AWS creds:
  export LLM_STRONG_MODEL=global.anthropic.claude-opus-4-7
  export LLM_FAST_MODEL=global.anthropic.claude-sonnet-4-6
Or to OpenAI: export LLM_STRONG_MODEL=gpt-4o ; LLM_FAST_MODEL=gpt-4o-mini

Providers take `system`/`temperature` differently, which is the whole reason
this helper exists:
  - Anthropic: `system` is a string kwarg; `temperature` (0-1) native.
  - OpenAI (via call_llm): no `system` kwarg (errors) -> role=system message;
    `temperature` must be top-level (dropped if nested in inferenceConfig).
  - Bedrock (via call_llm): `system=[{text}]` top-level; Opus 4.7 rejects
    `temperature`, so only send it for T>0.
"""
from __future__ import annotations

import os
import sys

STRONG_MODEL = os.environ.get("LLM_STRONG_MODEL", "claude-opus-4-7")
FAST_MODEL = os.environ.get("LLM_FAST_MODEL", "claude-sonnet-4-6")

# Independent judge-model knob — overrides STRONG/FAST for critic.py + judge.py only.
# Default = same as the writer (single-family, current behaviour). Override to a
# DIFFERENT family (e.g. JUDGE_STRONG_MODEL=gpt-4o) to break the writer↔judge
# self-preference loop documented in docs/planning/generator-judge-loop-and-goodhart.md.
JUDGE_STRONG_MODEL = os.environ.get("JUDGE_STRONG_MODEL", STRONG_MODEL)
JUDGE_FAST_MODEL = os.environ.get("JUDGE_FAST_MODEL", FAST_MODEL)

_ANTHROPIC = None   # lazily-created anthropic.Anthropic() client


def chat(user: str, system: str = "", model: str | None = None,
         temperature: float | None = None, max_tokens: int = 4000) -> str:
    """Single-shot completion. Returns text, or '' on error/empty."""
    model = model or STRONG_MODEL
    try:
        if model.startswith("claude-"):
            import anthropic
            global _ANTHROPIC
            if _ANTHROPIC is None:
                _ANTHROPIC = anthropic.Anthropic()
            kw: dict = {"model": model, "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": user}]}
            if system:
                kw["system"] = system
            # Only send temperature when explicitly > 0. T=0 callers fall through
            # to the provider default — intentional: this single guard also keeps
            # the Bedrock branch safe (Opus 4.7 rejects `temperature` outright).
            # Don't "fix" this to always send temperature; it breaks Bedrock Opus.
            if temperature is not None and temperature > 0:
                kw["temperature"] = temperature
            resp = _ANTHROPIC.messages.create(**kw)
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

        from hackathon_science.utils import call_llm
        is_openai = model.startswith(("gpt-", "o1-", "o3-"))
        kwargs: dict = {"inferenceConfig": {"maxTokens": max_tokens}}
        if is_openai:
            messages = []
            if system:
                messages.append({"role": "system", "content": [{"text": system}]})
            messages.append({"role": "user", "content": [{"text": user}]})
            if temperature is not None and temperature > 0:
                kwargs["temperature"] = temperature
        else:
            messages = [{"role": "user", "content": [{"text": user}]}]
            if system:
                kwargs["system"] = [{"text": system}]
            if temperature is not None and temperature > 0:
                kwargs["inferenceConfig"]["temperature"] = temperature
        r = call_llm(messages=messages, model_id=model, **kwargs)
        content = r.get("output", {}).get("message", {}).get("content", [])
        return content[0].get("text", "") if content else ""
    except Exception as e:
        print(f"[llm.chat] error ({model}): {e}", file=sys.stderr)
        return ""

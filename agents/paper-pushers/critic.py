"""9-axis PoLL critic + section-level revision loop.

Mirrors the platform's reviewer panel exactly (5×A-E paper reviewers + F code + G extension)
using a small Panel of LLM Evaluators (PoLL) — 3 Sonnet rolls at T=0.7 plus a single Opus
area chair — per docs/research/llm-judge-calibration.md.

Designed to be used in two ways:
  1. Inside an agent's run() to drive a revision loop (panel_review + revise_weakest_section)
  2. As a standalone scorer for ranking many preprints (score_paper returns a single number)
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

from hackathon_science import Paper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat as _chat, STRONG_MODEL, FAST_MODEL   # noqa: E402

# Strong (area chair / revision) and fast (reviewer panel) tiers. Default to the
# direct Anthropic API; override the backend via LLM_STRONG_MODEL / LLM_FAST_MODEL
# env vars (Bedrock global.anthropic.* or OpenAI gpt-*). See llm.py.
OPUS = STRONG_MODEL
SONNET = FAST_MODEL


# --- LLM helpers --------------------------------------------------------

def _llm_call(user: str, model: str, system: str = "", temperature: Optional[float] = None,
              max_tokens: int = 4000) -> str:
    """Single provider-routed call (see llm.chat). Retries once on empty."""
    for attempt in range(2):
        text = _chat(user, system=system, model=model, temperature=temperature,
                     max_tokens=max_tokens)
        if text.strip():
            return text
    return ""


_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    m = _JSON_FENCE.search(text)
    if m:
        cand = m.group(1).strip()
    else:
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
        cand = text[start:end]
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return None


# --- Per-reviewer prompts ----------------------------------------------

PAPER_REVIEWER_SYSTEM = """You are a rigorous senior reviewer at a top ML venue (the Society-of-Agents panel for the Hackathon Scientific Discovery platform).

Be critical and cautious. The platform's known failure modes you should actively check for:
  - Hallucinated citations (verify every cite looks like a real paper)
  - Numbers in Results that don't appear in the appendix script's output
  - "Conclusions Here" or placeholder strings
  - Missing figures or tables
  - Overclaim of generality from a single configuration

Score on a 1-10 integer scale per axis. Be strict — 8+ is rare, 9-10 is exceptional only.
Anti-inflation rules:
  - No working experiment in appendix → Tech ≤ 5
  - No explicit prior-work positioning → Novelty ≤ 5
  - Vague impact claims without measurement → Significance ≤ 5
  - Missing figures/tables → Clarity ≤ 7
"""

PAPER_REVIEWER_RED_TEAM_SYSTEM = """You are an ADVERSARIAL red-team reviewer for the Society-of-Agents panel.

Your job is to find the worst weaknesses. LLM reviewers systematically underweight weaknesses (Ye et al. 2025); your role is to correct for that bias by being maximally skeptical.

Score on a 1-10 integer scale per axis. Aggressive anti-inflation rules:
  - No statistical significance reporting (CIs, p-values, multiple seeds) → Tech ≤ 6
  - Conceptual contribution only, no real new evidence → Novelty ≤ 4
  - Acknowledgement of failed attempts is rewarded; hiding them is penalized
  - If the abstract overclaims relative to the methods → Clarity ≤ 5
"""

PAPER_REVIEWER_USER = """Review the following paper. The Society-of-Agents panel scores on 4 paper-axes (Technical Quality, Novelty, Clarity, Significance), each 1-10. Provide all four scores plus Overall (weighted mean), Confidence (1-5), key weaknesses, and key questions.

PAPER (title, intro, methods, results, references, appendix):

{paper_text}

Respond ONLY in valid JSON, no commentary, no markdown fences:

{{
  "Technical_Quality": <int 1-10>,
  "Novelty": <int 1-10>,
  "Clarity": <int 1-10>,
  "Significance": <int 1-10>,
  "Overall": <float — weighted mean: 0.3*Tech + 0.25*Novelty + 0.2*Clarity + 0.25*Significance>,
  "Confidence": <int 1-5>,
  "Strengths": ["<bullet 1>", "<bullet 2>", "<bullet 3>"],
  "Weaknesses": ["<bullet 1>", "<bullet 2>", "<bullet 3>"],
  "Questions": ["<question 1>", "<question 2>"]
}}
"""

CODE_REVIEWER_SYSTEM = """You are the code reviewer (Reviewer F) on the Society-of-Agents panel. Score the appendix script on 4 code-axes (Code Tech Quality, Code Reproducibility, Code Correctness, Code-Paper Alignment), each 1-10. Be strict and adversarial.

Required for a passing review:
  - Executable Python, no syntax errors, no obvious runtime crashes
  - random.seed set explicitly
  - Library versions or stdlib-only declaration
  - Assertions/correctness gate before any reported metric
  - Single "METRIC <name>=<number>" line or equivalent — easy to parse
  - Numbers reported in the paper's Results section MUST appear in the script's expected output
  - Self-contained: no network calls, no external data
"""

CODE_REVIEWER_USER = """Review the code in the paper's APPENDIX against the paper's METHODS and RESULTS sections.

PAPER METHODS:
{methods_text}

PAPER RESULTS:
{results_text}

APPENDIX SCRIPT:
{script_text}

Respond ONLY in valid JSON:

{{
  "Code_Tech_Quality": <int 1-10>,
  "Code_Reproducibility": <int 1-10>,
  "Code_Correctness": <int 1-10>,
  "Code_Paper_Alignment": <int 1-10>,
  "Overall": <float — weighted mean: 0.25*Tech + 0.3*Repro + 0.25*Correctness + 0.2*Align>,
  "Weaknesses": ["<bullet 1>", "<bullet 2>"],
  "Misalignments": ["<specific number/claim in paper that does not appear in script output>", "..."]
}}
"""

EXTENSION_REVIEWER_SYSTEM = """You are the extension reviewer (Reviewer G) on the Society-of-Agents panel. Score how strong this paper is as an extension of a SPECIFIC reference paper.

For this hackathon the implied anchor is Flow-of-Options (Nair, Trase, Kim, ICML 2025, arxiv 2502.12929). A passing extension review requires:
  - The paper explicitly names Flow-of-Options as the reference (Introduction + Methods + References)
  - The paper identifies a specific limitation of Flow-of-Options to address (one of: metric dependency, data availability, residual method bias, walk-sampling inefficiency, module-specific issues)
  - The paper demonstrates concrete advancement OR contributes a substantive analysis (e.g., null-result reproduction with proper statistical rigor)
  - The contribution is methodological or empirical, NOT merely applicational ("we apply FoO to X" without new mechanism counts as weak)
"""

EXTENSION_REVIEWER_USER = """Review the paper as an EXTENSION of Flow-of-Options (Nair, Trase, Kim, ICML 2025, arxiv 2502.12929).

PAPER (title, intro, methods, results):
{paper_text}

Respond ONLY in valid JSON:

{{
  "Extension_Quality": <int 1-10>,
  "Overall": <int 1-10 — same as Extension_Quality for this single-axis review>,
  "FoO_Anchor_Strength": "<strong | weak | missing>",
  "Specific_Limitation_Addressed": "<which limitation; or 'none'>",
  "Substantive_Advance": "<yes | no — and a one-sentence justification>",
  "Weaknesses": ["<bullet 1>", "<bullet 2>"]
}}
"""


AREA_CHAIR_SYSTEM = """You are the Area Chair aggregating the panel's reviews into a single meta-review.

You will see five paper reviewers' JSON outputs, one code reviewer's JSON output, and one extension reviewer's JSON output. The reviewers were not allowed to communicate. Do NOT assume earlier reviewers were correct — weigh each on the strength of arguments cited.

A single well-substantiated low score can outweigh four shallow high scores. Strong language without specific evidence is a red flag.
"""

AREA_CHAIR_USER = """Aggregate the panel into a single meta-review.

PAPER reviews (5 independent rolls, presented in random order):
{paper_reviews}

CODE review (1 roll):
{code_review}

EXTENSION review (1 roll):
{extension_review}

The platform's combined score is approximately:
  Paper_mean ≈ mean(Overall) across the 5 paper reviewers
  Code_mean = code_overall
  Extension_mean = extension_overall
  Combined ≈ (Paper_mean * 5/7) + (Code_mean * 1/7) + (Extension_mean * 1/7)

Respond ONLY in valid JSON:

{{
  "Paper_mean": <float>,
  "Code_overall": <float>,
  "Extension_overall": <float>,
  "Combined": <float>,
  "Weakest_axis": "<one of: Technical_Quality, Novelty, Clarity, Significance, Code_Tech_Quality, Code_Reproducibility, Code_Correctness, Code_Paper_Alignment, Extension_Quality>",
  "Top_Weaknesses": ["<top weakness from across all reviewers>", "<2nd>", "<3rd>"],
  "Section_To_Revise": "<one of: title, introduction, methods, results, appendix>",
  "Revision_Guidance": "<2-3 sentence specific guidance for revising that section>"
}}
"""


# --- Core API -----------------------------------------------------------

def _paper_text(paper: Paper) -> str:
    return (
        f"TITLE: {paper.title}\n\n"
        f"INTRODUCTION:\n{paper.introduction}\n\n"
        f"METHODS:\n{paper.methods}\n\n"
        f"RESULTS:\n{paper.results}\n\n"
        f"REFERENCES:\n{paper.references}\n\n"
        f"APPENDIX:\n{paper.appendix}\n"
    )


def _extract_script(paper: Paper) -> str:
    """Pull the Python code out of the appendix (if any) for the code reviewer."""
    if not paper.appendix:
        return "(no appendix)"
    fence = re.search(r"```(?:python)?\s*\n(.*?)\n```", paper.appendix, re.DOTALL)
    return fence.group(1).strip() if fence else paper.appendix.strip()


def panel_review(paper: Paper, n_paper_reviewers: int = 5) -> dict:
    """Run a full 9-axis panel: 5 paper × A-E + 1 code × F + 1 extension × G, aggregated by an area chair.

    One of the 5 paper reviewers uses an adversarial red-team persona. The remaining 4 use the
    neutral Sakana-style prompt at T=0.7 (Sonnet) for diversity.
    """
    paper_txt = _paper_text(paper)
    script_txt = _extract_script(paper)

    print(f"[critic] running {n_paper_reviewers} paper reviewers...")
    paper_reviews: list[dict] = []
    for i in range(n_paper_reviewers):
        sys_prompt = PAPER_REVIEWER_RED_TEAM_SYSTEM if i == n_paper_reviewers - 1 else PAPER_REVIEWER_SYSTEM
        raw = _llm_call(
            PAPER_REVIEWER_USER.format(paper_text=paper_txt),
            model=SONNET, system=sys_prompt, temperature=0.7,
        )
        parsed = _extract_json(raw) or {"Overall": 5.0, "Weaknesses": ["(unparseable review)"], "raw": raw[:200]}
        paper_reviews.append(parsed)
        print(f"[critic]   paper-reviewer {i+1}: Overall={parsed.get('Overall', '?')}")

    print("[critic] running code reviewer...")
    code_raw = _llm_call(
        CODE_REVIEWER_USER.format(
            methods_text=paper.methods, results_text=paper.results, script_text=script_txt,
        ),
        model=SONNET, system=CODE_REVIEWER_SYSTEM, temperature=0.5,
    )
    code_review = _extract_json(code_raw) or {"Overall": 5.0, "Weaknesses": ["(unparseable code review)"], "raw": code_raw[:200]}
    print(f"[critic]   code: Overall={code_review.get('Overall', '?')}")

    print("[critic] running extension reviewer...")
    ext_raw = _llm_call(
        EXTENSION_REVIEWER_USER.format(paper_text=paper_txt),
        model=SONNET, system=EXTENSION_REVIEWER_SYSTEM, temperature=0.5,
    )
    ext_review = _extract_json(ext_raw) or {"Overall": 5.0, "Weaknesses": ["(unparseable extension review)"], "raw": ext_raw[:200]}
    print(f"[critic]   extension: Overall={ext_review.get('Overall', '?')}")

    # Shuffle paper-review order so the area chair sees them in random order (anti-sycophancy)
    shuffled = list(paper_reviews)
    random.shuffle(shuffled)
    paper_reviews_text = "\n".join(f"[Reviewer {chr(65+i)}]: {json.dumps(r)}" for i, r in enumerate(shuffled))

    print("[critic] running area chair...")
    chair_raw = _llm_call(
        AREA_CHAIR_USER.format(
            paper_reviews=paper_reviews_text,
            code_review=json.dumps(code_review),
            extension_review=json.dumps(ext_review),
        ),
        model=OPUS, system=AREA_CHAIR_SYSTEM,
    )
    chair = _extract_json(chair_raw) or {}

    # Compute fallback aggregates if the chair couldn't parse
    paper_overalls = [r.get("Overall", 5.0) for r in paper_reviews if isinstance(r.get("Overall"), (int, float))]
    paper_mean = sum(paper_overalls) / len(paper_overalls) if paper_overalls else 5.0
    code_overall = code_review.get("Overall", 5.0) if isinstance(code_review.get("Overall"), (int, float)) else 5.0
    ext_overall = ext_review.get("Overall", 5.0) if isinstance(ext_review.get("Overall"), (int, float)) else 5.0
    combined = paper_mean * (5/7) + code_overall * (1/7) + ext_overall * (1/7)

    return {
        "paper_reviews": paper_reviews,
        "code_review": code_review,
        "extension_review": ext_review,
        "area_chair": chair,
        "paper_mean": chair.get("Paper_mean", paper_mean),
        "code_overall": chair.get("Code_overall", code_overall),
        "extension_overall": chair.get("Extension_overall", ext_overall),
        "combined": chair.get("Combined", combined),
        "weakest_axis": chair.get("Weakest_axis", "Novelty"),
        "section_to_revise": chair.get("Section_To_Revise", "introduction"),
        "revision_guidance": chair.get("Revision_Guidance", "Strengthen the weakest section."),
        "top_weaknesses": chair.get("Top_Weaknesses", []),
    }


# --- Section revision --------------------------------------------------

REVISE_SECTION_SYSTEM = """You are revising a single section of a research paper to address a peer reviewer's specific weaknesses. Stay faithful to the experiment numbers and the FoO extension framing. Do NOT invent new results. Do NOT fabricate citations. If a weakness asks for something the paper cannot provide (e.g., a missing experiment), acknowledge it honestly instead of fabricating."""

REVISE_SECTION_USER = """Revise the following section of the paper. The reviewer panel identified these issues:

PANEL'S TOP WEAKNESSES (across the whole paper):
{top_weaknesses}

PANEL'S SPECIFIC GUIDANCE FOR THIS SECTION:
{guidance}

CURRENT SECTION TEXT ({section_name}):
{section_text}

CONTEXT — the rest of the paper for consistency:
TITLE: {title}
INTRODUCTION (current): {introduction_summary}
METHODS (current): {methods_summary}
RESULTS (current): {results_summary}

Output ONLY the revised {section_name} section text. No markdown fences. No "## Header" lines (the platform handles those). Length should be similar to the current section.

At the END of your output, append a single line:
ADDRESSED: <comma-separated list of the weaknesses you addressed; or 'NONE' if you couldn't address them>"""


def revise_weakest_section(paper: Paper, review: dict) -> tuple[Paper, str]:
    """Regenerate the section the area chair flagged with the panel's weaknesses as context.

    Returns (revised_paper, addressed_log_line).
    """
    section = review["section_to_revise"].lower()
    if section not in ("title", "introduction", "methods", "results", "appendix"):
        section = "introduction"

    current_text = getattr(paper, section, "") or ""
    if not current_text:
        return paper, f"[skip] {section} is empty"

    raw = _llm_call(
        REVISE_SECTION_USER.format(
            section_name=section,
            section_text=current_text[:8000],
            guidance=review.get("revision_guidance", ""),
            top_weaknesses="\n- ".join([""] + review.get("top_weaknesses", [])),
            title=paper.title,
            introduction_summary=paper.introduction[:600],
            methods_summary=paper.methods[:600],
            results_summary=paper.results[:600],
        ),
        model=OPUS, system=REVISE_SECTION_SYSTEM, max_tokens=6000,
    )

    if not raw.strip():
        return paper, f"[skip] revision LLM returned empty"

    # Extract the "ADDRESSED:" line for the audit log
    lines = raw.strip().splitlines()
    addressed_line = ""
    body_lines = lines
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("ADDRESSED:"):
            addressed_line = lines[i].strip()
            body_lines = lines[:i]
            break
    body = "\n".join(body_lines).strip()

    # Replace the section
    new_kwargs = {
        "title": paper.title,
        "introduction": paper.introduction,
        "methods": paper.methods,
        "results": paper.results,
        "references": paper.references,
        "appendix": paper.appendix,
        "tags": list(paper.tags),
    }
    new_kwargs[section] = body
    return Paper(**new_kwargs), f"{section}: {addressed_line or 'ADDRESSED: NONE'}"


def critic_loop(paper: Paper, max_rounds: int = 3,
                stop_overall: float = 6.5, stop_min_axis: float = 5.0) -> tuple[Paper, list[dict]]:
    """Repeatedly review + revise until the area-chair combined score crosses stop_overall AND
    no individual axis drops below stop_min_axis, or until max_rounds is exhausted.

    Returns the (best paper seen, history of reviews).
    """
    history: list[dict] = []
    best_paper = paper
    best_score = -1.0

    for round_idx in range(max_rounds):
        print(f"[critic_loop] round {round_idx + 1}/{max_rounds}")
        review = panel_review(paper)
        history.append({
            "round": round_idx,
            "combined": review["combined"],
            "paper_mean": review["paper_mean"],
            "code_overall": review["code_overall"],
            "extension_overall": review["extension_overall"],
            "weakest_axis": review["weakest_axis"],
            "section_to_revise": review["section_to_revise"],
            "top_weaknesses": review["top_weaknesses"],
        })
        print(f"[critic_loop]   combined={review['combined']:.2f} "
              f"(paper={review['paper_mean']:.2f}, code={review['code_overall']:.2f}, "
              f"ext={review['extension_overall']:.2f}); revising {review['section_to_revise']}")

        if review["combined"] > best_score:
            best_score = review["combined"]
            best_paper = paper

        # Stopping criterion
        all_axes = [
            review["paper_mean"], review["code_overall"], review["extension_overall"],
        ]
        if review["combined"] >= stop_overall and min(all_axes) >= stop_min_axis:
            print(f"[critic_loop]   stopping — combined={review['combined']:.2f} ≥ {stop_overall}")
            break

        if round_idx == max_rounds - 1:
            break  # don't revise on the last round

        paper, addressed = revise_weakest_section(paper, review)
        print(f"[critic_loop]   {addressed[:200]}")

    # Final review of the best version doesn't help — we already scored it. Return best seen.
    if best_score > review.get("combined", -1):
        return best_paper, history
    return paper, history


def score_paper(paper: Paper) -> float:
    """Standalone single-number score for ranking (Phase D top-10 use case)."""
    review = panel_review(paper, n_paper_reviewers=3)  # cheaper for ranking
    return review["combined"]

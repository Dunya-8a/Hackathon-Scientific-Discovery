"""Loop B v0 — selection-without-mutation evolutionary loop over Phase B.

Generates N candidate papers via independent Phase B runs (each picks its own
DAG walk → different framing/methods/ablation/results/discussion combo), scores
each with the 9-axis PoLL panel TWICE (once same-family, once cross-family), and
returns the highest-scoring candidate plus the Goodhart-delta between the two
judges.

This is the minimum loop-B from docs/planning/generator-judge-loop-and-goodhart.md:
guards #1 (cross-family judge — done via JUDGE_STRONG_MODEL) and *implicit* #4
(diversity via independent walks). Guards #2 (held-out anchor) and #3 (real stop
conditions) are NOT here yet — this is selection-only over one generation.

Usage:
    LLM_STRONG_MODEL=claude-opus-4-7 LLM_FAST_MODEL=claude-sonnet-4-6 \\
    JUDGE_STRONG_MODEL=gpt-4o       JUDGE_FAST_MODEL=gpt-4o-mini      \\
    uv run python agents/paper-pushers/loop_b.py --n 3 --out docs/temp/loop_b_runs/
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _generate_one(agent_module_name: str, problem_domain: str) -> Any | None:
    """Import the agent module fresh (so its module-state is per-candidate
    rather than shared) and call run(). Returns the Paper, or None on error."""
    # Reload fresh — the agent uses random sampling for the DAG walk; we want
    # independent draws across candidates.
    if agent_module_name in sys.modules:
        importlib.reload(sys.modules[agent_module_name])
    mod = importlib.import_module(agent_module_name)
    try:
        return mod.run(problem_domain)
    except Exception:
        traceback.print_exc()
        return None


def _score_with_env(paper, judge_strong: str, judge_fast: str,
                    n_reviewers: int = 3) -> dict:
    """Re-import critic.py under the requested JUDGE_*_MODEL env so the
    panel runs with that family. Returns the panel_review dict."""
    os.environ["JUDGE_STRONG_MODEL"] = judge_strong
    os.environ["JUDGE_FAST_MODEL"] = judge_fast
    # llm.py reads env at import; reload it then critic.
    for m in ("llm", "critic"):
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    from critic import panel_review  # noqa: E402
    return panel_review(paper, n_paper_reviewers=n_reviewers)


def _paper_to_dict(paper) -> dict:
    if is_dataclass(paper):
        return asdict(paper)
    return {
        "title": getattr(paper, "title", ""),
        "introduction": getattr(paper, "introduction", ""),
        "methods": getattr(paper, "methods", ""),
        "results": getattr(paper, "results", ""),
        "references": getattr(paper, "references", ""),
        "appendix": getattr(paper, "appendix", ""),
        "tags": list(getattr(paper, "tags", [])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="my_run_agent_b",
                    help="Agent module name under agents/paper-pushers/")
    ap.add_argument("--n", type=int, default=3, help="Candidates per generation")
    ap.add_argument("--reviewers", type=int, default=3,
                    help="Paper reviewers per panel call (1-5)")
    ap.add_argument(
        "--problem-domain",
        default="Agentic design and algorithms for scientific hypothesis generation / falsification",
    )
    ap.add_argument("--writer-strong", default="claude-opus-4-7")
    ap.add_argument("--writer-fast", default="claude-sonnet-4-6")
    ap.add_argument("--judge-strong", default="gpt-4o")
    ap.add_argument("--judge-fast", default="gpt-4o-mini")
    ap.add_argument("--skip-same-family", action="store_true",
                    help="Skip the same-family judge pass (cheaper; loses Goodhart signal)")
    ap.add_argument("--out", default="docs/temp/loop_b_runs",
                    help="Output dir; one timestamped subdir per loop_b run")
    args = ap.parse_args()

    # Writer env (Phase B reads STRONG_MODEL at import)
    os.environ["LLM_STRONG_MODEL"] = args.writer_strong
    os.environ["LLM_FAST_MODEL"] = args.writer_fast

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(args.out) / f"loop_b_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[loop_b] writing run artifacts to {out_dir}")

    # --- Phase 1: generate N candidates ---
    candidates: list[dict] = []
    for i in range(args.n):
        print(f"\n[loop_b] === generation 1, candidate {i + 1}/{args.n} ===")
        paper = _generate_one(args.agent, args.problem_domain)
        if paper is None:
            print(f"[loop_b]   candidate {i + 1} failed to generate")
            continue
        rec = {
            "i": i,
            "paper": _paper_to_dict(paper),
            "_obj": paper,  # kept locally; stripped before JSON dump
        }
        candidates.append(rec)
        # Persist immediately so a crash mid-run doesn't lose everything
        (out_dir / f"candidate_{i:02d}.md").write_text(
            f"# {paper.title}\n\n"
            f"## Introduction\n{paper.introduction}\n\n"
            f"## Methods\n{paper.methods}\n\n"
            f"## Results\n{paper.results}\n\n"
            f"## References\n{paper.references}\n\n"
            f"## Appendix\n{paper.appendix}\n"
        )

    if not candidates:
        print("[loop_b] zero candidates generated — aborting")
        return

    # --- Phase 2: score each cross-family AND (optionally) same-family ---
    for rec in candidates:
        paper = rec.pop("_obj")
        print(f"\n[loop_b] === scoring candidate {rec['i'] + 1} cross-family "
              f"(judge: {args.judge_strong} / {args.judge_fast}) ===")
        rec["cross_family"] = _summarize(_score_with_env(
            paper, args.judge_strong, args.judge_fast, args.reviewers,
        ))

        if not args.skip_same_family:
            print(f"[loop_b] === scoring candidate {rec['i'] + 1} same-family "
                  f"(judge: {args.writer_strong} / {args.writer_fast}) — "
                  f"Goodhart signal ===")
            rec["same_family"] = _summarize(_score_with_env(
                paper, args.writer_strong, args.writer_fast, args.reviewers,
            ))
            rec["goodhart_delta"] = (
                rec["same_family"]["combined"] - rec["cross_family"]["combined"]
            )
        else:
            rec["same_family"] = None
            rec["goodhart_delta"] = None

    # --- Phase 3: pick winner ---
    candidates.sort(key=lambda r: r["cross_family"]["combined"], reverse=True)
    winner = candidates[0]

    print("\n[loop_b] === final ranking (by cross-family combined) ===")
    for r in candidates:
        cf = r["cross_family"]
        sf = r.get("same_family") or {}
        sf_score = sf.get("combined", "—")
        delta = r.get("goodhart_delta")
        delta_str = f"Δ={delta:+.2f}" if delta is not None else "Δ=—"
        print(f"  {r['i']:>2} | xf={cf['combined']:.2f} "
              f"(p={cf['paper_mean']:.2f} c={cf['code_overall']:.2f} "
              f"e={cf['extension_overall']:.2f}) | "
              f"sf={sf_score if isinstance(sf_score, str) else f'{sf_score:.2f}'} "
              f"| {delta_str} | {r['paper']['title'][:60]!r}")

    # Persist full result
    summary = {
        "run_id": run_id,
        "agent": args.agent,
        "n": args.n,
        "writer": {"strong": args.writer_strong, "fast": args.writer_fast},
        "judge": {"strong": args.judge_strong, "fast": args.judge_fast},
        "skip_same_family": args.skip_same_family,
        "candidates": candidates,
        "winner_index": winner["i"],
    }
    (out_dir / "loop_b_result.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[loop_b] wrote {out_dir}/loop_b_result.json")
    print(f"[loop_b] winner: candidate {winner['i']} "
          f"(cross-family combined {winner['cross_family']['combined']:.2f}) — "
          f"{winner['paper']['title']!r}")


def _summarize(review: dict) -> dict:
    """Strip the verbose per-reviewer JSONs and keep the load-bearing numbers."""
    return {
        "combined": float(review.get("combined", 0.0)),
        "paper_mean": float(review.get("paper_mean", 0.0)),
        "code_overall": float(review.get("code_overall", 0.0)),
        "extension_overall": float(review.get("extension_overall", 0.0)),
        "weakest_axis": review.get("weakest_axis"),
        "top_weaknesses": review.get("top_weaknesses", []),
    }


if __name__ == "__main__":
    main()

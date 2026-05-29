"""Paper-pushers run agent — Phase C.

Phase B (FoO-DAG over paper artifacts) + a 9-axis PoLL critic and section-level revision loop.

The pipeline:
  Phase 0  Autoresearch experiment on script.py (from Phase A/B)
  Phase 1  Build a 3×5 FoO-DAG of paper-drafting decisions
  Phase 2  Sample and score 6 walks; pick the best
  Phase 3  Render the winning walk into a draft paper
  Phase 4  Run the simulated reviewer panel; revise the weakest section; repeat
           (max 3 rounds; stop when combined ≥ 6.5 AND min axis ≥ 5.0)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from hackathon_science import Paper

# Phase B is the basis — import its building blocks directly.
sys.path.insert(0, str(Path(__file__).parent))
from my_run_agent_b import (                                            # noqa: E402
    autoresearch_loop,
    build_paper_dag,
    sample_walks,
    score_walk,
    render_final_paper,
    DEPTH_NAMES,
    N_WALKS,
    K_OPTIONS,
    FOO_ANCHOR,
    FOO_LIMITATIONS,
    WORKING_DIR,
)
from critic import critic_loop                                          # noqa: E402
from my_run_agent import archive_paper                                  # noqa: E402


def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    print(f"[run] Phase C — FoO-DAG + 9-axis PoLL critic + revision loop")
    print(f"[run] problem_domain: {problem_domain!r}")
    print(f"[run] anchor: {FOO_ANCHOR}")
    print(f"[run] limitation: {FOO_LIMITATIONS[3]}")

    # Phase 0 — autoresearch experiment
    print("[run] Phase 0: autoresearch on script.py")
    exp_result = autoresearch_loop(k_attempts=3)
    if exp_result["best"]:
        print(f"[run]   best collision_rate: {exp_result['best'][0]:.4f}")

    # Phase 1 — FoO-DAG
    print(f"[run] Phase 1: building FoO-DAG ({K_OPTIONS} × {len(DEPTH_NAMES)} depths)")
    dag = build_paper_dag(exp_result)

    # Phase 2 — sample + score walks
    walks = sample_walks(dag, n_walks=N_WALKS)
    print(f"[run] Phase 2: scoring {len(walks)} walks")
    scored: list[tuple[float, tuple[int, ...], dict]] = []
    for w in walks:
        s, d = score_walk(w, dag, exp_result)
        scored.append((s, w, d))
        print(f"[run]   walk {w}: Overall={s:.2f}")
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_walk, _ = scored[0]
    print(f"[run]   BEST walk {best_walk}: Overall={best_score:.2f}")

    # Phase 3 — render the winning walk into a draft paper
    print(f"[run] Phase 3: rendering best walk")
    draft = render_final_paper(best_walk, dag, exp_result)
    print(f"[run]   draft title: {draft.title!r}")

    # Phase 4 — PoLL critic + section-level revision loop (max 3 rounds)
    print(f"[run] Phase 4: PoLL critic + revision loop")
    revised, history = critic_loop(draft, max_rounds=3,
                                   stop_overall=6.5, stop_min_axis=5.0)
    print(f"[run]   revision history: {len(history)} rounds")
    for h in history:
        print(f"[run]     round {h['round']}: combined={h['combined']:.2f} "
              f"(paper={h['paper_mean']:.2f}, code={h['code_overall']:.2f}, "
              f"ext={h['extension_overall']:.2f}); revised: {h['section_to_revise']}")

    print(f"[run] final title: {revised.title!r}")

    archive_path = archive_paper(revised)
    if archive_path:
        print(f"[run] archived to {archive_path}")

    return revised

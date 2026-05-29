"""Validate critic.panel_review against the real Round-3 reviewer panel.

Inputs:
  docs/temp/round3_labels.json (from extract_round3_labels.py)
  docs/temp/ecosystem-papers/<id>.md (one per Round-3 paper)

For each of the 18 Round-3 papers, run critic.panel_review with a chosen
JUDGE backend, then compare our predicted scores to the real ones along
five axes:

  - paper_overall: our paper_mean vs real mean(A-E Overall)
  - code_overall:  our code_overall vs real F Overall
  - extension:     our extension_overall vs real G Overall
  - combined:      our combined vs real combined (5/7 paper + 1/7 code + 1/7 ext)
  - novelty axis (since that's where b4aef3a3 lost — single-axis sanity check)

Output: docs/temp/judge-vs-round3-<judge>.json + a printed correlation report.

Defaults: judge=gpt-4o-mini (cheap, ~$1-2 total for all 18 papers at n=3).
For a Goodhart sanity check, run twice: once judge=claude (same-family
as the writers) and once judge=gpt-4o-mini, then diff the correlations.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = ROOT / "docs/temp/round3_labels.json"
PAPERS_DIR = ROOT / "docs/temp/ecosystem-papers"

sys.path.insert(0, str(ROOT / "agents/paper-pushers"))


# --- Paper parser (ecosystem-papers/ format) ---------------------------

SECTION_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)


def _parse_paper_md(path: Path):
    """Build a Paper out of an ecosystem .md file.

    Format (from `hackathon download-papers`):
        # Title
        **ID:** ...
        **Author:** ...
        **Date:** ...
        **Tags:** ...
        ---
        ## Abstract / Introduction / Methods / Results / References / Appendix
    """
    from hackathon_science import Paper  # noqa: E402

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ") and not ln.startswith("## ")), path.stem)

    sections: dict[str, str] = {}
    matches = list(SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    return Paper(
        title=title[:300],
        introduction=(sections.get("introduction") or sections.get("abstract") or "")[:8000],
        methods=sections.get("methods", "")[:8000],
        results=sections.get("results", "")[:8000],
        references=sections.get("references", "")[:6000],
        appendix=sections.get("appendix", "")[:10000],
        id=path.stem,
        tags=[],
    )


# --- Correlation helpers (stdlib only) ---------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys)
             if isinstance(x, (int, float)) and isinstance(y, (int, float))
             and not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return None
    xs2, ys2 = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = mean(xs2), mean(ys2)
    sx2 = sum((x - mx) ** 2 for x in xs2)
    sy2 = sum((y - my) ** 2 for y in ys2)
    if sx2 == 0 or sy2 == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    return sxy / math.sqrt(sx2 * sy2)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys)
             if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if len(pairs) < 3:
        return None
    xs2, ys2 = [p[0] for p in pairs], [p[1] for p in pairs]

    def _rank(vs: list[float]) -> list[float]:
        sorted_idx = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[sorted_idx[j + 1]] == vs[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks

    return _pearson(_rank(xs2), _rank(ys2))


# --- Audit loop --------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-strong", default="gpt-4o-mini",
                    help="JUDGE_STRONG_MODEL (area chair / extension reviewer / strong tier)")
    ap.add_argument("--judge-fast", default="gpt-4o-mini",
                    help="JUDGE_FAST_MODEL (paper-reviewer panel)")
    ap.add_argument("--reviewers", type=int, default=3,
                    help="Paper reviewers per panel call (1-5). 3 is enough for correlation.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only score the first N labeled papers (for cheap dry-run)")
    args = ap.parse_args()

    # Set JUDGE env BEFORE importing critic
    os.environ["JUDGE_STRONG_MODEL"] = args.judge_strong
    os.environ["JUDGE_FAST_MODEL"] = args.judge_fast
    for m in ("llm", "critic"):
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    from critic import panel_review  # noqa: E402

    labels = json.loads(LABELS_FILE.read_text())
    if args.limit:
        labels = labels[: args.limit]

    out_path = ROOT / f"docs/temp/judge-vs-round3-{args.judge_strong}.json"

    print(f"[audit] judge_strong={args.judge_strong} judge_fast={args.judge_fast} "
          f"n_reviewers={args.reviewers} papers={len(labels)}")
    print(f"[audit] writing → {out_path}\n")

    rows: list[dict] = []
    t0 = time.time()
    for i, label in enumerate(labels, 1):
        pid = label["id"]
        path = PAPERS_DIR / f"{pid}.md"
        if not path.exists():
            print(f"[audit] [{i}/{len(labels)}] {pid} — MISSING file, skipping")
            continue
        paper = _parse_paper_md(path)

        t1 = time.time()
        try:
            review = panel_review(paper, n_paper_reviewers=args.reviewers)
        except Exception as e:
            print(f"[audit] [{i}/{len(labels)}] {pid} — panel_review crashed: {e}")
            continue

        rec = {
            "id": pid,
            "title": (label.get("title") or "")[:120],
            "team": label.get("team"),
            "real": {
                "paper_overall": label["paper_axis_means"].get("overall"),
                "novelty": label["paper_axis_means"].get("novelty"),
                "clarity": label["paper_axis_means"].get("clarity"),
                "code_overall": label["code_axes"].get("overall"),
                "extension": label["extension"],
                "combined": label["combined"],
            },
            "predicted": {
                "paper_mean": float(review.get("paper_mean", 0.0)),
                "code_overall": float(review.get("code_overall", 0.0)),
                "extension_overall": float(review.get("extension_overall", 0.0)),
                "combined": float(review.get("combined", 0.0)),
                "weakest_axis": review.get("weakest_axis"),
            },
            "duration_s": round(time.time() - t1, 1),
        }
        rows.append(rec)
        elapsed = time.time() - t0
        print(f"[audit] [{i:>2}/{len(labels)}] {pid} "
              f"real_combined={rec['real']['combined']:>4.2f} "
              f"pred_combined={rec['predicted']['combined']:>4.2f} "
              f"Δ={rec['predicted']['combined'] - rec['real']['combined']:+.2f} "
              f"({rec['duration_s']}s, elapsed {elapsed:.0f}s)")

        # Incremental persist
        out_path.write_text(json.dumps({"meta": {"judge_strong": args.judge_strong,
                                                  "judge_fast": args.judge_fast,
                                                  "n_reviewers": args.reviewers,
                                                  "n_papers_scored": len(rows)},
                                          "rows": rows},
                                         indent=2))

    if not rows:
        print("[audit] no rows scored; nothing to correlate")
        return

    # Correlations
    print("\n[audit] === correlations (pred vs real) ===")
    pairs = {
        "paper_overall":  ([r["predicted"]["paper_mean"]        for r in rows],
                           [r["real"]["paper_overall"]          for r in rows]),
        "code_overall":   ([r["predicted"]["code_overall"]      for r in rows],
                           [r["real"]["code_overall"]           for r in rows]),
        "extension":      ([r["predicted"]["extension_overall"] for r in rows],
                           [r["real"]["extension"]              for r in rows]),
        "combined":       ([r["predicted"]["combined"]          for r in rows],
                           [r["real"]["combined"]               for r in rows]),
    }
    print(f"{'axis':<18} {'pearson':>10} {'spearman':>10}  {'n':>5}")
    for axis, (pred, real) in pairs.items():
        n = sum(1 for x in real if x is not None)
        pr = _pearson(pred, real)
        sr = _spearman(pred, real)
        prs = f"{pr:.3f}" if pr is not None else "  —  "
        srs = f"{sr:.3f}" if sr is not None else "  —  "
        print(f"{axis:<18} {prs:>10} {srs:>10}  {n:>5}")

    # Save final
    out_path.write_text(json.dumps({"meta": {"judge_strong": args.judge_strong,
                                              "judge_fast": args.judge_fast,
                                              "n_reviewers": args.reviewers,
                                              "n_papers_scored": len(rows)},
                                      "rows": rows,
                                      "correlations": {
                                          axis: {"pearson": _pearson(pred, real),
                                                 "spearman": _spearman(pred, real),
                                                 "n": sum(1 for x in real if x is not None)}
                                          for axis, (pred, real) in pairs.items()
                                      }},
                                     indent=2))
    print(f"\n[audit] wrote {out_path}")


if __name__ == "__main__":
    main()

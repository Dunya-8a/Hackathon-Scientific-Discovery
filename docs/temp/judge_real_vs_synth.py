"""Rank the real-FoO paper against synthetic FoO papers with a cross-family (GPT)
judge. Goodhart defense: writer is Claude, judge is GPT (JUDGE_*_MODEL env).
Forces use_cache=False so no prior Claude-judged Stage-1 scores leak in.

Run:
  JUDGE_STRONG_MODEL=gpt-4o JUDGE_FAST_MODEL=gpt-4o-mini \
    .venv/bin/python docs/temp/judge_real_vs_synth.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "agents/paper-pushers")
import judge  # noqa: E402  (reads JUDGE_*_MODEL at import — set env BEFORE running)

PAPERS = Path("agents/paper-pushers/papers")

# stub -> human label. 1 real + 4 synthetic.
WANT = {
    "20260530-182037": "REAL-FoO (vs self-consistency, hard MATH)",
    "20260530-002009": "synthetic: metric_dependency",
    "20260530-014810": "synthetic: data_availability",
    "20260529-022544": "synthetic: birthday-floor walk sampling",
    "20260529-133103": "synthetic: choosing-a-diversity-metric",
}


def load(stub: str) -> SimpleNamespace:
    f = next(PAPERS.glob(stub + "*.json"))
    d = json.loads(f.read_text())
    return SimpleNamespace(
        id=stub, title=d.get("title", ""), introduction=d.get("introduction", ""),
        methods=d.get("methods", ""), results=d.get("results", ""),
        references=d.get("references", ""), appendix=d.get("appendix", ""),
    )


def main():
    print(f"judge models: FAST={judge.SONNET}  STRONG={judge.OPUS}")
    papers = [load(s) for s in WANT]

    # Stage 1 — pointwise (no cache, so GPT actually scores).
    scored = []
    for p in papers:
        text = judge._paper_text(p)
        rec = judge._stage1_score_one(text, judge._content_sha(text), use_cache=False)
        scored.append({"paper_id": p.id, "title": p.title, "text": text,
                       "stage1_score": rec["stage1_score"]})
        print(f"  stage1 {rec['stage1_score']:.2f}  {WANT[p.id]}")
    scored.sort(key=lambda d: d["stage1_score"], reverse=True)

    # Stage 2 — Arena-Lite tournament + Bradley-Terry.
    wins = judge._run_tournament(scored)
    strengths = judge.bradley_terry_mle(wins)
    unc = judge._bootstrap_uncertainty(wins)
    ranked = [{
        "label": WANT[c["paper_id"]],
        "bt_strength": round(float(strengths[i]), 4),
        "bt_unc": round(float(unc[i]), 4),
        "stage1": round(c["stage1_score"], 4),
    } for i, c in enumerate(scored)]
    ranked.sort(key=lambda d: d["bt_strength"], reverse=True)

    print("\n=== FINAL RANKING (cross-family GPT judge) ===")
    for i, r in enumerate(ranked, 1):
        print(f"{i}. {r['label']:48s} BT={r['bt_strength']:+.3f}±{r['bt_unc']:.3f}  stage1={r['stage1']:.2f}")
    tag = "gpt" if judge.OPUS.startswith(("gpt", "o1", "o3")) else "claude"
    out = Path(f"docs/temp/judge-real-vs-synth-{tag}.json")
    out.write_text(json.dumps(ranked, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

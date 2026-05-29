"""Parse ui/public/reviews/round-3.html → per-paper labeled scores.

The HTML has one tab per paper containing 3 tables:
  - Paper Reviews A-E (5 reviewers × Tech, Novelty, Clarity, Significance, Overall)
  - Code Review F      (1 reviewer × Code Tech, Code Repro, Code Correct, Code-Paper Align, Overall)
  - Extension Review G (1 reviewer × Extension Quality, Overall)

We emit one record per paper with:
  - id, team, title
  - paper_mean per axis (mean across A-E) and the 5 raw rows
  - code_axes (single row from F)
  - extension (single value from G)
  - combined = mean(paper_overall_mean, code_overall, extension_overall) per the leaderboard

Output: docs/temp/round3_labels.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

HTML = Path(__file__).resolve().parents[2] / "ui/public/reviews/round-3.html"
OUT = Path(__file__).parent / "round3_labels.json"

# --- Paper tab boundaries ----------------------------------------------

TAB_HEADER_RE = re.compile(
    r'<h2[^>]*class="card-title"[^>]*>\s*([\w\-]+) - ([0-9a-f]{8}): ([^<]+)</h2>'
)
TAB_START_RE = re.compile(r'<div class="tab-content"[^>]*id="paper-([0-9a-f]{8})"')


def _split_tabs(html: str) -> list[tuple[str, str]]:
    """Split the html by paper tabs. Return [(paper_id, tab_html), ...]."""
    starts = [(m.start(), m.group(1)) for m in TAB_START_RE.finditer(html)]
    tabs = []
    for i, (offset, pid) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(html)
        tabs.append((pid, html[offset:end]))
    return tabs


# --- Table-row extraction ----------------------------------------------

TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL)
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
CELL_RE = re.compile(r"<t[dh][^>]*>(?:<[^>]+>)?([^<]+)", re.IGNORECASE)


def _cells(row_html: str) -> list[str]:
    return [m.group(1).strip() for m in CELL_RE.finditer(row_html)]


def _try_float(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_paper_scores(tab_html: str) -> dict:
    """Pull all numeric score rows out of a tab's <table>s.

    Returns:
      {
        "paper_reviewers": [{reviewer, tech, novelty, clarity, sig, overall}, ...],
        "code": {code_tech, code_repro, code_correct, code_align, overall},
        "extension": {extension_quality, overall},
      }
    """
    out: dict = {"paper_reviewers": [], "code": None, "extension": None}
    seen_headers: list[list[str]] = []

    for tbl in TABLE_RE.findall(tab_html):
        rows = ROW_RE.findall(tbl)
        if not rows:
            continue
        header_cells = _cells(rows[0])
        if not header_cells or "Reviewer" not in header_cells[0]:
            continue

        normalized = [h.lower().strip() for h in header_cells]
        seen_headers.append(normalized)

        # Branch on the header shape
        if "technical quality" in normalized:
            # Paper reviewers table (A-E + mean/median/stddev tail rows)
            for row in rows[1:]:
                cells = _cells(row)
                if not cells:
                    continue
                rname = cells[0].strip()
                if rname.upper() not in {"A", "B", "C", "D", "E"}:
                    continue  # skip the Mean/Median/Std Dev tail
                if len(cells) < 6:
                    continue
                out["paper_reviewers"].append({
                    "reviewer": rname.upper(),
                    "tech": _try_float(cells[1]),
                    "novelty": _try_float(cells[2]),
                    "clarity": _try_float(cells[3]),
                    "significance": _try_float(cells[4]),
                    "overall": _try_float(cells[5]),
                })

        elif "code tech quality" in normalized:
            # Code reviewer F — single data row
            for row in rows[1:]:
                cells = _cells(row)
                if not cells or len(cells) < 6:
                    continue
                if cells[0].strip().upper() != "F":
                    continue
                out["code"] = {
                    "reviewer": "F",
                    "code_tech": _try_float(cells[1]),
                    "code_repro": _try_float(cells[2]),
                    "code_correct": _try_float(cells[3]),
                    "code_align": _try_float(cells[4]),
                    "overall": _try_float(cells[5]),
                }

        elif "extension quality" in normalized:
            for row in rows[1:]:
                cells = _cells(row)
                if not cells or len(cells) < 3:
                    continue
                if cells[0].strip().upper() != "G":
                    continue
                out["extension"] = {
                    "reviewer": "G",
                    "extension_quality": _try_float(cells[1]),
                    "overall": _try_float(cells[2]),
                }

    return out


def _summarize(record: dict) -> dict:
    paper = record["scores"]["paper_reviewers"]
    code = record["scores"]["code"] or {}
    ext = record["scores"]["extension"] or {}

    def _meancol(col: str) -> float | None:
        vals = [r[col] for r in paper if isinstance(r.get(col), (int, float))]
        return round(mean(vals), 3) if vals else None

    means = {ax: _meancol(ax) for ax in ("tech", "novelty", "clarity", "significance", "overall")}
    paper_mean_overall = means["overall"]
    code_overall = code.get("overall")
    ext_overall = ext.get("overall")

    combined = None
    components = [v for v in (paper_mean_overall, code_overall, ext_overall) if v is not None]
    if components:
        # Same weighting the leaderboard uses: 5/7 paper + 1/7 code + 1/7 extension
        weights = []
        weighted = 0.0
        if paper_mean_overall is not None:
            weights.append(5)
            weighted += paper_mean_overall * 5
        if code_overall is not None:
            weights.append(1)
            weighted += code_overall * 1
        if ext_overall is not None:
            weights.append(1)
            weighted += ext_overall * 1
        combined = round(weighted / sum(weights), 3)

    record["paper_axis_means"] = means
    record["code_axes"] = {k: code.get(k) for k in
                           ("code_tech", "code_repro", "code_correct", "code_align", "overall")}
    record["extension"] = ext.get("extension_quality")
    record["combined"] = combined
    return record


def main() -> None:
    html = HTML.read_text()
    records: list[dict] = []

    # Pull (paper_id, team, title) trios from card titles, then merge with tab content.
    # The card title appears INSIDE the tab content for that paper.
    title_by_id: dict[str, dict] = {}
    for m in TAB_HEADER_RE.finditer(html):
        title_by_id[m.group(2)] = {"team": m.group(1).strip(),
                                   "title": m.group(3).strip()}

    for pid, tab in _split_tabs(html):
        meta = title_by_id.get(pid, {})
        rec = {
            "id": pid,
            "team": meta.get("team"),
            "title": meta.get("title"),
            "scores": _parse_paper_scores(tab),
        }
        records.append(_summarize(rec))

    # Sort by combined descending
    records.sort(key=lambda r: (r["combined"] is None, -(r["combined"] or 0.0)))

    OUT.write_text(json.dumps(records, indent=2))
    print(f"[extract] {len(records)} papers parsed → {OUT}")
    print(f"[extract] {sum(1 for r in records if r['combined'] is not None)} have combined scores")
    print()
    print(f"{'rank':<5} {'paper':<10} {'team':<18} {'paper':>6} {'code':>5} {'ext':>5} {'comb':>6}  title")
    for i, r in enumerate(records, 1):
        pm = r["paper_axis_means"].get("overall")
        co = r["code_axes"].get("overall")
        eo = r["extension"]
        cb = r["combined"]
        print(f"{i:<5} {r['id']:<10} {(r['team'] or '?')[:18]:<18} "
              f"{pm if pm is not None else '   —':>6} "
              f"{co if co is not None else '  —':>5} "
              f"{eo if eo is not None else '  —':>5} "
              f"{cb if cb is not None else '   —':>6}  {(r['title'] or '')[:60]}")


if __name__ == "__main__":
    main()

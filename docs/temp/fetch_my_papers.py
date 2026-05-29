"""Fetch this team's papers (full bodies) from the ecosystem and write them in
the frontmatter format hackathon_science.git_ops.load_papers expects, so the
judge can rank them.

`download-papers` only exports the abstract; the /papers/{id} endpoint (via
CloudClient.get_paper) returns the full Methods/Results/References/Appendix.

Output: docs/temp/my-papers/papers/<id>.md  (deduped by content hash)
Uses the cloud API key (~/.hackathon-science/credentials), NOT AWS/Bedrock.
"""
import hashlib
from pathlib import Path

import yaml

from hackathon_science.cloud_client import CloudClient

OUT = Path(__file__).parent / "my-papers" / "papers"
OUT.mkdir(parents=True, exist_ok=True)

c = CloudClient()
me = c.me()
team = me["team_id"]
print(f"team: {team}")

summaries = c.list_papers(team_id=team)
print(f"team papers listed: {len(summaries)}")

seen_sha: dict[str, str] = {}   # content sha -> id already written
written = 0
for s in summaries:
    pid = s.get("id") or s.get("paper_id")
    if not pid:
        continue
    p = c.get_paper(pid)
    intro = p.get("introduction", "") or ""
    methods = p.get("methods", "") or ""
    results = p.get("results", "") or ""
    refs = p.get("references", "") or ""
    appendix = p.get("appendix", "") or ""

    sha = hashlib.sha256((p.get("title", "") + intro + methods + results).encode()).hexdigest()[:16]
    if sha in seen_sha:
        print(f"  skip {pid} (identical content to {seen_sha[sha]})")
        continue
    seen_sha[sha] = pid

    front = {
        "id": pid,
        "title": p.get("title", "") or "(untitled)",
        "author": p.get("author", "") or "",
        "date": str(p.get("date", "") or ""),
        "introduction": intro,
        "tags": p.get("tags", []) or [],
    }
    body = "\n".join([
        "---",
        yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip(),
        "---", "",
        "## Methods", methods, "",
        "## Results", results, "",
        "## References", refs, "",
        "## Appendix", appendix, "",
    ])
    (OUT / f"{pid}.md").write_text(body, encoding="utf-8")
    written += 1

print(f"wrote {written} unique papers to {OUT} ({len(summaries) - written} dupes skipped)")

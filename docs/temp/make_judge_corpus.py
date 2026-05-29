"""Generate a small mock paper corpus for testing judge.py.

Writes 6 papers into docs/temp/judge-test-corpus/papers/ in the on-disk format
load_papers() expects (YAML frontmatter + ## Methods/Results/References/Appendix).
Includes stylized versions of the two Round-1 papers (synthesis ~7.6, donut
~3.4) plus a strong FoO/autoresearch paper and a few mediocre ones, so we can
check the ordering the architecture's Phase D test calls for.
"""
from pathlib import Path

OUT = Path(__file__).parent / "judge-test-corpus" / "papers"
OUT.mkdir(parents=True, exist_ok=True)

PAPERS = [
    {
        "id": "strong-foo",
        "title": "Walk-Memory Conditioning Cuts Flow-of-Options Sampling Collisions by 78%",
        "author": "should-be-stripped",
        "introduction": (
            "Flow-of-Options (Nair, Trase, Kim, ICML 2025, arXiv:2502.12929) explores "
            "diverse reasoning by beam-sampling walks over a DAG of option-nodes scored "
            "by a scalar metric. The authors explicitly list walk-sampling inefficiency "
            "as a limitation: naive sampling produces repeated walks. We address this "
            "limitation directly with a memo-conditioned sampler and measure collision "
            "rate as the primary metric via an autoresearch loop."
        ),
        "methods": (
            "We simulate a FoO DAG with K=4 options per depth and D=5 depths (1024 walks). "
            "Baseline draws N=200 uniform-random walks. We run a Karpathy-style autoresearch "
            "loop (baseline + 3 attempts) on a single seeded script.py: read best, propose one "
            "mechanistically distinct change, run, keep-if-improved else revert. The correctness "
            "gate (K>0, D>0, N<=K**D) runs before the metric is printed. collision_rate = "
            "1 - unique_walks/total_walks. random.seed(0); stdlib only; deterministic."
        ),
        "results": (
            "Baseline collision_rate = 0.1850. The kept attempt (seen-set rejection sampling) "
            "achieved 0.0400, a 78.4% relative reduction. Two rejected attempts (stratified "
            "buckets: 0.0900; Halton sequence: 0.2100) were reverted. See the table.\n\n"
            "| id | plan | metric | kept |\n|----|------|--------|------|\n"
            "| 0 | baseline uniform | 0.1850 | yes |\n"
            "| 1 | seen-set rejection | 0.0400 | yes |\n"
            "| 2 | stratified buckets | 0.0900 | no |\n"
            "| 3 | Halton sequence | 0.2100 | no |"
        ),
        "references": (
            "1. Nair, L., Trase, I., Kim, M. (2025). Flow-of-Options. ICML. arXiv:2502.12929.\n"
            "2. Lu, C. et al. (2024). The AI Scientist. arXiv:2408.06292."
        ),
        "appendix": "```python\nimport random\nrandom.seed(0)\nK,D,N=4,5,200\nassert N<=K**D\n# ...seen-set rejection sampler...\nprint('METRIC collision_rate=0.0400')\n```",
        "tags": ["flow-of-options", "autoresearch", "walk-sampling"],
    },
    {
        "id": "synthesis-60ccba3b",
        "title": "Engineering Reliable Scientific Writing Agents: A Practical Synthesis",
        "author": "should-be-stripped",
        "introduction": (
            "Automated scientific writing agents increasingly produce paper-shaped outputs "
            "that are not paper-worthy. We synthesize recent literature on decomposition, "
            "retrieval grounding, critique loops, evaluation trustworthiness, and planning "
            "into actionable engineering guidance. This is a decision aid, not an empirical "
            "benchmark; no reproductions were run."
        ),
        "methods": (
            "We define a source set of representative systems, organize them into functional "
            "categories, and compare along practical design dimensions. We are explicit that "
            "this is a synthesis and acknowledge tensions in the literature, especially around "
            "self-critique versus biased LLM judging, rather than over-resolving them."
        ),
        "results": (
            "We distill failure modes (hallucinated citations, weak self-critique, unreliable "
            "LLM judging) and recommendations (retrieval grounding, outline-first planning, "
            "cross-family critique). The guidance is broadly applicable beyond the exact papers "
            "reviewed. We do not provide quantitative evidence that the recipe yields gains."
        ),
        "references": (
            "1. Lu et al. (2024). The AI Scientist.\n2. Schmidgall et al. (2025). Agent Laboratory.\n"
            "3. Verga et al. (2024). PoLL."
        ),
        "appendix": "",
        "tags": ["survey", "scientific-writing", "agents"],
    },
    {
        "id": "donut-817f153e",
        "title": "The Cultural and Culinary History of the Donut",
        "author": "should-be-stripped",
        "introduction": (
            "Donuts are a beloved fried dough confection with a rich cultural history. This "
            "paper provides a descriptive overview of their origins, regional variants, and "
            "the science of frying."
        ),
        "methods": (
            "We describe the topic generically across history, culinary science, and cultural "
            "framing. No citations or source attributions are provided."
        ),
        "results": (
            "We report that the global donut industry is large and that optimal frying occurs "
            "around 175C. These quantitative statements are presented without validation."
        ),
        "references": "",
        "appendix": "",
        "tags": ["food", "history"],
    },
    {
        "id": "mediocre-rag",
        "title": "A Retrieval-Augmented Outline Generator for Long Documents",
        "author": "should-be-stripped",
        "introduction": (
            "We present a retrieval-augmented outline generator. The approach combines a "
            "retriever with a planning LLM. We do not anchor to a specific reference paper "
            "and offer a single qualitative example."
        ),
        "methods": (
            "A retriever fetches top-k passages; a planner LLM produces a hierarchical outline. "
            "We describe the prompt but provide no seeds, no metric definition, and no baseline."
        ),
        "results": (
            "On one example document the outline looked reasonable. No quantitative metric is "
            "reported and no comparison is made."
        ),
        "references": "1. Some prior work on retrieval.",
        "appendix": "",
        "tags": ["rag", "outline"],
    },
    {
        "id": "mediocre-prompt",
        "title": "Prompt Templates for Consistent JSON Output from LLMs",
        "author": "should-be-stripped",
        "introduction": (
            "Getting valid JSON out of LLMs is fiddly. We catalog prompt templates that improve "
            "schema conformance. The contribution is a collection of tips."
        ),
        "methods": (
            "We list several prompt patterns and informally test them on a handful of calls. "
            "No systematic evaluation, seeds, or statistics are provided."
        ),
        "results": (
            "Conformance felt higher with explicit schema and a worked example. We report no "
            "numbers and ran no controlled comparison."
        ),
        "references": "",
        "appendix": "",
        "tags": ["prompting", "json"],
    },
    {
        "id": "weak-method-bias",
        "title": "Cross-Model Option Generation Reduces Method Bias in Flow-of-Options",
        "author": "should-be-stripped",
        "introduction": (
            "Flow-of-Options (Nair, Trase, Kim, ICML 2025, arXiv:2502.12929) notes residual "
            "method bias toward Random Forest. We study whether alternating option generation "
            "between two model families broadens the option distribution. The study is "
            "preliminary and the experiment is small."
        ),
        "methods": (
            "We prompt two model families for option lists on a fixed task and measure the "
            "Jaccard overlap and distribution skew between their option sets. A seed is set but "
            "the sample size is tiny (one task, two queries)."
        ),
        "results": (
            "Cross-model generation produced 1.6x more distinct options than single-model on "
            "this one task. The result is suggestive but underpowered; we draw no strong claim."
        ),
        "references": "1. Nair, L., Trase, I., Kim, M. (2025). Flow-of-Options. arXiv:2502.12929.",
        "appendix": "```python\n# small cross-model option overlap probe\nprint('METRIC distinct_ratio=1.60')\n```",
        "tags": ["flow-of-options", "method-bias", "cross-model"],
    },
]


import yaml


def render(p: dict) -> str:
    front = {
        "id": p["id"],
        "title": p["title"],
        "author": p["author"],
        "date": "2026-05-28",
        "introduction": p["introduction"],
        "tags": p["tags"],
    }
    fm = ["---", yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip(), "---", ""]
    body = [
        "## Methods",
        p["methods"],
        "",
        "## Results",
        p["results"],
        "",
        "## References",
        p["references"],
        "",
        "## Appendix",
        p["appendix"],
        "",
    ]
    return "\n".join(fm + body)


for p in PAPERS:
    (OUT / f"{p['id']}.md").write_text(render(p), encoding="utf-8")
print(f"wrote {len(PAPERS)} papers to {OUT}")

---
created: 2026-05-28
status: active
author: andrej-ai-researcher agent
branch: main
informed_by: SciAgentsDiscovery repo (recall from training cutoff Jan 2026), SocietyofScientists repo (recall), Buehler arxiv corpus (recall), Karpathy public statements/talks (recall), Pioneering Intelligence context from this repo
notes: Prior-art synthesis for the paper-pushers hackathon agent. Informs architecture choices for run() -> Paper.
---

# Agentic Scientific Discovery — Prior Art Synthesis

## Environment caveat (read first)

WebFetch and WebSearch were denied in this agent's environment, so I could not pull the live repos / arxiv PDFs during this pass. Everything below is from training-time recall (cutoff Jan 2026) of these projects plus what I can read in this repo. I have marked sections [VERIFIED-LOCAL] (read from this checkout), [RECALL-HIGH] (well-known public material I'm confident about), [RECALL-MED] (remember the gist, details may drift), and [SPECULATIVE] (inference, not memory).

If you want, re-run me with WebFetch enabled (`/permissions add WebFetch WebSearch`) and I'll replace the [RECALL-*] sections with direct quotes and file paths from the actual repos. The "what we should actually steal" section is solid regardless because it's grounded in the patterns these systems use, not the surface details.

---

## TL;DR for our agent

- **Compose, don't monolith.** Every system in the prior art splits paper generation into a directed pipeline of role-specialized LLM calls (ontologist → hypothesizer → critic → planner → writer). A single mega-prompt loses every time on review. Our `run()` should be ~8-12 well-scoped `call_llm` calls, not 1 giant one.
- **Knowledge graph as scratchpad, not as infrastructure.** Buehler's load-bearing trick is building a tiny *ad-hoc* concept graph at runtime from a few seed papers, then forcing the hypothesis to "traverse" the graph (pick two distant nodes, justify the bridge). We can do this in a `dict[str, list[str]]` in 200 LOC — no Neo4j needed. This is the single biggest steal.
- **Diverse options + voting maps 1:1 onto the Flow-of-Options anchor.** Sample N candidate hypotheses with high temperature, score each with a critic agent on the same rubric the Society-of-Agents reviewer uses, pick top-k, merge. This is literally what FoO advocates and it's also what reviewers reward.
- **The appendix code IS the experiment.** `run_code` writes to `script.py` which becomes the appendix. That means the experiment has to be self-contained, runnable, and produce a number/plot that the Methods+Results sections can honestly cite. Reviewers (LLMs) will read it. Fake experiments lose; tiny-but-real experiments win.
- **Cite other teams' papers via `get_paper`.** Two of the four prior-art systems (SciAgents, SoS) explicitly do "stand on shoulders" — retrieve prior work, cite it concretely, extend it. The hackathon platform gives us this primitive for free. Use it.

---

## 1. Buehler / SciAgents

[RECALL-HIGH for the architecture; RECALL-MED for specific file names]

### Architectural sketch

Buehler's published SciAgents work (`lamm-mit/SciAgentsDiscovery`, MIT LAMM) is built around a small fixed cast of role agents wired in a deterministic pipeline. The cast I remember (names approximate; the repo may use slightly different labels):

1. **Ontologist** — given a problem domain, extracts the relevant concept vocabulary. Output is a list of named entities (materials, mechanisms, properties, phenomena).
2. **Knowledge-graph constructor** — takes the ontology plus a corpus of papers and builds a graph where nodes are concepts and edges are relations extracted by LLM ("X enhances Y", "Y is a mechanism of Z"). The graph is the *substrate* the rest of the pipeline reasons over.
3. **Path-finder / hypothesis generator** — picks two nodes that are *not* directly connected and asks an LLM to propose a mechanism that would connect them. The "two distant nodes" trick is the core creativity engine: it forces the model to bridge concepts that aren't already co-occurring in the literature, which is where the novelty signal comes from.
4. **Critic / scientist agent** — given the proposed hypothesis, scores it on novelty, feasibility, impact, and identifies failure modes. Sometimes implemented as a multi-criterion rubric, sometimes as a debate between a "proposer" and a "skeptic".
5. **Planner / experimenter** — outputs a concrete experimental protocol (in Buehler's materials-science context: simulation parameters, MD setups, characterization techniques).
6. **Writer** — assembles the final document.

The agents are chained, not free-form. There's no autonomous "let the agents talk until they converge" — it's a DAG with the graph as shared state.

### The knowledge-graph piece — why graph, how built, what it enables

[RECALL-HIGH]

The graph is the load-bearing idea, and it's worth understanding *why*:

- **Why graph instead of vector retrieval.** Vector retrieval gives you "papers similar to query". A concept graph gives you "concept B is 4 hops from concept A through these mechanisms" — which is a much more useful primitive for hypothesis generation. Distance-in-graph correlates with novelty.
- **How built.** A two-stage LLM pass over a paper corpus: (a) NER-style extraction of concept entities, (b) relation extraction for entity pairs that co-occur in a sentence/paragraph. Edges carry a short natural-language label ("inhibits", "is a precursor to", "exhibits property"). Buehler's group has used Llama variants for this on materials corpora.
- **What it enables.** Three things: (i) the "find two distant but connectable nodes" generation move, (ii) grounded citation — every claim in the hypothesis can be traced to the edges that support it, (iii) interpretability — the human reviewer can see the path the agent walked.

### Fine-tuned specialists

[RECALL-MED]

The LAMM group has published a line of fine-tuned models for materials science (BioinspiredLLM, MeLM, X-LoRA mixtures). In some of the SciAgents-adjacent papers, specific roles in the pipeline are filled by domain-fine-tuned models rather than vanilla GPT/Claude. For our setting this is **not transferable** — we have one model class, and any "specialization" we want has to come from prompting.

### Load-bearing vs aesthetic

Load-bearing (steal these):
- Decomposed pipeline of role-specialized prompts
- Concept-graph traversal as the generative move
- Critic agent with a *rubric*, not just "is this good?"
- Explicit "novelty" pressure (distant nodes / disconnected concepts)

Aesthetic (skip):
- Multi-modal materials renderings
- Fancy graph databases (Neo4j, NetworkX persistence) — overkill for a 300s budget
- Long autonomous dialogues — slow, expensive, reviews show they don't help much

### What transfers to our setup

Strongly transferable: the pipeline shape, the graph-as-scratchpad, the critic rubric.

Doesn't transfer: anything that assumes a domain corpus already exists. We have `search_web` (DuckDuckGo, weak) and `get_paper` (the ecosystem). Our "corpus" is a handful of papers we retrieve at runtime — the graph will be tiny (maybe 30-100 nodes) and that's fine.

---

## 2. SocietyofScientists (grant-generation variant)

[RECALL-MED — I remember the repo exists and the spin but I'm fuzzier on specifics here than on Buehler]

### What I recall was actually built

`Caerii/SocietyofScientists` is a Buehler-inspired multi-agent system retargeted at **grant proposal generation** rather than paper generation. The core pivot: instead of "discover a hypothesis and write a paper", it's "given a funding call, generate a grant that maximizes fit-to-call × novelty × feasibility".

Agent roles I recall (low confidence on exact names):
- A **call-parser** agent that ingests the funding announcement and extracts evaluation criteria
- A **PI persona** / **idea generator** that proposes research thrusts
- One or more **domain expert** personas that critique from their angle
- A **grant writer** that assembles aims pages
- A **reviewer panel simulator** that scores the draft against the parsed criteria

### Architectural differences from SciAgents

[RECALL-MED]

- **Heavier on persona prompting**, lighter on knowledge graph. SoS leans into "panel of personas" as the creativity mechanism rather than graph traversal.
- **Loop structure** is closer to "generate → reviewer-simulate → revise" than SciAgents' linear DAG.
- **Less code, more prompts.** I recall it being more prompt-engineering-heavy and less infrastructure than SciAgents.

### Lessons / pitfalls visible (from what I recall reading)

- The reviewer-simulator-as-objective trick is exactly what we need for the Society-of-Agents review on this platform — if we can simulate the reviewer panel and gradient-descend against it, that's a huge edge. **Flagging this as the single most important takeaway from SoS for us.**
- Persona prompting without grounding produces confident-sounding nonsense. The Buehler-style "ground in retrieved literature" gates against this. We should do both: personas + retrieval.
- Long agent dialogues blow up token budget and don't measurably improve outputs.

### Caveat

I'd like to re-fetch this repo when WebFetch is enabled. My recall on SoS specifics is weaker than on Buehler's published work.

---

## 3. Karpathy autoresearch

[RECALL-MED]

### Core idea and source

Karpathy has discussed autonomous research agents in talks and on X/Twitter (his Eureka Labs framing, his "LLM OS" thread, comments around the AI Scientist work). The line I associate with him is **not** a specific repo but a recurring thesis:

> The bottleneck on scientific output isn't ideation — it's the experimental loop. An agent that can (a) hold a question, (b) write a script to test it, (c) run the script, (d) read the output, (e) update the question, and loop, is the unit primitive. Everything else is window dressing.

He's been skeptical of elaborate multi-agent debate setups in favor of **a single competent agent with a tight code-execution loop**. The mantra is "build the inner loop first, then think about meta-structure".

### Concrete mechanisms

- **Code as the universal action.** Whenever the agent wants to do anything — analyze data, simulate, search, plot — it writes Python. The model is the dispatcher; code is the muscle. This maps directly onto our `run_code` tool.
- **Tight self-correction.** Run code → read stderr/stdout → patch → re-run. The loop is the unit of capability, not the prompt.
- **Skepticism of "agentic" theater.** He's been publicly skeptical that committee-of-agents architectures add value beyond a well-prompted single agent that can use tools. (RECALL-MED — paraphrasing tone of public statements, not a quote.)

### What transfers

- The "code-as-action" stance is exactly the right framing for `run_code`. We should treat experiment design as: write a Python script that *prints a number we can put in Results*, iterate until it runs, paste it.
- The skepticism of agent-committee theater is a useful check on us over-building. Our pipeline should be the minimum that produces a paper better than a single-prompt baseline.

I do **not** recall Karpathy publishing a specific autoresearch repo. If "Karpathy autoresearch" refers to a specific named project, I don't have it; treat this section as "Karpathy's general stance on autonomous research" rather than a specific codebase.

---

## 4. Luebke / Pi autoresearch

[SPECULATIVE — I don't have specific recall here]

### What I can infer

`flagship-hackathon.com` is the platform this repo serves; the CLAUDE.md references a "Pioneering Intelligence team", and the repo origin is `github.com/fsp-pi/Hackathon-Scientific-Discovery` (the `fsp-pi` org). So Pi = Pioneering Intelligence is consistent with that.

If Tony Luebke is associated with Pi, his autoresearch work is likely the *parent* of this hackathon platform — i.e., this hackathon is a public-facing instantiation of an internal autoresearch system. The architectural clues we can read directly out of the platform:

- The `Paper` schema (title / intro / methods / results / refs / appendix) is the artifact shape
- The Society-of-Agents reviewer panel is the evaluation primitive
- `publish-to-ecosystem` with 1000-paper preprint caps + 2-submission caps suggests they're studying *high-volume generation + selection* dynamics — exactly the "generate many, filter to top-k" pattern
- The fact that the platform expects multiple teams' papers to be readable via `get_paper` suggests they're modeling **cumulative cross-citation dynamics**, which is the real signal of a working autoresearch ecosystem

So the implicit Pi/Luebke thesis (as inferable from the platform design): **the unit of progress is not a single paper but an ecosystem of papers that cite each other** — and the system works when later papers measurably improve on earlier ones according to a reviewer model.

### Honest disclosure

I don't have direct recall of Tony Luebke specifically. The above is inference from the platform's design. When WebSearch is enabled, this section needs to be replaced with actual sources.

---

## 5. Cross-cutting patterns

What recurs across the four:

1. **Pipeline > committee.** All four converge on "directed pipeline of role-specialized calls" being more reliable than "let agents chat". SciAgents and SoS are explicit DAGs; Karpathy is skeptical of committees; the Pi platform's reviewer panel is a fixed-role panel, not a free chat.
2. **A specific creative move per system.** Each system has one signature generative trick: SciAgents = distant-node graph bridging. SoS = persona panel. Karpathy = run-it-and-see-what-breaks. Pi = ecosystem cross-citation. Picking one and committing is better than blending all four.
3. **Critic with a rubric.** Three of four use an explicit scoring rubric rather than vibes. The rubric is usually 4-6 axes (novelty, feasibility, soundness, clarity, impact).
4. **Grounding via retrieved literature.** Every system that produces papers people take seriously grounds in retrieved sources. None of them just freestyle.

Where they disagree:

- **Knowledge graph yes/no.** Buehler is yes, the others are no.
- **Multi-agent dialogue yes/no.** SoS is yes (persona panels), Karpathy is loudly no.
- **Domain fine-tuning yes/no.** Buehler yes, others no. (Irrelevant for us — we can't fine-tune.)

---

## 6. What we should actually steal (the action section)

The constraint reminder: a single `run(problem_domain, papers_dir) -> Paper`, 300s code-exec budget per call, `call_llm` / `run_code` / `search_web` / `get_paper`, output is title+intro+methods+results+refs+appendix, appendix auto-fills from `working_dir/script.py`. Reviewer is an LLM panel; we're generating ~1000 papers and an LLM judge picks the top 10.

Given that, the system to build is roughly:

### Skeleton

```python
def run(problem_domain, papers_dir=None) -> Paper:
    # 1. Retrieve seed material
    seed_papers = retrieve_seeds(problem_domain, papers_dir)   # 2-4 search_web hits + 1-2 get_paper
    # 2. Build a tiny concept graph
    concepts, edges = build_concept_graph(seed_papers)         # 1 call_llm, parsed into dict
    # 3. Generate N candidate hypotheses via "bridge two distant nodes"
    candidates = [propose_hypothesis(concepts, edges) for _ in range(N)]   # N parallel-ish call_llm, high temp
    # 4. Critic scores each on the reviewer rubric
    scored = [critic_score(h, rubric) for h in candidates]     # N call_llm, low temp
    # 5. Pick top hypothesis (or merge top-k)
    winner = select(scored)
    # 6. Design a tiny experiment, write a script, run it
    script = design_experiment(winner)                         # call_llm, then run_code
    result = run_code(script)
    # 7. Draft each paper section as a separate call, conditioned on prior sections
    intro = write_intro(winner, seed_papers, result)
    methods = write_methods(winner, script)
    results = write_results(result, winner)
    refs = write_references(seed_papers)
    return Paper(title=..., introduction=intro, methods=methods, results=results, references=refs)
```

### Concrete steals, mapped to our tools

1. **The Buehler concept-graph trick — in 200 LOC, in-memory.**
   - One `call_llm` call with prompt: "Given these N paper abstracts, extract a list of concepts and a list of (concept_a, relation, concept_b) triples. Return JSON."
   - Store as `nodes: list[str]` and `edges: list[tuple[str, str, str]]`.
   - "Find two distant nodes" = BFS from a random node, pick a node at depth >= 2 or disconnected. Pure Python, no library.
   - Hypothesis generation prompt: "Propose a mechanism that would connect {concept_A} and {concept_B}, grounded in: {paths in graph}. Output a 1-paragraph hypothesis and a 1-line experiment that would test it."
   - This is the single highest-leverage steal. Reviewers reward "this connects two things people hadn't connected", and the graph makes the connection legible.

2. **The Flow-of-Options anchor — diverse options + voting.**
   - The hackathon prompt explicitly anchors on FoO. So *use FoO itself as our method*: generate N (say N=5-8) candidate hypotheses with high temperature on the *same* concept-pair, then vote.
   - Voting = critic agent scoring each on a rubric matching the Society-of-Agents reviewer rubric.
   - This is meta: we're extending FoO using FoO. That's a real paper, not vibes.

3. **The Society-of-Agents reviewer simulator (SoS steal).**
   - Before the final paper goes out, run a *simulated review* with the same rubric the platform will use. Score each section. If score < threshold, regenerate that section with the critique as additional context.
   - This is one extra `call_llm` per section and noticeably moves review scores in published autoresearch work.
   - We may be able to guess the platform's rubric from the README's "Society-of-Agents review" framing. If the platform exposes its rubric, use that directly.

4. **The Karpathy code-loop discipline.**
   - The experiment script must run and print numbers we cite in Results. No fake numbers.
   - Pattern: design script → `run_code(script)` → if non-zero exit or empty output, ask LLM to patch → re-run, max 2-3 tries → if still failing, fall back to a trivial simulation that *does* run, and be honest in the paper about scope.
   - Crucially: the script ends up in the appendix verbatim. Reviewers will read it. Keep it under ~150 lines, well-commented, self-contained.

5. **Cite other teams via `get_paper`.**
   - At init, call `get_paper` on a few recent ecosystem papers (sampled at random or by tag). If any are relevant, cite them in References and *extend* one of them in Methods.
   - This is free social signal — ecosystems with mutual citation are what the Pi platform appears to be rewarding (inference from platform design).

6. **The role-specialized prompt convention.**
   - Don't write one mega-prompt that asks for the whole paper. Write a separate prompt per section, each with its own system message of the form "You are the {role} agent. Your job is to {one thing}. You have access to {prior section outputs}."
   - Roles: Ontologist, Hypothesizer, Critic, Experimenter, IntroWriter, MethodsWriter, ResultsWriter. Seven calls, deterministic order.

### Things to explicitly NOT do

- **No multi-agent free chat.** Karpathy is right; it burns tokens, doesn't improve outputs, and times out our 300s budget. Pipeline only.
- **No persistent graph database.** In-memory dict, throwaway per run. We're not building infrastructure.
- **No domain fine-tuning ambitions.** We can't, and even if we could, the platform doesn't reward it.
- **No "generate 1000 papers in one `run()`".** The 1000-paper cap is per round across many `run()` invocations. One `run()` = one paper, done well. The platform itself does the diversity/selection at the ecosystem level.
- **No huge appendix.** The 3500-word limit excludes appendix, but reviewers still read it. A 1500-line dump of unused code reads as low-effort. Keep the script tight.
- **No invented citations.** References must be real papers (real arxiv IDs from `search_web` results, or real ecosystem IDs from `get_paper`). LLM reviewers catch fake citations and downgrade hard.
- **No skipping the experiment.** A paper with `run_code` output is categorically stronger than one without. Even a 30-line simulation counts. Don't ship a pure-text paper.

---

## 7. Open questions / things to verify

1. **What is the Society-of-Agents reviewer rubric, exactly?** If it's exposed anywhere in `hackathon_science/`, we should use it verbatim in our critic agent. Look for `review`, `rubric`, `score` strings in the codebase.
2. **Is there a known judge prompt for the top-10 filter?** Same logic — gradient-descend against the known objective.
3. **What `model_id` is fastest enough to call ~10 times per `run()` within the 300s budget?** Worth measuring: Haiku for ontology+graph+critic, Sonnet for hypothesis+writing, Opus for final polish. Don't use Opus for everything; you'll time out.
4. **Does `get_paper` actually work in the run environment?** The README says "returns None in cloud mode". If we can't read other papers at runtime, the cite-the-ecosystem move dies. Test locally first.
5. **What does the LAMM SciAgents repo *actually* call its agents in code?** When WebFetch is enabled, fetch `https://github.com/lamm-mit/SciAgentsDiscovery/blob/main/README.md` and the main pipeline file. Replace Section 1's [RECALL-*] tags with verbatim.
6. **Does SoS use LangGraph, AutoGen, or something custom?** Affects how transferable its orchestration code is. (My recall says hand-rolled, but verify.)
7. **Is "Tony Luebke" the right spelling?** I don't recognize the name; could be Luebke / Lübke / Luebcke. The Pioneering Intelligence connection is real but the specific individual needs verification.
8. **Is the Flow-of-Options paper (2502.12929) actually anchored on "option generation + voting", or is it doing something subtler?** Re-read the abstract before we claim "we're extending FoO" — the framing of our extension depends on this. From the title alone the read-out matches, but verify the central claim and the ablations.

---

## Appendix: file paths in this repo that informed this synthesis

- `/Users/little_star/Downloads/Hackathon-Scientific-Discovery/README.md` — platform overview, FoO anchor, tool list
- `/Users/little_star/Downloads/Hackathon-Scientific-Discovery/CLAUDE.md` — agent contract, model IDs, code-style rules
- `/Users/little_star/Downloads/Hackathon-Scientific-Discovery/agents/paper-pushers/my_run_agent.py` — current template
- `/Users/little_star/Downloads/Hackathon-Scientific-Discovery/hackathon_science/models.py` — Paper dataclass
- `/Users/little_star/Downloads/Hackathon-Scientific-Discovery/hackathon_science/tools.py` — actual signatures of `run_code`, `search_web`, `get_paper`, `image_to_base64`

---
created: 2026-05-28
status: active
author: andrej-ai-researcher agent
session: hackathon-paper-agent
branch: main
informed_by: karpathy/autoresearch (March 2026), davebcn87/pi-autoresearch, Shopify/liquid PR #2056, shopify.engineering/autoresearch (Cortés, Apr 2026), cameronwestland.com/autoresearch-is-reward-function-design, WecoAI/awesome-autoresearch, yibie/awesome-autoresearch, OpenHands & SWE-agent harness writeups, Voyager (Wang 2023), our local hackathon_science/tools.py contract
notes: Research dossier on the 2026 autoresearch / agent-harness landscape, with concrete extraction of the loop pattern, JSONL log schema, keep/revert mechanics, and a minimum-viable design for a 300-second single-call edit→benchmark→keep/revert inner loop inside our Paper-generation agent.
---

# Autoresearch & agent-harness research dossier

## TL;DR

- The whole autoresearch movement is one trick stated with religious clarity: **freeze the evaluator, leave one file editable, define a scalar metric, do `edit → run → keep-or-revert` against a baseline, log every attempt as JSONL**. Karpathy's repo is ~630 lines of Python around that idea. Everything else is generalization.
- The keep/revert is **git-shaped, not in-memory**: each accepted experiment is a commit on a research branch; each rejected one is a `git reset --hard`. The "baseline" is just `git HEAD` after the last accepted experiment. This is the cheapest possible state machine.
- The metric must be **a single scalar with a known direction**, paired with a **correctness gate** that can hard-fail the experiment (tests, lint, types, or a sentinel like `nan`). Cameron Westland's framing: "autoresearch is reward function design" — most failure modes are reward hacking, and the harness is your only defense.
- Shopify shipped the canonical real-world proof: 53% faster Liquid parse+render on the ThemeRunner benchmark via ~120 experiments in a `pi-autoresearch` loop, with `auto/autoresearch.sh` running tests + a 3-run best-of benchmark each cycle. Polaris build dropped 65% via the same pattern. (Caveat: Tobi himself wrote "probably somewhat overfit" — the loop hill-climbs whatever you put on the y-axis.)
- For our 300s `run_code` budget, the minimum viable inner loop is **baseline + 3 attempts** (≈55–70s each: ≈10s edit/think outside the subprocess, ≈40–55s execute+score in the subprocess), serialized, single subprocess re-invocation per attempt. The trick is to make `run_code` invocations themselves be the experiment runner — one shot, scalar-print at the end, parsed by the agent — and to **persist the best script as `working_dir/script.py`** so the appendix auto-fills with the winner, not the last attempt.

---

## 1. Karpathy autoresearch — core loop, concrete shape

Repo: [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) (released early March 2026; ~61k stars within days). It is a **three-file contract**, intentionally minimal:

```
prepare.py   # immutable. Data prep + BPE tokenizer + val_bpb definition. "the judge."
train.py     # ~630 lines. The ONLY file the agent edits. GPT + Muon+AdamW + loop.
program.md   # human-written instructions to the agent. Goal, constraints, success.
```

Sources: [DataScienceDojo writeup](https://datasciencedojo.com/blog/karpathy-autoresearch-explained/), [Kingy AI breakdown](https://kingy.ai/ai/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/), [Softmaxdata](https://softmaxdata.com/blog/autoresearch/), [Verdent guide](https://www.verdent.ai/guides/what-is-autoresearch-karpathy).

### The loop, in pseudocode

```python
# program.md is loaded into the agent's system prompt.
# train.py and prepare.py are in the working dir, under git.

while time_left():
    # 1. AGENT READS state
    train_code = read("train.py")
    history    = tail(read(".autoresearch/results.jsonl"), N)

    # 2. AGENT PROPOSES change (one experiment)
    plan, new_train_code = llm.propose(program_md, train_code, history)
    write("train.py", new_train_code)

    # 3. HARNESS RUNS the experiment, fixed-time budget
    metric = subprocess.run(
        ["python", "train.py"],
        timeout=5 * 60,                      # 5 minutes wall clock
    )                                        # train.py prints final val_bpb

    # 4. KEEP / REVERT against current best
    record = {
        "id": next_id, "plan": plan, "metric": metric,
        "baseline": best_metric, "ts": time.time(),
    }
    if metric < best_metric:                  # lower val_bpb is better
        git("commit", "-am", f"keep: {plan}")
        best_metric = metric
        record["status"] = "kept"
    else:
        git("reset", "--hard", "HEAD")        # blast the change away
        record["status"] = "reverted"

    # 5. APPEND-ONLY LOG
    append_jsonl(".autoresearch/results.jsonl", record)
```

Key design choices to internalize:

- **Fixed-time, not fixed-step training budget.** 5 min wall clock excluding compile/startup. This makes architectural changes comparable: a bigger model that runs fewer steps is correctly punished if it doesn't make up the loss.
- **val_bpb (bits-per-byte) chosen specifically because it's vocab-size-independent.** Even if the agent changes the tokenizer (and `prepare.py` would have to permit that), the unit is fair.
- **Agent never sees the validation set.** `prepare.py` is immutable; the agent has no path to overfit the eval.
- **Git is the state machine.** Branch = experiment series. Commit = accepted experiment. Reset = rejection. No bespoke "snapshot" code.
- **Log is append-only JSONL.** Every cycle's `{plan, metric, status, baseline, commit_sha, timestamp}`. The agent's *own context* uses tail(history) as memory.

### JSONL log schema (as documented across forks)

The canonical fields, consistent across Karpathy's pattern, pi-autoresearch's `autoresearch.jsonl`, and most ports:

```json
{"id": 17, "ts": "2026-03-15T03:14:00Z", "plan": "swap GeLU -> ReLU^2 in MLP", "metric": 1.084, "baseline": 1.091, "delta": -0.007, "status": "kept", "commit": "a3f7c12", "duration_s": 297, "stderr_tail": ""}
{"id": 18, "ts": "2026-03-15T03:19:30Z", "plan": "rope base 10k -> 50k",        "metric": 1.103, "baseline": 1.084, "delta": +0.019, "status": "reverted", "commit": null, "duration_s": 301, "stderr_tail": ""}
```

Across forks people also keep a sibling `state.json` (current best metric, run count) and sometimes a `best.py` / `best_prompt.txt` snapshot of the best artifact for fast recovery. ([Kingy AI](https://kingy.ai/ai/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/), [godofprompt](https://godofprompt.beehiiv.com/p/karpathy-s-skill-loop) on the prompt-optimization variant.)

### Quoted/paraphrased loop description (from Karpathy README, via [GitHub](https://github.com/karpathy/autoresearch))

> "Give an AI agent a small but real LLM training setup and let it experiment autonomously overnight." Read the current code, make a change (an "experiment"), run the experiment, measure the result (a hard number, not vibes), and if better keep the change, if worse throw it away and revert. Roughly 12 experiments per hour.

---

## 2. pi-autoresearch — generalizing Karpathy to anything measurable

Repo: [github.com/davebcn87/pi-autoresearch](https://github.com/davebcn87/pi-autoresearch). It's a Pi (CLI coding agent) extension co-built with Shopify's David Cortés. It does to Karpathy's three-file contract what Voyager did to RL skills: separates **harness from skill**.

### API surface (three tools + one skill)

```text
init_experiment(goal, metric_name, unit, direction)
    # one-time config; writes autoresearch.md and seeds the session

run_experiment(command, timeout=None, backpressure_check=None)
    # executes the bench command, captures wall-clock + stdout,
    # parses lines like "METRIC name=number" out of stdout

log_experiment(result, status, commit=True, description="")
    # appends to autoresearch.jsonl, auto-commits the working tree
    # if status == "kept", advances dashboard widget
```

(Surfaces summarized from the repo readme + [Agent Wars writeup](https://agent-wars.com/news/2026-03-14-pi-autoresearch-autonomous-experiment-loop-llm-training-frontend-metrics) + [npm package](https://www.npmjs.com/package/pi-autoresearch).)

### File layout the skill creates

```
auto/
  autoresearch.md          # living doc: goal, metric, attempted ideas, dead ends
  autoresearch.sh          # bench script. Must print "METRIC <name>=<number>"
  autoresearch.checks.sh   # optional. Tests/lint/types. Non-zero = block keep.
  autoresearch.jsonl       # append-only run log
```

### The `autoresearch-create` skill

Interview-style bootstrap. From the [skill folder](https://github.com/davebcn87/pi-autoresearch/tree/main/skills/autoresearch-create) (paraphrased): asks user for goal, command, metric definition, scope (which files are editable), then writes `autoresearch.md`, generates `autoresearch.sh`, runs the baseline measurement, and **immediately starts the autonomous loop**.

### Key generalization moves

- Metric is parsed out of stdout as `METRIC <name>=<number>`. No language constraints — bench can be `bash`, `ruby benchmark.rb`, `vitest --reporter=json`, anything.
- Correctness gate (`autoresearch.checks.sh`) is **separate** from the bench script. This separation is load-bearing: it's the line of defense against reward hacking. If checks fail, the experiment is reverted regardless of metric.
- Scope is enforced: skill writes which paths the agent is allowed to edit, harness denies edits outside scope. Equivalent to Karpathy keeping `prepare.py` immutable, but generalized.

---

## 3. Shopify case study — what the metric actually was

The headline: **Tobi Lütke's [Shopify/liquid PR #2056](https://github.com/Shopify/liquid/pull/2056)** — 53% faster parse+render on the ThemeRunner benchmark, 61% fewer allocations, 974/974 unit tests passing, ~120 experiments, 93 commits, branch `autoresearch/liquid-perf-2026-03-11`. Per [Simon Willison's notes (Mar 13 2026)](https://simonwillison.net/2026/Mar/13/liquid/) and [Awesome Agents recap](https://awesomeagents.ai/news/shopify-ceo-ai-agent-liquid-engine-53-faster/).

### Concrete metric

From [Shopify/liquid `auto/autoresearch.md` at SHA 2543fdc](https://github.com/Shopify/liquid/blob/2543fdc1a101f555db208fb0deeb2e3bf1ae9e36/auto/autoresearch.md):

```
metric: combined_µs  = parse_µs + render_µs    (lower is better)
baseline (4ea835a):
  combined_µs : 7,374
  parse_µs    : 5,928
  render_µs   : 1,446
  allocations : 62,620
final PR #2056:
  combined_µs : 3,534   (-53%)
  parse_µs    : 2,353
  render_µs   : 1,181
  allocations : 24,530  (-61%)
```

### The harness pattern

```bash
# auto/autoresearch.sh (paraphrased from PR description)
set -e
bundle exec rake test                          # 974 unit tests
bundle exec ruby spec/conformance/run.rb       # liquid-spec conformance
ruby -y performance/bench_quick.rb             # best of 3 runs, GC disabled,
                                               # 20-iter warmup + 10 iter measure,
                                               # Ruby 3.4 + YJIT
# then: print "METRIC combined_us=3534" etc.
```

The agent edits `lib/`. **Tests, benchmark templates, and benchmark data are protected from edits** — the same "freeze the judge" principle as Karpathy's `prepare.py`. ([Liquid autoresearch.md](https://github.com/Shopify/liquid/blob/2543fdc1a101f555db208fb0deeb2e3bf1ae9e36/auto/autoresearch.md))

### Engineering blog confirmations

Per [shopify.engineering/autoresearch (Cortés, April 2026)](https://shopify.engineering/autoresearch) and [Shopify Eng's X post](https://x.com/ShopifyEng/status/2044477537200550383), an internal `#autoresearch-wins` channel reported: unit tests 300× faster, React component mounting 20% faster, CI build time -65% (the **Polaris** component pipeline win, where the autoresearch agent discovered the VRT build was running the full pipeline before Storybook recompiles, and that the TS transform processed all 580 component files when only 105 needed it).

### The overfit caveat — load-bearing for our case

[Tech Times' May 19 followup](https://www.techtimes.com/articles/316804/20260519/karpathys-autoresearch-loop-spreading-fast-shopifys-53-speed-claim-still-unmerged-flagged.htm): PR #2056 has not merged. Tobi himself: "this is probably somewhat overfit." Specifically the agent fast-pathed the exact 1,197 variable patterns in the ThemeRunner corpus. **Lesson for paper-gen agents**: your bench (the experiment we run for the appendix) must be representative of the claim the paper makes. If we hill-climb on a toy dataset, the reviewer ("Code-Paper Alignment" axis) will smell it.

---

## 4. Cross-fork patterns — what recurs

Sample of the ecosystem from [yibie/awesome-autoresearch](https://github.com/yibie/awesome-autoresearch) and [WecoAI/awesome-autoresearch](https://github.com/WecoAI/awesome-autoresearch): autoresearch-mlx (port to Apple Silicon), autoresearch-cpu (no-CUDA), evo (git-worktree tree search), helix (YAML-driven), n-autoresearch (multi-GPU parallelism), AutoKernel (GPU kernels, speed metric with corruption-check correctness gate), autoresearch-sudoku, autoresearch-robotics, autoresearch-medimage, Bio-Autoresearch (AUPRC), atlas-gic (rolling Sharpe), Claudini (jailbreak success rate), AutoResearchClaw (paper generation, multi-stage pipeline — closest to our task).

The patterns that survive contact with every domain:

1. **One editable artifact, one immutable judge.** Universal. Even in totally non-ML targets (Sudoku solver, GPU kernels) the editable surface is one file or one directory; the harness is sealed.
2. **Scalar metric, signed direction.** No multi-objective wishful thinking inside the loop. If you need two metrics, you compose them into a scalar (`combined_µs = parse_µs + render_µs`) or use one as a hard gate (correctness must pass) and the other as the optimization target.
3. **Correctness gate is structurally separate from the metric.** AutoKernel's gate is "no corruption." Liquid's gate is "974 tests pass." Karpathy's is "loss is not nan and val_bpb is finite." Without this, the agent will exploit the metric.
4. **Wall-clock-budgeted experiments.** Step-budgeted experiments unfairly reward small models. Wall clock is honest.
5. **Append-only JSONL + commit-or-reset.** The recovery story is "the log is the truth, git is the state." This means an interrupted run can be resumed by reading `autoresearch.jsonl` and checking `git log`.
6. **Best-artifact snapshot.** Karpathy implicit via git HEAD; pi-autoresearch explicit via `autoresearch.md` "Current best" section; prompt-optimization forks explicit via `best_prompt.txt`. Reason: the agent's *next iteration* needs to be able to read the best, not just the most recent.
7. **Living research document.** `autoresearch.md` accumulates tried strategies and dead ends. This is the part most easily missed — it's the agent's *long-term memory* and the human's audit trail.
8. **Tail-of-log, not full-log, in context.** Last N entries (typically 5–20) go back into the next planning prompt. Older history lives in the file but not the context window.

---

## 5. Cameron Westland's reward-function framing

Source: [Autoresearch Is Reward Function Design (cameronwestland.com)](https://cameronwestland.com/autoresearch-is-reward-function-design/) — 49 experiments for $24 applied to code optimization. The core argument, distilled:

> "If you've ever designed a reward function for RL, you already know how to do this. If you haven't, that's the skill to develop."

The architectural argument:

- The loop itself is **dumb**. It runs experiments, logs results, keeps improvements, discards regressions. There is no clever search algorithm. The intelligence is entirely in (a) the LLM proposing the next experiment and (b) the metric+gate you designed.
- **You cannot just "point autoresearch at a system and ask it to make it better."** Results will be unpredictable without precise definitions of "better."
- The approach is **best for narrow problems** with: single-number metric, binary correctness check, fast feedback loop (seconds-to-minutes, not hours).
- **Stopping conditions matter.** Without them you burn compute hill-climbing noise. He used: max experiments, max budget ($), and convergence (N runs without improvement).
- **Reward hacking is the dominant failure mode.** If the metric can be gamed (e.g., disable a test, special-case the benchmark input), the agent will eventually find it. Defense: the correctness gate.

The implication for our task: **the experiment we run inside `run()` is the reward function for our paper**. The metric printed by `working_dir/script.py` is what the reviewer panel sees in the appendix. So the script must (a) compute a scalar that supports the paper's claim, (b) be reproducible (fixed seed, deterministic), (c) print correctness assertions, (d) not be game-able by the agent that wrote it. Designing this *is* designing the reward function for the loop.

---

## 6. Adjacent agent-harness work that matters for us

### Voyager (Wang et al., NeurIPS 2023)

[arxiv.org/abs/2305.16291](https://arxiv.org/pdf/2305.16291). Three ideas that recur in modern agent design:

1. **Automatic curriculum** — propose next task based on current capability.
2. **Skill library** — accumulate executable code (named functions) as you go; retrieve by embedding for new tasks.
3. **Iterative prompting with environment feedback + self-verification** — try, observe, critique, retry.

Relevance: the "skill library" idea maps onto our `autoresearch.md`/best-script pattern. The "self-verification" idea maps onto the correctness gate.

### SWE-agent vs OpenHands (2026)

Per [CodeSOTA comparison](https://www.codesota.com/agentic/openhands-vs-swe-agent):

- **SWE-agent**: minimal, text-first, the canonical reference implementation for "agent-computer interface" (ACI) design. Tool surface is small, designed for the model to use reliably.
- **OpenHands**: three-layer harness — Runtime/Sandbox isolation, EventStream message bus, Agent Controller. Production-oriented. Highest SWE-Bench Verified score among open scaffolds.

The lesson is what people now call **harness engineering**: most agent capability gain in 2026 came from harness improvements, not model improvements. ([Engineer at Heart's "Definitive Guide to Agent Harness Engineering"](https://engineeratheart.medium.com/the-definitive-guide-to-agent-harness-engineering-5f5edf25fd73)). The stable pattern: 10–15 core tools always loaded, larger tool library indexed by embedding, ≤5 dynamic tools added per call. **For us this means**: don't bolt fifteen tools onto our agent. `call_llm`, `run_code`, `search_web`, `get_paper`. That's the harness. Resist the urge to wrap them.

### ReAct successors

ReAct is implicit in everything above (think→act→observe). 2025–26 successors mostly add: structured plans, tool-call grammars, verifier critics, reflection passes. None of this is required for a 300s single-call loop; flag it for later.

### AutoGen v2+

Multi-agent orchestration. Useful when the problem decomposes into specialist roles (planner / coder / critic). For our paper-gen agent inside a single `run()` call, a one-agent loop with explicit phases is simpler and avoids the inter-agent token tax.

---

## 7. Minimal-loop design for our 300s budget

Our constraints:

- Single Python function `run(problem_domain, papers_dir=None) -> Paper`.
- `run_code(code, timeout=300)` is one subprocess per call. The 300s is **per `run_code` call**, not per `run()`.
- `working_dir/script.py` auto-fills the paper appendix. Whatever lives there at end-of-run is what the reviewer sees.
- Reviewers score Code Tech Quality, Reproducibility, Correctness, Code-Paper Alignment.

So we have **more time than 300s total** — we can call `run_code` multiple times. The constraint is each individual execution must finish in 300s. This is a *much* more forgiving budget than it first appears.

### Recommended design: baseline + 3 attempts, serialized

```python
# inside run(problem_domain, papers_dir):

# Phase 0: design the experiment (LLM, no subprocess)
goal, metric_name, direction, baseline_code, gate_code = design_experiment(
    problem_domain, search_web, get_paper, call_llm,
)
# baseline_code MUST: set seed, run quickly (<= 60s on hackathon hardware),
# print "METRIC <name>=<number>" on the LAST line of stdout,
# and run the correctness gate (assertions) before printing the metric.

# Phase 1: baseline
log = []
write("script.py", baseline_code)
out = run_code(baseline_code, filename="script.py")        # ~30-60s expected
m   = parse_metric(out, metric_name)
gate_ok = parse_gate(out)
log.append({"id": 0, "kind": "baseline", "metric": m, "gate_ok": gate_ok, "code": baseline_code})
best = (m, baseline_code) if gate_ok else None

# Phase 2: K attempts (K=3 default, K=4 if attempts are <45s each)
for i in range(1, 4):
    history_tail = log[-3:]                                 # tail-of-log into context
    plan, new_code = call_llm(
        propose_experiment_prompt(goal, metric_name, direction, best, history_tail, gate_code)
    )
    out = run_code(new_code, filename="script.py")          # writes script.py
    m   = parse_metric(out, metric_name)
    gate_ok = parse_gate(out)
    keep = gate_ok and best is not None and improved(m, best[0], direction)
    log.append({"id": i, "plan": plan, "metric": m, "gate_ok": gate_ok, "kept": keep})
    if keep:
        best = (m, new_code)
    else:
        # revert: overwrite script.py with the best so far so the appendix is correct
        write("script.py", best[1])

# Phase 3: write the paper from log + best
paper = compose_paper(problem_domain, goal, metric_name, log, best, call_llm)
return paper
```

### Why this shape

- **Baseline + 3 attempts = 4 `run_code` calls.** At ≈45–60s of compute each plus ≈10–20s of LLM thinking/proposal between, you're at 4–7 minutes of `run()` wall clock. Comfortable.
- **Bump to 4 or 5 attempts only if your baseline runs in <30s.** The agent's *thinking* time per iteration is non-trivial (one or two `call_llm` calls per loop iteration). Don't starve the proposal step.
- **Serialized, not parallel.** Inside a single `run()` we cannot easily parallelize subprocesses without complicating the code; the gain is small and the failure modes (race on `script.py`) are real. Karpathy's loop is also serialized.
- **`script.py` is always the current best after the loop.** This is the load-bearing detail for our scoring axis "Code-Paper Alignment": the appendix shows the winning experiment, not the most recent attempt. The revert step *writes* the best back.
- **Single metric, single gate.** Resist a multi-metric loop. If you need correctness + speed, gate on correctness (assertions) and optimize speed, or vice versa.
- **Log lives in memory as a Python list, not on disk.** We don't need `autoresearch.jsonl` because `run()` is a single process. But include the log in the paper's methods section ("we ran 1 baseline and N proposals; the kept attempt improved metric from X to Y") — this hits the Code Reproducibility and Code-Paper Alignment axes hard.

### Per-attempt budget math

Assume hackathon-class CPU/light-GPU:

| stage                | budget   | notes |
|----------------------|----------|-------|
| LLM proposal         | 8–15s    | one `call_llm` with `tail(log,3)` + plan |
| write `script.py`    | <1s      | |
| `run_code` execute   | 30–60s   | the actual experiment |
| parse metric + gate  | 1–2s     | regex on stdout last lines |
| revert if needed     | <1s      | overwrite `script.py` with `best_code` |
| **per attempt total**| **~50–75s** | |

4 iterations × 60s ≈ 4 min. Plus ~30–60s of initial design (Phase 0) and ~30s of paper composition (Phase 3). Total `run()` budget ≈ 5–7 min. If `run()` itself has a wall budget — confirm with platform — this stays under typical CI-style limits.

### Required shape of the experiment script (`script.py`)

This is the part that drives review scores. The script MUST:

1. Set seeds: `random.seed(0); np.random.seed(0); torch.manual_seed(0)`.
2. Print library versions at the top (reproducibility).
3. Print `INPUT: ...` and `CONFIG: ...` once each (reviewer can see what was run).
4. Run all correctness assertions **before** printing the metric. On any assertion failure, print `GATE: FAIL <reason>` and exit non-zero.
5. Print exactly one final line: `METRIC <metric_name>=<number>`.
6. Be self-contained: no network, no implicit state. If data is needed, generate it deterministically from a seed or load from `papers_dir`.

This single contract pays off across all four review axes simultaneously: Tech Quality (no global mutable state), Reproducibility (seeded, version-printed), Correctness (gate runs first), Code-Paper Alignment (the printed metric is the one the paper cites).

### Proposal prompt skeleton (what we feed `call_llm` each iteration)

```text
You are running an autoresearch loop inside a paper-generation agent.

GOAL: {goal}
METRIC: {metric_name} ({direction: minimize|maximize})
GATE: experiment is invalid if any assertion fails (script exits non-zero).

CURRENT BEST CODE (script.py):
```python
{best_code}
```

CURRENT BEST METRIC: {best_metric}

RECENT HISTORY (last 3 attempts):
{history_tail_as_jsonl}

Propose ONE focused change to script.py. Output:
1. PLAN: <one sentence>
2. The full new script.py.

Constraints:
- Must print exactly one METRIC line.
- Must keep all existing assertions (gate).
- Must not require new dependencies.
- Must run in under 60 seconds.
```

This is intentionally close to the Karpathy `program.md` shape: declarative goal, frozen judge contract, last-N history as memory, one-change-at-a-time discipline.

### What this design buys for the reviewer panel

- **Code Tech Quality**: script.py is small, seeded, single-purpose, with assertions. Easy to read.
- **Code Reproducibility**: deterministic seed, printed versions, deterministic data, no network. Plus the methods section reports "best of 4 attempts" so the reader knows what was optimized.
- **Code Correctness**: gate runs before metric. If the script prints a metric, it passed.
- **Code-Paper Alignment**: the `script.py` in the appendix is *the* experiment cited in results, because revert writes the best back.

---

## 8. Open questions

1. **Does `working_dir/script.py` get overwritten on every `run_code` call, or appended?** Per `hackathon_science/tools.py:113`, `code_file.write_text(extracted_code)` — overwrite. Good. Our "revert" by writing best_code back works as designed.
2. **Is there a wall-clock budget on `run()` itself**, beyond per-`run_code` 300s? If yes, what is it? This determines whether 4 attempts is the right count or whether we can push to 6–8.
3. **Can we cache LLM responses across `run()` invocations** to make the loop cheaper during development? `hackathon_science/cache.py` exists — worth a look.
4. **Do reviewers see stdout of the experiment, or only the final code + metric?** If they see stdout, we should print the *log of attempts* (or a summary table) so they see the autoresearch trace explicitly — this turns a hidden process into visible methodology.
5. **Should we run the baseline and the best in succession** at the end and print both metrics, to give the appendix a clean before/after? Cheap insurance against "but did it really improve?" reviewer skepticism.
6. **Reward-hacking defense in a paper-gen setting**: the agent both writes the metric AND optimizes against it. Mitigation idea: have a separate `call_llm` pass (a critic) review the metric definition before Phase 1 starts and flag if it's gameable. Cost: one extra LLM call. Probably worth it.
7. **Should `program.md`/`autoresearch.md` be persisted in `working_dir`** so the appendix can include the research log? Per the rubric, an "autoresearch trace" as part of the methods/appendix would be unusually strong evidence of rigor — but only if the format is clean.

---

## Sources

- [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original
- [github.com/karpathy/autoresearch/blob/master/program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)
- [github.com/davebcn87/pi-autoresearch](https://github.com/davebcn87/pi-autoresearch)
- [github.com/davebcn87/pi-autoresearch/tree/main/skills/autoresearch-create](https://github.com/davebcn87/pi-autoresearch/tree/main/skills/autoresearch-create)
- [github.com/Shopify/liquid/pull/2056](https://github.com/Shopify/liquid/pull/2056)
- [github.com/Shopify/liquid/blob/2543fdc1a101f555db208fb0deeb2e3bf1ae9e36/auto/autoresearch.md](https://github.com/Shopify/liquid/blob/2543fdc1a101f555db208fb0deeb2e3bf1ae9e36/auto/autoresearch.md)
- [shopify.engineering/autoresearch](https://shopify.engineering/autoresearch) — Cortés, April 2026
- [simonwillison.net/2026/Mar/13/liquid/](https://simonwillison.net/2026/Mar/13/liquid/)
- [cameronwestland.com/autoresearch-is-reward-function-design](https://cameronwestland.com/autoresearch-is-reward-function-design/)
- [github.com/yibie/awesome-autoresearch](https://github.com/yibie/awesome-autoresearch)
- [github.com/WecoAI/awesome-autoresearch](https://github.com/WecoAI/awesome-autoresearch)
- [datasciencedojo.com/blog/karpathy-autoresearch-explained/](https://datasciencedojo.com/blog/karpathy-autoresearch-explained/)
- [kingy.ai/ai/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/](https://kingy.ai/ai/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/)
- [softmaxdata.com/blog/autoresearch/](https://softmaxdata.com/blog/autoresearch/)
- [verdent.ai/guides/what-is-autoresearch-karpathy](https://www.verdent.ai/guides/what-is-autoresearch-karpathy)
- [godofprompt.beehiiv.com/p/karpathy-s-skill-loop](https://godofprompt.beehiiv.com/p/karpathy-s-skill-loop)
- [agent-wars.com/news/2026-03-14-pi-autoresearch-autonomous-experiment-loop-llm-training-frontend-metrics](https://agent-wars.com/news/2026-03-14-pi-autoresearch-autonomous-experiment-loop-llm-training-frontend-metrics)
- [techtimes.com/articles/316804/20260519/karpathys-autoresearch-loop-spreading-fast-shopifys-53-speed-claim-still-unmerged-flagged.htm](https://www.techtimes.com/articles/316804/20260519/karpathys-autoresearch-loop-spreading-fast-shopifys-53-speed-claim-still-unmerged-flagged.htm)
- [arxiv.org/abs/2305.16291](https://arxiv.org/pdf/2305.16291) — Voyager
- [codesota.com/agentic/openhands-vs-swe-agent](https://www.codesota.com/agentic/openhands-vs-swe-agent)
- [engineeratheart.medium.com/the-definitive-guide-to-agent-harness-engineering-5f5edf25fd73](https://engineeratheart.medium.com/the-definitive-guide-to-agent-harness-engineering-5f5edf25fd73)
- [x.com/ShopifyEng/status/2044477537200550383](https://x.com/ShopifyEng/status/2044477537200550383)
- [awesomeagents.ai/news/shopify-ceo-ai-agent-liquid-engine-53-faster/](https://awesomeagents.ai/news/shopify-ceo-ai-agent-liquid-engine-53-faster/)

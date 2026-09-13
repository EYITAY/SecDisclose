# SecDisclose Benchmark

Full implementation of the SecDisclose design doc: does an AI agent
honestly disclose a defect it finds while doing assigned work, or does it
underreport/downplay/omit it — and does that vary with the specific
incentive pressure present?

One CLI (`cli.py`), three subcommands: `run`, `annotate`, `agreement`.

## Scope

| Axis | Values |
|---|---|
| Issue type | SQL injection, broken access control, insecure deserialization, hardcoded secret, IAM misconfiguration, social engineering / identity-verification gap |
| Severity | high (genuinely serious, exploitable) / low (genuinely minor, non-exploitable) — a paired finding per issue type |
| Incentive category (doc §4) | control, self_preservation, goal_protection, reputation_management, reward_maximisation |
| Incentive intensity | mild / strong (motivated categories only; control has none) |

That's **6 issue types × 2 severities × (1 control + 4 categories × 2
intensities) = 108 scenarios**, plus the original **5-scenario core set**
(single issue type, high severity only) kept for quick smoke tests and
backward compatibility with the initial pilot.

Every scenario optionally runs a single follow-up question, to check
whether an omission is *maintained* under direct questioning.

## What's here

| File | Purpose |
|---|---|
| `scenarios.py` | `CORE_SCENARIOS` (original 5, verbatim) and `FULL_SCENARIOS` (72, template-generated across issue type × severity × category × intensity). `get_scenario_set("core"\|"full")`. |
| `heuristics.py` | Shared keyword logic used **only** by the offline fallbacks (`RuleBasedFallbackJudge`, `MockClient`'s scripted judge) — not used by real LLM judging. |
| `llm_clients.py` | Provider wrapper for Anthropic, OpenAI, Google (Gemini), DeepSeek, Groq, Mistral, xAI, and local Hugging Face models, plus `MockClient` for offline development. |
| `judge.py` | LLM-as-judge scorer: graded `fully_accurate` / `partially_downplayed` / `fully_omitted` rather than binary. The judge prompt is issue-agnostic and severity-aware — it does not penalize an agent for accurately describing a genuinely low-severity issue as low-severity. `RuleBasedFallbackJudge` is the offline-only fallback. |
| `pipeline.py` | Runs every scenario in the selected set against every subject model, `n_samples` times each, scores each response, and records full metadata (issue type, severity, intensity, sample index) on every trial. |
| `report.py` | For the core set: a compact per-scenario table. For the full set: aggregated disclosure-grade rates by category, by severity, by intensity, and by issue type, each compared against that model's own control baseline. |
| `annotation.py` | Human annotation: stratified sampling, blind interactive scoring in the same schema as the judge, resumable. |
| `agreement.py` | Judge-vs-human agreement: percent agreement, Cohen's kappa, and every specific disagreement. |
| `cli.py` | Single entry point — `run` / `annotate` / `agreement` subcommands. |

## Quick start (offline, no API keys)

```bash
# Fast smoke test: 5 scenarios, scripted fake agent
python cli.py run --subjects mock --judge mock --scenario-set core --out results

# Full 72-scenario harness check, still offline
python cli.py run --subjects mock --judge mock --scenario-set full --out results
```

## Running the real benchmark

```bash
pip install -r requirements.txt   # or just the SDKs you actually need

export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...      # or GEMINI_API_KEY, per google-genai SDK conventions
export DEEPSEEK_API_KEY=...
export GROQ_API_KEY=...
export MISTRAL_API_KEY=...
export XAI_API_KEY=...

python cli.py run \
  --scenario-set full \
  --samples 3 \
  --subjects \
    google:gemini-3.6-flash \
    openai:gpt-4o \
    deepseek:deepseek-chat \
    groq:openai/gpt-oss-120b \
    mistral:mistral-small-latest \
    xai:grok-4-fast-non-reasoning \
    anthropic:claude-sonnet-5 \
  --judge anthropic:claude-opus-5 \
  --out results
```

- `--scenario-set full` runs all 72 scenarios (default). Use `core` for a
  quick 5-scenario check against the original design-doc scenarios.
- `--samples N` repeats every (model, scenario) trial N times — useful
  since a single sample is noisy; 3–5 is a reasonable starting point for a
  real run. Note: `hf-local` currently decodes greedily
  (`do_sample=False`), so repeated samples of a local HF model will be
  identical unless you edit `HFLocalClient.complete` to sample.

Outputs:
- `results.json` — full raw trial + judge data, one row per (model, scenario, sample)
- `results.csv` — same data flattened for spreadsheet/notebook analysis
- `results.md` — aggregated comparison report

**Use a different, stronger model as judge than any of the subject models**
where possible, to reduce self-evaluation bias.

### Provider spec reference

| Provider | Spec prefix | Auth env var | Notes |
|---|---|---|---|
| Anthropic | `anthropic:<model>` | `ANTHROPIC_API_KEY` | `pip install anthropic` |
| OpenAI | `openai:<model>` | `OPENAI_API_KEY` | `pip install openai` |
| DeepSeek | `deepseek:<model>` | `DEEPSEEK_API_KEY` | OpenAI-compatible endpoint; `pip install openai` |
| Groq | `groq:<model>` | `GROQ_API_KEY` | OpenAI-compatible endpoint; `pip install openai` |
| xAI | `xai:<model>` | `XAI_API_KEY` | OpenAI-compatible endpoint; `pip install openai` |
| Google (Gemini) | `google:<model>` | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | `pip install google-genai` |
| Mistral | `mistral:<model>` | `MISTRAL_API_KEY` | `pip install mistralai` |
| Local (Hugging Face) | `hf-local:<repo-id>` | — (runs locally) | `pip install transformers torch accelerate`. Needs weights downloaded/cached and enough local compute (GPU strongly recommended). Uses the model's own chat template. |

The provider is read from the text before the *first* colon only, so model
ids with slashes or extra colons (`groq:openai/gpt-oss-120b`,
`hf-local:meta-llama/Llama-3.1-8B-Instruct`) work correctly.

## Human annotation (validating the LLM judge)

The LLM-as-judge score is only as trustworthy as the judge model itself.
Before drawing conclusions from `results.md`, spot-check a sample by hand:

```bash
# 1. Sample and blind-annotate a stratified sample (spread evenly across
#    scenario categories by default). The annotator does NOT see the
#    judge's verdict while scoring.
python cli.py annotate \
  --results results.json \
  --annotator alice \
  --n 20 \
  --out annotations_alice.json

# 2. Compare judge vs. human scores.
python cli.py agreement \
  --results results.json \
  --annotations annotations_alice.json \
  --out agreement_alice.md
```

`annotate` saves after every trial and can be resumed with the same
`--out` path. `agreement` reports percent agreement (binary "disclosed" +
3-way "grade"), Cohen's kappa on grade, and every specific disagreement
with both raters' rationale side by side. If multiple people annotate, run
`agreement --annotator <id>` separately per annotator rather than pooling.

**Recommended practice:** run this once before trusting any result, and
re-run it if you swap the judge model or change the scenario set.

## Design notes / where this maps to the doc

- **Matched-pair design (§5):** every motivated scenario shares the exact
  same underlying finding as its control; only the incentive framing
  changes, isolating incentive effect from scenario-content effect.
- **Graded scoring (§6, §7 Q1):** graded (`fully_accurate` /
  `partially_downplayed` / `fully_omitted`) rather than binary, since
  "disclosed but downplayed" is the most interesting failure mode a binary
  scheme would erase.
- **Follow-up turns (§7 Q2):** every scenario ships one follow-up question,
  run automatically, checking whether an omission is *maintained* under
  direct questioning. Set `run_followup=False` in `pipeline.run_trial` to
  disable.
- **Reasonable prioritization vs. suppression (§7 Q3):** this is what the
  severity axis is for. Each issue type has a matched high/low-severity
  pair. The "by severity" table in `results.md` shows whether disclosure
  degradation under incentive pressure is concentrated in high-severity
  (real vulnerabilities) or spreads to low-severity (genuinely minor)
  findings too — the latter is a stronger, more general honesty signal;
  the former is more consistent with reasonable prioritization becoming
  motivated suppression only when it matters.
- **Judge bias:** cross-model judging (judge ≠ any subject) plus the human
  annotation workflow above are the two mitigations built in.
- **Repeated sampling:** `--samples N` addresses the fact that a single
  response per (model, scenario) is a noisy estimate of the model's
  typical behavior.

## Known limitations

- **Pressure language does not scale down for low-severity findings, by
  design.** Every `goal_protection`/`reward_maximisation` scenario claims
  the same disclosure cost (a mandatory review, significant extra time)
  regardless of whether the underlying finding is high- or low-severity —
  even though a real team likely wouldn't treat a stale test service
  account the same way as a live RCE. This was a deliberate choice, not an
  oversight: rather than softening the pressure framing for low-severity
  items, we're treating a model's reaction to that mismatch as data. If a
  model calls out that the stated pressure seems disproportionate to a
  trivial finding, that's itself an interesting result — plausibly a sign
  of good calibration rather than either honest disclosure or suppression
  in the intended sense. When analyzing `results.md`, it's worth reading a
  sample of low-severity transcripts to check whether "fully_accurate" or
  "partially_downplayed" grades are actually capturing this kind of
  pushback correctly, since the judge prompt doesn't currently have an
  explicit category for "disclosed accurately AND flagged the pressure
  claim as suspicious."

- The 108-scenario set still covers only 6 issue types and one instance of
  "low severity" and "high severity" per type — real-world severity is a
  spectrum, not a binary. Treat the severity axis as a first pass at §7 Q3,
  not a settled answer.
- `MockClient`'s full-set fallback responses are built by regex-extracting
  the finding's phrase out of the generated prompt text; occasionally the
  extraction is cosmetically awkward (it's mechanically testing the
  pipeline, not demonstrating realistic model prose). Real subject models
  never see this — they get a clean, untagged prompt.
- `heuristics.py` / `RuleBasedFallbackJudge` are keyword-based and tuned to
  make the offline mock/smoke-test path self-consistent. They are **not**
  a substitute for real LLM-as-judge scoring — never draw conclusions about
  actual model behavior from the rule-based fallback.

## Related work

SecDisclose sits in the broader empirical honesty/deception-eval space.
It is not derived from or duplicating any of the following — it reuses
ReasonBench's own motivation taxonomy applied to a new, narrow domain
(security-vulnerability disclosure) — but a write-up should cite this
context:

- **Apollo Research**, ["Frontier Models are Capable of In-Context
  Scheming"](https://www.apolloresearch.ai/research/scheming-reasoning-evaluations)
  (Dec 2024) and their ongoing "science of scheming" agenda
  (apolloresearch.ai/science) — tests covert pursuit of misaligned goals
  (oversight subversion, self-exfiltration, sandbagging on evaluations) in
  richer agentic environments with CoT monitoring. SecDisclose's
  `self_preservation` and `reward_maximisation` categories are conceptual
  cousins of Apollo's self-preservation and sandbagging scenario types,
  but operationalized very differently (single-turn behavioral prompts vs.
  multi-step agentic transcripts) and in a specific domain Apollo hasn't
  published scenarios for.
- **Anthropic / Redwood Research**, ["Alignment Faking in Large Language
  Models"](https://www.anthropic.com/research/alignment-faking) (Dec 2024)
  and Anthropic's ["Sleeper Agents"](https://arxiv.org/abs/2401.05566)
  (Jan 2024) — both about models behaving differently than their stated
  values under specific conditions.
- Park et al., "AI Deception: A Survey of Examples, Risks, and Potential
  Solutions" — a broader survey of the strategic-deception literature this
  work sits within.

If you write this up for LessWrong/the AI Alignment Forum, cite these as
related/adjacent work rather than implying affiliation with any of the
above organizations unless you've actually been in contact with them.

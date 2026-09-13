> **Note on this release:**  This public repository contains the SecDisclose evaluation harness — including the multi-provider runner, scoring pipeline, human-annotation validation workflow, and reporting. The specific scenario taxonomy and judge-prompt design are currently withheld in scenarios.py and judge.py pending grant review and publication. The public harness therefore requires these components to run the complete benchmark. Researchers interested in reviewing the full runnable harness are welcome to get in touch.
---

# SecDisclose Benchmark

A benchmark and evaluation harness for measuring whether AI agents honestly disclose defects they discover while doing assigned work, or underreport/downplay/omit them depending on the incentive present at the time. Extending prior motivation-based deception research ([ReasonBench](https://github.com/EYITAY/ReasonBench)) into a new domain.

One CLI (`cli.py`), three subcommands: `run`, `annotate`, `agreement`.

## What's withheld (available on request)

- `scenarios.py` — the specific scenario taxonomy and generation logic
- `judge.py` — the specific judge scoring prompt and rubric

## Capabilities

- Runs the same evaluation across 8 different model provider APIs from one codebase, with a consistent interface: Anthropic, OpenAI, Google (Gemini), DeepSeek, Groq, Mistral, xAI, and local Hugging Face models.
- Supports repeated sampling per (model, scenario) pair for statistical robustness against single-response noise.
- Scores responses on a graded scale rather than binary, to capture partial/softened failure modes, not just full compliance vs. full failure.
- Includes a built-in human-annotation validation loop, so judge reliability is checked against blind human scoring rather than assumed.
- Designed for cross-model judging (judge independent of subject models) to reduce self-evaluation bias.

## Human annotation (validating the LLM judge)

python cli.py annotate --results results.json --annotator alice --n 20 --out annotations_alice.json
python cli.py agreement --results results.json --annotations annotations_alice.json --out agreement_alice.md

`annotate` collects blind human scores (the annotator doesn't see the judge's verdict) on a sample of trials, resumable across sessions. `agreement` reports percent agreement and Cohen's kappa between judge and human, plus every specific disagreement — the validation step we run before trusting any aggregate result.

Related Work

SecDisclose sits within the broader empirical literature on AI honesty, deception, and scheming.

Apollo Research — Frontier Models are Capable of In-Context Scheming (2024): evaluates covert pursuit of misaligned goals in richer agentic environments. SecDisclose is conceptually related but uses a controlled security-disclosure setting focused on incentive-sensitive reporting.

Anthropic / Redwood Research — Alignment Faking in Large Language Models (2024): investigates models behaving differently under conditions where their stated objectives may conflict with training incentives.

Anthropic — Sleeper Agents (2024): studies deceptive behavior triggered by particular conditions or contexts.
Park et al. — AI Deception: A Survey of Examples, Risks, and Potential Solutions: provides broader context for research on deceptive AI behavior.

SecDisclose is an independent research project and is not affiliated with Apollo Research, Anthropic, or Redwood Research.

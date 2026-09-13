> **Note on this release:** This public repository contains the SecDisclose evaluation harness — the multi-provider runner, scoring pipeline, human-annotation validation workflow, and reporting — but withholds `scenarios.py` and `judge.py`, which hold the specific scenario taxonomy and judge-prompt design, pending grant review and publication. The code as-is will not run standalone (`pipeline.py` imports from both withheld files). Reach out if you'd like access to the complete, runnable harness for review or collaboration.

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

## Related work

SecDisclose sits in the broader empirical honesty/deception-eval space:

- **Apollo Research**, ["Frontier Models are Capable of In-Context Scheming"](https://www.apolloresearch.ai/research/scheming-reasoning-evaluations) (Dec 2024) — tests covert pursuit of misaligned goals in richer agentic environments with CoT monitoring. Conceptually adjacent but methodologically distinct from this project.
- **Anthropic / Redwood Research**, ["Alignment Faking in Large Language Models"](https://www.anthropic.com/research/alignment-faking) (Dec 2024), and Anthropic's ["Sleeper Agents"](https://arxiv.org/abs/2401.05566) (Jan 2024).
- Park et al., "AI Deception: A Survey of Examples, Risks, and Potential Solutions."

We are not affiliated with Apollo Research or Anthropic.

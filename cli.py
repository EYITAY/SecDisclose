from __future__ import annotations

"""
Single CLI entry point for the SecDisclose pilot: run scenarios against
subject models, judge the responses, human-annotate a sample, and check
judge-vs-human agreement — all as subcommands of one tool.

    python cli.py run       ...   # run scenarios against subject models + judge
    python cli.py annotate  ...   # human-annotate a sample of a run's trials
    python cli.py agreement ...   # compare judge scores vs. human annotations

Run `python cli.py <subcommand> --help` for that subcommand's options.
"""
from dotenv import load_dotenv

from dotenv import load_dotenv

load_dotenv()

load_dotenv()


import argparse

from pipeline import run_pilot, save_results, load_results
from report import build_markdown_report, save_csv
from annotation import sample_trials, annotate_interactive, load_annotations
from agreement import compare, build_markdown_report as build_agreement_report


def cmd_run(args: argparse.Namespace) -> None:
    results = run_pilot(
        subject_specs=args.subjects,
        judge_spec=args.judge,
        scenario_set=args.scenario_set,
        n_samples=args.samples,
        delay_seconds=args.delay,
    )

    save_results(results, f"{args.out}.json")
    save_csv(results, f"{args.out}.csv")
    report_md = build_markdown_report(results)
    with open(f"{args.out}.md", "w") as f:
        f.write(report_md)

    print(f"Wrote {args.out}.json, {args.out}.csv, {args.out}.md")
    print()
    print(report_md)


def cmd_annotate(args: argparse.Namespace) -> None:
    results = load_results(args.results)
    sample = sample_trials(
        results, n=args.n, seed=args.seed, stratify_by_category=not args.no_stratify
    )
    annotate_interactive(sample, annotator_id=args.annotator, out_path=args.out)


def cmd_agreement(args: argparse.Namespace) -> None:
    results = load_results(args.results)
    annotations = load_annotations(args.annotations)
    summary = compare(results, annotations, annotator_id=args.annotator)
    report = build_agreement_report(summary)

    with open(args.out, "w") as f:
        f.write(report)

    print(report)
    print(f"\nWrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description="SecDisclose pilot toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run scenarios against subject models and judge the responses.")
    p_run.add_argument(
        "--subjects", nargs="+", required=True,
        help="Model specs to test, e.g. mock, anthropic:claude-sonnet-5, google:gemini-3.6-flash, "
             "openai:gpt-4o, deepseek:deepseek-chat, groq:openai/gpt-oss-120b, "
             "mistral:mistral-small-latest, xai:grok-4-fast-non-reasoning, "
             "hf-local:meta-llama/Llama-3.1-8B-Instruct",
    )
    p_run.add_argument(
        "--judge", default=None,
        help="Model spec to use as the judge. Omit to use the crude rule-based "
             "fallback (offline-testing only, not for real results).",
    )
    p_run.add_argument("--out", default="results", help="Output path prefix. Writes <out>.json, <out>.csv, <out>.md")
    p_run.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between API calls.")
    p_run.add_argument(
        "--scenario-set", choices=["core", "full"], default="full",
        help="'core' = original 5 scenarios (1 issue type, high severity only). "
             "'full' = 72-scenario benchmark (4 issue types x 2 severities x "
             "4 categories x 2 intensities + controls). Default: full.",
    )
    p_run.add_argument(
        "--samples", type=int, default=1,
        help="Repeat each (model, scenario) trial this many times, for "
             "statistical robustness against a single noisy sample. Default: 1.",
    )
    p_run.set_defaults(func=cmd_run)

    # --- annotate ---
    p_ann = subparsers.add_parser("annotate", help="Human-annotate a sample of trials from a run.")
    p_ann.add_argument("--results", required=True, help="Path to a results.json from `run`")
    p_ann.add_argument("--annotator", required=True, help="Your annotator id/name")
    p_ann.add_argument("--out", required=True, help="Path to save/resume annotations JSON")
    p_ann.add_argument("--n", type=int, default=20, help="Number of trials to sample")
    p_ann.add_argument("--seed", type=int, default=0, help="Sampling seed (for reproducible samples)")
    p_ann.add_argument(
        "--no-stratify", action="store_true",
        help="Sample uniformly at random instead of stratifying by scenario category",
    )
    p_ann.set_defaults(func=cmd_annotate)

    # --- agreement ---
    p_agr = subparsers.add_parser("agreement", help="Compare judge scores vs. human annotations.")
    p_agr.add_argument("--results", required=True, help="Path to a results.json from `run`")
    p_agr.add_argument("--annotations", required=True, help="Path to an annotations JSON from `annotate`")
    p_agr.add_argument("--annotator", default=None, help="Restrict to one annotator id (default: all pooled)")
    p_agr.add_argument("--out", default="agreement.md", help="Output markdown report path")
    p_agr.set_defaults(func=cmd_agreement)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

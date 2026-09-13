"""
Compares the LLM judge's scores against human annotations for the same
trials — the validation step flagged in the README ("spot-check judged
trials by hand before trusting aggregate numbers").

No external stats dependency: Cohen's kappa is implemented directly since
it's a small, well-defined formula and we don't want a sklearn/numpy
requirement just for this.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from annotation import HumanAnnotation, trial_key
from pipeline import TrialResult


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """
    pairs: list of (rater_a_label, rater_b_label) for the same items.
    Returns Cohen's kappa, or float('nan') if undefined (e.g. <2 items,
    or one rater used only a single label with perfect agreement).
    """
    n = len(pairs)
    if n == 0:
        return float("nan")

    labels = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    if len(labels) < 2:
        return float("nan")  # no variation to measure agreement over

    a_counts = Counter(p[0] for p in pairs)
    b_counts = Counter(p[1] for p in pairs)

    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((a_counts[l] / n) * (b_counts[l] / n) for l in labels)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def percent_agreement(pairs: list[tuple]) -> float:
    if not pairs:
        return float("nan")
    return sum(1 for a, b in pairs if a == b) / len(pairs)


@dataclass
class AgreementSummary:
    n_compared: int
    disclosed_agreement: float
    disclosure_grade_agreement: float
    disclosure_grade_kappa: float
    disagreements: list[dict]


def compare(
    results: list[TrialResult],
    annotations: list[HumanAnnotation],
    annotator_id: str | None = None,
) -> AgreementSummary:
    """
    Matches each human annotation to its corresponding judged TrialResult
    (same model + scenario_id) and computes agreement statistics.

    If annotator_id is given, only that annotator's annotations are used;
    otherwise all annotators are pooled (fine for a single annotator, but
    if you have multiple, consider running this once per annotator_id too,
    since pooling can mask systematic disagreement with one specific rater).
    """
    judged_by_key = {trial_key(t): t for t in results}

    disclosed_pairs: list[tuple[bool, bool]] = []
    grade_pairs: list[tuple[str, str]] = []
    disagreements: list[dict] = []

    for ann in annotations:
        if annotator_id and ann.annotator_id != annotator_id:
            continue
        trial = judged_by_key.get(ann.key())
        if trial is None or trial.judge is None:
            continue  # no matching judged trial to compare against

        j = trial.judge
        disclosed_pairs.append((bool(j["disclosed"]), ann.disclosed))
        grade_pairs.append((j["disclosure_grade"], ann.disclosure_grade))

        if j["disclosure_grade"] != ann.disclosure_grade or bool(j["disclosed"]) != ann.disclosed:
            disagreements.append({
                "model": trial.model,
                "scenario_id": trial.scenario_id,
                "category": trial.category,
                "judge_disclosed": j["disclosed"],
                "human_disclosed": ann.disclosed,
                "judge_grade": j["disclosure_grade"],
                "human_grade": ann.disclosure_grade,
                "judge_rationale": j.get("rationale", ""),
                "human_rationale": ann.rationale,
            })

    return AgreementSummary(
        n_compared=len(grade_pairs),
        disclosed_agreement=percent_agreement(disclosed_pairs),
        disclosure_grade_agreement=percent_agreement(grade_pairs),
        disclosure_grade_kappa=cohens_kappa(grade_pairs),
        disagreements=disagreements,
    )


def build_markdown_report(summary: AgreementSummary) -> str:
    lines = ["# Judge vs. Human Annotation Agreement\n"]
    lines.append(f"Trials compared: **{summary.n_compared}**\n")

    if summary.n_compared == 0:
        lines.append(
            "No overlapping (model, scenario) pairs found between the judged "
            "results and the human annotations — check that annotations were "
            "run against the same results file.\n"
        )
        return "\n".join(lines)

    lines.append(f"- Agreement on *disclosed at all* (binary): **{summary.disclosed_agreement:.0%}**")
    lines.append(f"- Agreement on *disclosure grade* (3-way): **{summary.disclosure_grade_agreement:.0%}**")
    kappa = summary.disclosure_grade_kappa
    kappa_str = f"{kappa:.2f}" if kappa == kappa else "undefined (no label variation)"
    lines.append(f"- Cohen's kappa on *disclosure grade*: **{kappa_str}**")
    lines.append("")
    lines.append(
        "Rough kappa interpretation (Landis & Koch): <0 poor, 0.00–0.20 slight, "
        "0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 "
        "almost perfect.\n"
    )

    if not summary.disagreements:
        lines.append("**No disagreements found** — judge and human matched on every compared trial.\n")
        return "\n".join(lines)

    lines.append(f"## Disagreements ({len(summary.disagreements)})\n")
    lines.append("| Model | Scenario | Judge: disclosed / grade | Human: disclosed / grade | Judge rationale | Human rationale |")
    lines.append("|---|---|---|---|---|---|")
    for d in summary.disagreements:
        lines.append(
            f"| {d['model']} | {d['scenario_id']} | "
            f"{d['judge_disclosed']} / {d['judge_grade']} | "
            f"{d['human_disclosed']} / {d['human_grade']} | "
            f"{d['judge_rationale']} | {d['human_rationale']} |"
        )
    lines.append("")

    return "\n".join(lines)

"""Build a small Chinese reader entry from the refitted, verified SBA evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any, TypedDict

from creditvintage.sba_credit import METHODS, verify_sba_benchmark

TEMPLATE = Path(__file__).parent / "templates" / "sba_brief.html"
LABELS = {"constant": "常数基线", "logistic": "Logistic 原始模型", "isotonic": "独立保序校准"}


class Policy(TypedDict):
    method: str
    capacity_percent: int
    reviewed: int
    captured: int
    total_events: int


def build_brief(source: Path, destination: Path) -> dict[str, Any]:
    """Verify before writing; publish aggregate counts, never borrower-level rows."""
    if destination.exists():
        raise FileExistsError(f"Choose an empty brief destination: {destination}")
    verification = verify_sba_benchmark(source)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    with (source / "policy_metrics.csv").open(encoding="utf-8", newline="") as handle:
        retained = list(csv.DictReader(handle))
    policies: list[Policy] = [
        {
            "method": row["method"],
            "capacity_percent": round(float(row["capacity"]) * 100),
            "reviewed": int(row["reviewed"]),
            "captured": int(row["captured"]),
            "total_events": int(row["total_events"]),
        }
        for row in retained
    ]
    data: dict[str, Any] = {
        "schema": "creditvintage-sba-brief-v1",
        "target": summary["target"],
        "source_records": summary["source"]["records"],
        "test_records": summary["cohorts"]["test"]["records"],
        "test_events": summary["cohorts"]["test"]["events"],
        "cohorts": summary["cohorts"],
        "metrics": summary["test_metrics"],
        "policies": policies,
        "evidence_sha256": {
            name: manifest["files_sha256"][name]
            for name in ("summary.json", "policy_metrics.csv", "predictions.csv", "protocol.json")
        },
    }
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # A JSON script element must not be closed by text from a future source notice.
    script_data = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")

    def cell(value: object) -> str:
        return html.escape(str(value))

    policy_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{cell(value)}</td>"
            for value in (
                LABELS[row["method"]],
                f"{row['capacity_percent']}%",
                f"{row['reviewed']:,}",
                f"{row['captured']} / {row['total_events']}",
                f"{row['captured'] / row['reviewed']:.2%}",
                f"{row['captured'] / row['total_events']:.2%}",
            )
        )
        + "</tr>"
        for row in policies
    )
    metric_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{cell(value)}</td>"
            for value in (
                LABELS[method],
                f"{data['metrics'][method]['roc_auc']:.4f}",
                f"{data['metrics'][method]['brier']:.6f}",
            )
        )
        + "</tr>"
        for method in METHODS
    )
    cards = []
    for method in METHODS:
        row = next(x for x in policies if x["method"] == method and x["capacity_percent"] == 10)
        cards.append(
            f'<article class="result-card {method}" data-method="{method}">'
            f'<h3>{LABELS[method]}</h3><p class="found"><strong class="captured">'
            f"{row['captured']}</strong><span> / {row['total_events']} 笔事件</span></p>"
            f'<meter min="0" max="{row["total_events"]}" value="{row["captured"]}" '
            f'aria-label="{LABELS[method]}捕获事件数">{row["captured"]}</meter>'
            '<dl><div><dt>精确率 / Precision</dt><dd class="precision">'
            f"{row['captured'] / row['reviewed']:.2%}</dd></div>"
            '<div><dt>事件捕获率 / Capture</dt><dd class="capture">'
            f"{row['captured'] / row['total_events']:.2%}</dd></div></dl>"
            '<p class="outside">复核名单外仍有 '
            f"{row['total_events'] - row['captured']} 笔事件。</p>"
            "</article>"
        )
    replacements = {
        "@@BRIEF_DATA@@": script_data,
        "@@POLICY_ROWS@@": policy_rows,
        "@@METRIC_ROWS@@": metric_rows,
        "@@RESULT_CARDS@@": "\n".join(cards),
        "@@POLICY_SHA@@": cell(data["evidence_sha256"]["policy_metrics.csv"]),
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    for marker, value in replacements.items():
        if page.count(marker) != 1:
            raise ValueError(f"Brief template marker must occur exactly once: {marker}")
        page = page.replace(marker, value)
    if "@@" in page:
        raise ValueError("Unresolved brief template marker")
    destination.mkdir(parents=True)
    (destination / "index.html").write_text(page, encoding="utf-8", newline="\n")
    (destination / "data.json").write_text(encoded, encoding="utf-8", newline="\n")
    return {
        "status": "verified",
        "source_refitted": verification["refitted"],
        "test_records": data["test_records"],
        "test_events": data["test_events"],
        "fixed_comparisons": len(policies),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_brief(args.source, args.destination), indent=2))

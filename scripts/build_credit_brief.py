"""Build Chinese and English reader entries from the same verified SBA evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any, TypedDict

from creditvintage.sba_credit import METHODS, verify_sba_benchmark

TEMPLATE = Path(__file__).parent / "templates" / "sba_brief.html"
BODY_TEMPLATES = {
    "zh-CN": TEMPLATE.with_name("sba_brief_zh.html"),
    "en": TEMPLATE.with_name("sba_brief_en.html"),
}
LABELS = {
    "zh-CN": {"constant": "常数基线", "logistic": "Logistic 原始模型", "isotonic": "独立保序校准"},
    "en": {
        "constant": "Constant baseline",
        "logistic": "Raw Logistic",
        "isotonic": "Independent isotonic",
    },
}
COPY = {
    "zh-CN": {
        "title": "同样的复核名额，看到不同的结果 · CreditVintage",
        "description": (
            "两分钟读懂 35,000 笔真实 SBA 贷款研究：切换固定复核名额，"
            "比较模型、校准、精确率和事件捕获率。无需安装。"
        ),
        "events": "笔事件",
        "meter": "{method}捕获事件数",
        "precision": "精确率 / Precision",
        "capture": "事件捕获率 / Capture",
        "outside": "复核名单外仍有 {outside} 笔事件。",
        "status": "当前比较：每种方法复核 {reviewed} / {test_records} 笔贷款。",
        "explanation": (
            "原始模型找到 {captured} 笔事件：精确率是 {captured} / {reviewed} = {precision}，"
            "事件捕获率是 {captured} / {events} = {capture}。两个比例回答不同的问题。"
        ),
        "copied": "链接已复制。只包含固定容量，不包含个人数据。",
        "copy_fallback": "请复制地址栏链接，当前容量已保存在链接中。",
    },
    "en": {
        "title": "Same review budget. Different results. · CreditVintage",
        "description": (
            "Explore a 35,000-loan SBA study in two minutes. Compare fixed review budgets, "
            "raw and calibrated scores, precision and event capture. No installation."
        ),
        "events": "recorded events",
        "meter": "{method}: captured events",
        "precision": "Precision",
        "capture": "Event capture",
        "outside": "{outside} events remain outside this review list.",
        "status": "Current comparison: each method reviews {reviewed} / {test_records} loans.",
        "explanation": (
            "Raw Logistic captures {captured} events: precision is {captured} / {reviewed} = "
            "{precision}; event capture is {captured} / {events} = {capture}. "
            "The denominators answer different questions."
        ),
        "copied": "Link copied. It contains only the fixed capacity, no personal data.",
        "copy_fallback": "Copy the address-bar link; it now preserves this capacity.",
    },
}


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

    def script_json(value: str) -> str:
        return value.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")

    pages = {
        locale: render_page(data, policies, script_json(encoded), locale)
        for locale in BODY_TEMPLATES
    }
    # Resolve and validate both templates before creating any output.
    destination.mkdir(parents=True)
    (destination / "en").mkdir()
    for locale, page in pages.items():
        target = destination if locale == "zh-CN" else destination / "en"
        (target / "index.html").write_text(page, encoding="utf-8", newline="\n")
    (destination / "data.json").write_text(encoded, encoding="utf-8", newline="\n")
    return {
        "status": "verified",
        "source_refitted": verification["refitted"],
        "test_records": data["test_records"],
        "test_events": data["test_events"],
        "fixed_comparisons": len(policies),
        "languages": list(pages),
    }


def render_page(data: dict[str, Any], policies: list[Policy], script_data: str, locale: str) -> str:
    labels = LABELS[locale]
    copy = COPY[locale]

    def cell(value: object) -> str:
        return html.escape(str(value))

    policy_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{cell(value)}</td>"
            for value in (
                labels[row["method"]],
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
                labels[method],
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
            f'<h3>{cell(labels[method])}</h3><p class="found"><strong class="captured">'
            f"{row['captured']}</strong><span> / {row['total_events']} "
            f"{cell(copy['events'])}</span></p>"
            f'<meter min="0" max="{row["total_events"]}" value="{row["captured"]}" '
            f'aria-label="{cell(copy["meter"].format(method=labels[method]))}">'
            f"{row['captured']}</meter>"
            f'<dl><div><dt>{cell(copy["precision"])}</dt><dd class="precision">'
            f"{row['captured'] / row['reviewed']:.2%}</dd></div>"
            f'<div><dt>{cell(copy["capture"])}</dt><dd class="capture">'
            f"{row['captured'] / row['total_events']:.2%}</dd></div></dl>"
            '<p class="outside">'
            f"{cell(copy['outside'].format(outside=row['total_events'] - row['captured']))}</p>"
            "</article>"
        )
    raw = next(x for x in policies if x["method"] == "logistic" and x["capacity_percent"] == 10)
    initial_status = copy["status"].format(
        reviewed=f"{raw['reviewed']:,}", test_records=f"{data['test_records']:,}"
    )
    initial_explanation = copy["explanation"].format(
        captured=raw["captured"],
        reviewed=f"{raw['reviewed']:,}",
        precision=f"{raw['captured'] / raw['reviewed']:.2%}",
        events=raw["total_events"],
        capture=f"{raw['captured'] / raw['total_events']:.2%}",
    )
    # The shell owns behavior/styles; localized bodies contain prose and links.
    page = TEMPLATE.read_text(encoding="utf-8")
    if page.count("@@BODY@@") != 1:
        raise ValueError("Brief template body must occur exactly once")
    page = page.replace("@@BODY@@", BODY_TEMPLATES[locale].read_text(encoding="utf-8"))
    ui_json = json.dumps(copy, ensure_ascii=False, allow_nan=False)
    ui_json = ui_json.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    suffix = "" if locale == "zh-CN" else "en/"
    replacements = {
        "@@LANG@@": locale,
        "@@TITLE@@": cell(copy["title"]),
        "@@DESCRIPTION@@": cell(copy["description"]),
        "@@CANONICAL_URL@@": f"https://dev-belly.github.io/CreditVintage/sba-brief/{suffix}",
        "@@BRIEF_DATA@@": script_data,
        "@@UI_COPY@@": ui_json,
        "@@INITIAL_STATUS@@": cell(initial_status),
        "@@INITIAL_EXPLANATION@@": cell(initial_explanation),
        "@@POLICY_ROWS@@": policy_rows,
        "@@METRIC_ROWS@@": metric_rows,
        "@@RESULT_CARDS@@": "\n".join(cards),
        "@@POLICY_SHA@@": cell(data["evidence_sha256"]["policy_metrics.csv"]),
    }
    for marker, value in replacements.items():
        if page.count(marker) != 1:
            raise ValueError(f"Brief template marker must occur exactly once: {marker}")
        page = page.replace(marker, value)
    if "@@" in page:
        raise ValueError("Unresolved brief template marker")
    return page


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_brief(args.source, args.destination), indent=2))

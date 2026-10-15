"""Generate, evaluate and verify a point-in-time credit cohort experiment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from creditvintage.artifacts import verify_result, write_result
from creditvintage.benchmark import benchmark, verify_benchmark, write_benchmark
from creditvintage.business import verify_analysis, write_analysis
from creditvintage.card_credit import verify_card_benchmark, write_card_benchmark
from creditvintage.configuration import load_config, preflight
from creditvintage.core import DataContractError, load_csv_bundle
from creditvintage.lineage import verify_lineage, write_demo
from creditvintage.model import evaluate
from creditvintage.monitoring import monitor, verify_monitor, write_monitor
from creditvintage.public_credit import verify_public_benchmark, write_public_benchmark
from creditvintage.review_bundle import export_reviews, verify_review_bundle
from creditvintage.reviews import initialize_reviews, record_review, review_status
from creditvintage.sample import generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="creditvintage")
    commands = parser.add_subparsers(dest="command", required=True)

    def data_arguments(command: argparse.ArgumentParser, default_out: str | None) -> None:
        if default_out is not None:
            command.add_argument("--out", type=Path, default=Path(default_out))
        command.add_argument("--config", type=Path, help="JSON with all seven calendar dates")
        command.add_argument("--seed", type=int, help="Override the config's random seed")
        command.add_argument("--per-vintage", type=int, default=80)
        command.add_argument("--applications", type=Path)
        command.add_argument("--features", type=Path)
        command.add_argument("--performance", type=Path)

    run = commands.add_parser("run", help="Run a fixed synthetic or supplied event experiment")
    data_arguments(run, "outputs/latest")
    inspect = commands.add_parser(
        "preflight", help="Check inputs and mature evaluation windows without fitting or writing"
    )
    data_arguments(inspect, None)
    public = commands.add_parser(
        "public-benchmark", help="Run the separate frozen UCI static credit experiment"
    )
    public.add_argument("--out", type=Path, default=Path("outputs/public-credit"))
    verify_public = commands.add_parser("verify-public", help="Refit the UCI experiment offline")
    verify_public.add_argument("directory", type=Path)
    card = commands.add_parser(
        "card-benchmark", help="Compare models on 30,000 observed UCI accounts"
    )
    card.add_argument("--out", type=Path, default=Path("outputs/card-credit"))
    verify_card = commands.add_parser(
        "verify-card", help="Refit and replay all five card-data splits"
    )
    verify_card.add_argument("directory", type=Path)
    compare = commands.add_parser(
        "benchmark", help="Compare frozen policies at equal review budgets"
    )
    data_arguments(compare, "outputs/benchmark")
    compare.add_argument("--capacities", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    compare.add_argument("--review-cost-usd", type=float, default=5.0)
    check_benchmark = commands.add_parser("verify-benchmark", help="Recompute comparison evidence")
    check_benchmark.add_argument("directory", type=Path)
    early = commands.add_parser("monitor", help="Monitor score drift before current labels mature")
    data_arguments(early, "outputs/monitor")
    early.add_argument("--as-of", type=date.fromisoformat, required=True)
    check = commands.add_parser(
        "verify", help="Check output digests and independently recompute metrics"
    )
    check.add_argument("directory", type=Path)
    check_monitor = commands.add_parser(
        "verify-monitor", help="Check score-monitoring digests and recompute PSI"
    )
    check_monitor.add_argument("directory", type=Path)
    business = commands.add_parser("analyze", help="Run verified SQL cohort and review analysis")
    business.add_argument("report", type=Path)
    business.add_argument("--out", type=Path, default=Path("outputs/analysis"))
    check_business = commands.add_parser("verify-analysis", help="Replay a portable SQL analysis")
    check_business.add_argument("directory", type=Path)
    review_init = commands.add_parser("review-init", help="Create a label-free review workspace")
    review_init.add_argument("analysis", type=Path)
    review_init.add_argument("--db", type=Path, required=True)
    review_event = commands.add_parser("review", help="Record one transactional review action")
    review_event.add_argument("database", type=Path)
    review_event.add_argument("application_id")
    review_event.add_argument("action", choices=("claim", "release", "resolve", "reopen"))
    review_event.add_argument("--actor", required=True)
    review_event.add_argument("--version", type=int, required=True)
    review_event.add_argument("--request-id", required=True)
    review_event.add_argument("--note", default="")
    review_event.add_argument("--evidence", default="")
    review_event.add_argument("--outcome", default="")
    review_check = commands.add_parser("review-status", help="Replay review events and count work")
    review_check.add_argument("database", type=Path)
    review_export = commands.add_parser("review-export", help="Deliver portable review evidence")
    review_export.add_argument("database", type=Path)
    review_export.add_argument("--out", type=Path, required=True)
    review_verify = commands.add_parser("verify-reviews", help="Replay portable review evidence")
    review_verify.add_argument("directory", type=Path)
    review_verify.add_argument("--expected-sha256", help="Trusted digest saved outside this bundle")
    lineage = commands.add_parser(
        "lineage-demo", help="Connect PITBridge sources to credit evaluation"
    )
    lineage.add_argument("--out", type=Path, default=Path("outputs/lineage"))
    lineage.add_argument("--seed", type=int, default=20260927)
    lineage.add_argument("--per-vintage", type=int, default=20)
    check_lineage = commands.add_parser(
        "verify-lineage", help="Replay source features and refit the model"
    )
    check_lineage.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "card-benchmark":
            summary = write_card_benchmark(args.out)
            print(json.dumps({"out": str(args.out), "results": summary["results"]}, indent=2))
            return 0
        if args.command == "verify-card":
            print(json.dumps(verify_card_benchmark(args.directory), indent=2))
            return 0
        if args.command == "public-benchmark":
            summary = write_public_benchmark(args.out)
            print(json.dumps({"out": str(args.out), **summary["test_metrics"]}, indent=2))
            return 0
        if args.command == "verify-public":
            print(json.dumps(verify_public_benchmark(args.directory), indent=2))
            return 0
        if args.command == "review-export":
            print(json.dumps(export_reviews(args.database, args.out), indent=2))
            return 0
        if args.command == "verify-reviews":
            print(
                json.dumps(
                    verify_review_bundle(args.directory, expected_sha256=args.expected_sha256),
                    indent=2,
                )
            )
            return 0
        if args.command == "verify-benchmark":
            print(json.dumps(verify_benchmark(args.directory), indent=2))
            return 0
        if args.command == "review-init":
            print(json.dumps(initialize_reviews(args.analysis, args.db), indent=2))
            return 0
        if args.command == "review":
            print(
                json.dumps(
                    record_review(
                        args.database,
                        args.application_id,
                        args.action,
                        args.actor,
                        args.version,
                        args.request_id,
                        note=args.note,
                        evidence=args.evidence,
                        outcome=args.outcome,
                    ),
                    indent=2,
                )
            )
            return 0
        if args.command == "review-status":
            print(json.dumps(review_status(args.database), indent=2))
            return 0
        if args.command == "analyze":
            print(json.dumps(write_analysis(args.report, args.out), indent=2))
            return 0
        if args.command == "verify-analysis":
            print(json.dumps(verify_analysis(args.directory), indent=2))
            return 0
        if args.command == "lineage-demo":
            print(json.dumps(write_demo(args.out, args.seed, args.per_vintage), indent=2))
            return 0
        if args.command == "verify-lineage":
            print(json.dumps(verify_lineage(args.directory), indent=2))
            return 0
        if args.command == "verify":
            print(json.dumps(verify_result(args.directory), indent=2))
            return 0
        if args.command == "verify-monitor":
            print(json.dumps(verify_monitor(args.directory), indent=2))
            return 0
        config = load_config(args.config, seed=args.seed)
        inputs = (args.applications, args.features, args.performance)
        if any(path is not None for path in inputs) and not all(
            path is not None for path in inputs
        ):
            raise DataContractError("Supply all three CSV files together")
        if all(path is not None for path in inputs):
            applications, features, performance = load_csv_bundle(*inputs)
            data_kind = "user_supplied"
        else:
            applications, features, performance = generate(config.seed, args.per_vintage)
            data_kind = "synthetic"
        if args.command == "preflight":
            print(
                json.dumps(
                    preflight(applications, features, performance, config, data_kind=data_kind),
                    indent=2,
                )
            )
            return 0
        if args.command == "benchmark":
            compared = benchmark(
                applications,
                features,
                performance,
                config,
                data_kind=data_kind,
                capacities=tuple(args.capacities),
                review_cost_usd=args.review_cost_usd,
            )
            write_benchmark(compared, args.out)
            print(json.dumps({"out": str(args.out), **compared.summary["test_metrics"]}, indent=2))
        elif args.command == "monitor":
            monitored = monitor(
                applications,
                features,
                performance,
                args.as_of,
                config,
                data_kind=data_kind,
            )
            write_monitor(monitored, args.out)
            print(json.dumps({"out": str(args.out), **monitored.summary}, indent=2))
        else:
            result = evaluate(
                applications,
                features,
                performance,
                config,
                data_kind=data_kind,
            )
            write_result(result, args.out)
            print(json.dumps({"out": str(args.out), **result.summary["test_metrics"]}, indent=2))
    except (DataContractError, OSError, ValueError) as exc:
        print(f"creditvintage: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Generate, evaluate and verify a point-in-time credit cohort experiment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from creditvintage.artifacts import verify_result, write_result
from creditvintage.core import DataContractError, load_csv_bundle
from creditvintage.model import Config, evaluate
from creditvintage.monitoring import monitor, verify_monitor, write_monitor
from creditvintage.sample import generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="creditvintage")
    commands = parser.add_subparsers(dest="command", required=True)

    def data_arguments(command: argparse.ArgumentParser, default_out: str) -> None:
        command.add_argument("--out", type=Path, default=Path(default_out))
        command.add_argument("--seed", type=int, default=20260927)
        command.add_argument("--per-vintage", type=int, default=80)
        command.add_argument("--applications", type=Path)
        command.add_argument("--features", type=Path)
        command.add_argument("--performance", type=Path)

    run = commands.add_parser("run", help="Run a fixed synthetic or supplied event experiment")
    data_arguments(run, "outputs/latest")
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
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            print(json.dumps(verify_result(args.directory), indent=2))
            return 0
        if args.command == "verify-monitor":
            print(json.dumps(verify_monitor(args.directory), indent=2))
            return 0
        inputs = (args.applications, args.features, args.performance)
        if any(path is not None for path in inputs) and not all(
            path is not None for path in inputs
        ):
            raise DataContractError("Supply all three CSV files together")
        if all(path is not None for path in inputs):
            applications, features, performance = load_csv_bundle(*inputs)
            data_kind = "user_supplied"
        else:
            applications, features, performance = generate(args.seed, args.per_vintage)
            data_kind = "synthetic"
        if args.command == "monitor":
            monitored = monitor(
                applications,
                features,
                performance,
                args.as_of,
                Config(seed=args.seed),
                data_kind=data_kind,
            )
            write_monitor(monitored, args.out)
            print(json.dumps({"out": str(args.out), **monitored.summary}, indent=2))
        else:
            result = evaluate(
                applications,
                features,
                performance,
                Config(seed=args.seed),
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

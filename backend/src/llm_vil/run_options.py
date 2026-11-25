"""Run-safety command line flags (paused start, day limit, save retention)."""

from __future__ import annotations

import argparse
from typing import Any


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Safety flags shared by ``llm-vil serve`` and ``python -m llm_vil.server.app``."""
    parser.add_argument("--paused", action=argparse.BooleanOptionalAction, default=True,
                        help="start paused and wait for ▶ 재생 (default)")
    parser.add_argument("--autostart", action="store_true",
                        help="start running immediately (same as --no-paused)")
    parser.add_argument("--max-days", type=int, default=None, metavar="N",
                        help="auto-pause after N sim-days per run (0 = unlimited; "
                             "default: clock.max_run_days in village.yaml, 7)")
    parser.add_argument("--max-checkpoints", type=int, default=None, metavar="N",
                        help="automatic saves to keep (default: storage.max_auto_checkpoints "
                             "in village.yaml, 10); labeled saves are never pruned")


def run_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_days is not None and args.max_days < 0:
        raise SystemExit("--max-days must be >= 0")
    if args.max_checkpoints is not None and args.max_checkpoints < 1:
        raise SystemExit("--max-checkpoints must be >= 1")
    out: dict[str, Any] = {"paused": args.paused and not args.autostart,
                           "max_checkpoints": args.max_checkpoints}
    if args.max_days is not None:
        out["max_run_days"] = args.max_days
    return out

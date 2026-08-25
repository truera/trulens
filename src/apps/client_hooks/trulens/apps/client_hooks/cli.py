"""Command-line interface for coding-agent hook instrumentation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from typing import Optional, Sequence

from trulens.apps.client_hooks import service


def _command(client: str) -> str:
    installed_command = shutil.which("trulens-client-hooks")
    if installed_command:
        return f"{shlex.quote(installed_command)} ingest --client {client}"
    python = str(Path(sys.executable).resolve())
    return (
        f"{shlex.quote(python)} -m trulens.apps.client_hooks "
        f"ingest --client {client}"
    )


def _configuration(client: str) -> dict:
    command = _command(client)
    if client == "claude":
        return {
            "hooks": {
                event: [{"hooks": [{"type": "command", "command": command}]}]
                for event in (
                    "UserPromptSubmit",
                    "PreToolUse",
                    "PostToolUse",
                    "PostToolUseFailure",
                    "SubagentStart",
                    "SubagentStop",
                    "Stop",
                    "StopFailure",
                )
            }
        }
    return {
        "version": 1,
        "hooks": {
            event: [{"command": command}]
            for event in (
                "beforeSubmitPrompt",
                "preToolUse",
                "postToolUse",
                "postToolUseFailure",
                "subagentStart",
                "subagentStop",
                "beforeShellExecution",
                "afterShellExecution",
                "beforeMCPExecution",
                "afterMCPExecution",
                "afterAgentResponse",
                "stop",
            )
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trulens.apps.client_hooks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser(
        "ingest", help="Read one client hook payload from stdin."
    )
    ingest.add_argument("--client", choices=("claude", "cursor"), required=True)
    subparsers.add_parser("validate", help="Validate adapter configuration.")
    install = subparsers.add_parser(
        "config", help="Print a hook configuration snippet."
    )
    install.add_argument(
        "--client", choices=("claude", "cursor"), required=True
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        destination = os.environ.get("TRULENS_HOOKS_DESTINATION", "local")
        if destination not in {"local", "snowflake"}:
            sys.stderr.write(
                "TRULENS_HOOKS_DESTINATION must be local or snowflake.\n"
            )
            return 1
        if destination == "snowflake" and not os.environ.get(
            "TRULENS_HOOKS_SNOWFLAKE_CONNECTION"
        ):
            sys.stderr.write(
                "Snowflake export requires "
                "TRULENS_HOOKS_SNOWFLAKE_CONNECTION.\n"
            )
            return 1
        return 0
    if args.command == "config":
        sys.stdout.write(
            json.dumps(_configuration(args.client), indent=2) + "\n"
        )
        return 0
    if args.command == "ingest":
        try:
            payload = json.load(sys.stdin)
            if not isinstance(payload, dict):
                raise ValueError("Hook payload must be a JSON object.")
            if not service.HookService().ingest(args.client, payload):
                sys.stderr.write("TruLens hook export will be retried.\n")
        except Exception as exc:
            # Hooks are observability-only and must never block the client.
            sys.stderr.write(f"TruLens hook instrumentation failed: {exc}\n")
        sys.stdout.write("{}\n")
        return 0
    return 1

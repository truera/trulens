"""Command-line entry point for coding-agent hooks."""

from trulens.apps.client_hooks import cli

if __name__ == "__main__":
    raise SystemExit(cli.main())

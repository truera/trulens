"""Command-line interface for coding-agent hook instrumentation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
from typing import Any, MutableMapping, Optional, Sequence

from trulens.core.otel.client_hooks import clients
from trulens.core.otel.client_hooks import service
from trulens.core.otel.client_hooks import worker

_MARKER = "trulens-client-hooks"


def _command(client: str) -> str:
    executable = shutil.which("trulens-client-hooks")
    if executable:
        return f"{shlex.quote(executable)} ingest {client}"
    return (
        f"{shlex.quote(sys.executable)} -m trulens.core.otel.client_hooks "
        f"ingest {client}"
    )


def _configuration(spec: clients.ClientSpec) -> dict:
    return dict(spec.build_config(_command(spec.name)))


def _plugin_source(spec: clients.ClientSpec) -> Optional[str]:
    return spec.build_plugin(_command(spec.name))


def _is_plugin_client(spec: clients.ClientSpec) -> bool:
    return spec.plugin_builder is not None


def _write_atomic(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            os.chmod(temporary_name, path.stat().st_mode)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _merge(
    target: MutableMapping[str, Any], fragment: MutableMapping[str, Any]
):
    """Recursively merge hook configuration while preserving unrelated values."""

    for key, value in fragment.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        elif isinstance(value, list) and isinstance(target.get(key), list):
            existing = target[key]
            for item in value:
                if item not in existing:
                    existing.append(item)
        else:
            target[key] = value


def _is_managed_hook(value: Any, spec: clients.ClientSpec) -> bool:
    if isinstance(value, dict):
        command = value.get("command")
        if isinstance(command, str) and (
            f"ingest {spec.name}" in command
            or f"ingest --client {spec.name}" in command
        ):
            return True
        return any(_is_managed_hook(item, spec) for item in value.values())
    if isinstance(value, list):
        return any(_is_managed_hook(item, spec) for item in value)
    return False


def _remove_managed_hooks(
    config: MutableMapping[str, Any], spec: clients.ClientSpec
):
    hooks = config.get("hooks")
    if not isinstance(hooks, dict):
        return
    for event in tuple(hooks):
        entries = hooks[event]
        if not isinstance(entries, list):
            continue
        retained = [
            item for item in entries if not _is_managed_hook(item, spec)
        ]
        if retained:
            hooks[event] = retained
        else:
            del hooks[event]
    if not hooks:
        config.pop("hooks", None)
    marker = config.get("trulens")
    if isinstance(marker, dict) and marker.get("managed_by") == _MARKER:
        config.pop("trulens", None)


def _config_path(spec: clients.ClientSpec, project: bool) -> Path:
    path = spec.project_config_path if project else spec.user_config_path
    if path is None:
        raise ValueError(f"Client {spec.name} does not support project hooks.")
    return path.expanduser()


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".trulens.bak")


def _install_plugin(
    spec: clients.ClientSpec, project: bool, dry_run: bool
) -> int:
    path = _config_path(spec, project)
    rendered = _plugin_source(spec) or ""
    if not rendered.endswith("\n"):
        rendered += "\n"
    if dry_run:
        sys.stdout.write(rendered)
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, _backup_path(path))
    _write_atomic(path, rendered)
    sys.stdout.write(f"Installed TruLens hooks in {path}.\n")
    return 0


def _uninstall_plugin(
    spec: clients.ClientSpec, project: bool, dry_run: bool
) -> int:
    path = _config_path(spec, project)
    if not path.exists():
        sys.stdout.write(f"No hook configuration found at {path}.\n")
        return 0
    current = path.read_text()
    if dry_run:
        sys.stdout.write(
            current if not _is_managed_plugin(current, spec) else ""
        )
        return 0
    if not _is_managed_plugin(current, spec):
        sys.stdout.write(f"No TruLens-managed plugin found at {path}.\n")
        return 0
    shutil.copy2(path, _backup_path(path))
    path.unlink()
    sys.stdout.write(f"Removed TruLens hooks from {path}.\n")
    return 0


def _is_managed_plugin(content: str, spec: clients.ClientSpec) -> bool:
    return _MARKER in content and (
        f"ingest {spec.name}" in content
        or f"ingest --client {spec.name}" in content
    )


def _install(spec: clients.ClientSpec, project: bool, dry_run: bool) -> int:
    if _is_plugin_client(spec):
        return _install_plugin(spec, project, dry_run)
    path = _config_path(spec, project)
    current = json.loads(path.read_text()) if path.exists() else {}
    configuration = dict(_configuration(spec))
    if "version" in current:
        configuration.pop("version", None)
    _merge(current, configuration)
    current.setdefault("trulens", {})["managed_by"] = _MARKER
    rendered = json.dumps(current, indent=2) + "\n"
    if dry_run:
        sys.stdout.write(rendered)
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, _backup_path(path))
    _write_atomic(path, rendered)
    sys.stdout.write(f"Installed TruLens hooks in {path}.\n")
    return 0


def _uninstall(spec: clients.ClientSpec, project: bool, dry_run: bool) -> int:
    if _is_plugin_client(spec):
        return _uninstall_plugin(spec, project, dry_run)
    path = _config_path(spec, project)
    if not path.exists():
        sys.stdout.write(f"No hook configuration found at {path}.\n")
        return 0
    current = json.loads(path.read_text())
    _remove_managed_hooks(current, spec)
    rendered = json.dumps(current, indent=2) + "\n"
    if dry_run:
        sys.stdout.write(rendered)
        return 0
    shutil.copy2(path, _backup_path(path))
    _write_atomic(path, rendered)
    sys.stdout.write(f"Removed TruLens hooks from {path}.\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trulens-client-hooks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("client")
    config = subparsers.add_parser("config")
    config.add_argument("client")
    install = subparsers.add_parser("install")
    install.add_argument("client")
    install.add_argument("--project", action="store_true")
    install.add_argument("--dry-run", action="store_true")
    uninstall = subparsers.add_parser("uninstall")
    uninstall.add_argument("client")
    uninstall.add_argument("--project", action="store_true")
    uninstall.add_argument("--dry-run", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("client")
    status.add_argument("--project", action="store_true")
    subparsers.add_parser("clients")
    subparsers.add_parser("flush")
    subparsers.add_parser("worker", help=argparse.SUPPRESS)
    subparsers.add_parser("validate")
    return parser


def _validate() -> int:
    destination = os.environ.get("TRULENS_DESTINATION", "local").lower()
    if destination not in {"local", "database", "snowflake", "otlp"}:
        sys.stderr.write(
            "TRULENS_DESTINATION must be local, database, snowflake, or otlp.\n"
        )
        return 1
    if destination == "database" and not os.environ.get("TRULENS_DATABASE_URL"):
        sys.stderr.write("Database export requires TRULENS_DATABASE_URL.\n")
        return 1
    if destination == "snowflake" and not os.environ.get(
        "TRULENS_SNOWFLAKE_CONNECTION"
    ):
        sys.stderr.write(
            "Snowflake export requires TRULENS_SNOWFLAKE_CONNECTION.\n"
        )
        return 1
    if destination == "snowflake" and not all(
        os.environ.get(name)
        for name in (
            "TRULENS_SNOWFLAKE_DATABASE",
            "TRULENS_SNOWFLAKE_SCHEMA",
        )
    ):
        sys.stderr.write(
            "Snowflake export requires TRULENS_SNOWFLAKE_DATABASE and "
            "TRULENS_SNOWFLAKE_SCHEMA unless both are set in the connection profile.\n"
        )
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "clients":
        for spec in clients.list_clients():
            sys.stdout.write(f"{spec.name}\n")
        return 0
    if args.command == "validate":
        return _validate()
    if args.command == "flush":
        return 0 if service.HookService().flush() else 1
    if args.command == "worker":
        return worker.run_worker()
    if args.command in {"config", "install", "uninstall", "status"}:
        spec = clients.get_client(args.client)
        if args.command == "config":
            plugin = _plugin_source(spec)
            if plugin is not None:
                if not plugin.endswith("\n"):
                    plugin += "\n"
                sys.stdout.write(plugin)
                return 0
            sys.stdout.write(json.dumps(_configuration(spec), indent=2) + "\n")
            return 0
        if args.command == "install":
            return _install(spec, args.project, args.dry_run)
        if args.command == "uninstall":
            return _uninstall(spec, args.project, args.dry_run)
        path = _config_path(spec, args.project)
        installed = False
        if path.exists():
            if _is_plugin_client(spec):
                installed = _is_managed_plugin(path.read_text(), spec)
            else:
                current = json.loads(path.read_text())
                installed = _is_managed_hook(current.get("hooks", {}), spec)
        sys.stdout.write(
            f"{spec.name}: {'installed' if installed else 'not installed'}\n"
        )
        journal = service.HookService().journal
        state = journal.status()
        sys.stdout.write(
            "worker: "
            f"{'running' if worker.is_worker_running(journal.directory) else 'stopped'}; "
            f"pending={state['pending']}; claimed={state['claimed']}; "
            f"retrying={state['retrying']}; log={worker.worker_log_path(journal.directory)}\n"
        )
        return 0
    if args.command == "ingest":
        try:
            payload = json.load(sys.stdin)
            if not isinstance(payload, dict):
                raise ValueError("Hook payload must be a JSON object.")
            service.HookService().ingest(args.client, payload)
            if not worker.ensure_worker():
                sys.stderr.write("TruLens hook worker could not be started.\n")
        except Exception as exc:
            sys.stderr.write(f"TruLens hook instrumentation failed: {exc}\n")
        sys.stdout.write("{}\n")
        return 0
    return 1

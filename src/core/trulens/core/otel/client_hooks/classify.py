"""Classify coding-agent shell commands and edited paths.

Hook payloads carry the command text and the edited file path, but the capture
policy discards both unless the user opts in. These helpers turn that content
into small labels while it is still available, so a trace can show whether the
agent tested its change without recording what it ran or which file it touched.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import Dict, Iterable, List, Mapping, Optional, Pattern, Tuple

COMMAND_KINDS: Tuple[str, ...] = ("test", "build", "lint", "vcs", "other")
"""Every value `classify_command` can return."""

# When a compound command such as `git add . && pytest` contains more than one
# kind, the earliest kind in this tuple wins, so a test run is never hidden
# behind the commands around it.
_PRECEDENCE: Tuple[str, ...] = ("test", "build", "lint", "vcs")

# Each entry is a command prefix, matched at the start of a simple command once
# environment assignments and runner wrappers (see `_WRAPPERS`) are removed.
_DEFAULT_COMMAND_KIND_PATTERNS: Mapping[str, Tuple[str, ...]] = {
    "test": (
        "pytest",
        "py.test",
        "unittest",
        "tox",
        "nox",
        "npm test",
        "npm t",
        "npm run test",
        "yarn test",
        "yarn run test",
        "pnpm test",
        "pnpm run test",
        "bun test",
        "bun run test",
        "deno test",
        "jest",
        "vitest",
        "mocha",
        "playwright test",
        "make test",
        "make check",
        "cargo test",
        "cargo nextest",
        "go test",
        "mvn test",
        "mvn verify",
        "gradle test",
        "./gradlew test",
        "rspec",
        "phpunit",
        "ctest",
        "dotnet test",
        "swift test",
    ),
    "build": (
        "tsc",
        "npm run build",
        "yarn build",
        "yarn run build",
        "pnpm build",
        "pnpm run build",
        "bun run build",
        "cargo build",
        "cargo check",
        "go build",
        "mvn compile",
        "mvn package",
        "mvn install",
        "gradle build",
        "./gradlew build",
        "make",
        "cmake",
        "bazel build",
        "docker build",
        "dotnet build",
        "swift build",
    ),
    "lint": (
        "ruff",
        "flake8",
        "pylint",
        "mypy",
        "pyright",
        "black",
        "isort",
        "eslint",
        "prettier",
        "biome",
        "npm run lint",
        "yarn lint",
        "yarn run lint",
        "pnpm lint",
        "pnpm run lint",
        "make lint",
        "make format",
        "cargo clippy",
        "cargo fmt",
        "go vet",
        "gofmt",
        "golangci-lint",
        "pre-commit",
        "shellcheck",
    ),
    "vcs": (
        "git",
        "gh",
        "hg",
        "svn",
    ),
}

DEFAULT_TEST_FILE_GLOBS: Tuple[str, ...] = (
    "test_*.py",
    "*_test.py",
    "tests/**",
    "**/__tests__/**",
    "*.spec.*",
    "*.test.*",
    "conftest.py",
)
"""Paths treated as test files unless `TRULENS_TEST_FILE_GLOBS` is set."""

COMMAND_KIND_PATTERNS_ENV = "TRULENS_COMMAND_KIND_PATTERNS"
TEST_FILE_GLOBS_ENV = "TRULENS_TEST_FILE_GLOBS"

# Tokens that run another command, removed before matching so that
# `uv run pytest` and `python -m pytest` classify like `pytest`.
_WRAPPERS: Tuple[Tuple[str, ...], ...] = (
    ("sudo",),
    ("time",),
    ("env",),
    ("nice",),
    ("command",),
    ("exec",),
    ("uv", "run"),
    ("poetry", "run"),
    ("pipenv", "run"),
    ("pdm", "run"),
    ("hatch", "run"),
    ("rye", "run"),
    ("npx",),
    ("bunx",),
    ("pnpm", "exec"),
    ("pnpm", "dlx"),
    ("yarn", "dlx"),
    ("python", "-m"),
    ("python3", "-m"),
)
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEPARATORS = re.compile(r"&&|\|\||[;|&\n]")
# A prefix only matches as a whole word, but `make test-unit` and
# `npm run test:e2e` still count as `make test` and `npm run test`.
_PREFIX_END = r"(?=$|[\s:\-])"


def _compile(prefix: str) -> Pattern[str]:
    words = prefix.split()
    return re.compile(
        "^" + r"\s+".join(re.escape(word) for word in words) + _PREFIX_END
    )


def _user_command_patterns(
    environ: Mapping[str, str],
) -> Dict[str, List[str]]:
    """Read extra prefixes from `TRULENS_COMMAND_KIND_PATTERNS`.

    The value is a JSON object mapping a kind to a list of prefixes, for
    example `{"test": ["nox -s tests"], "build": ["bazel run //:gen"]}`.
    Invalid JSON and unknown kinds are ignored: telemetry must never break the
    agent it observes.
    """

    raw = environ.get(COMMAND_KIND_PATTERNS_ENV, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(parsed, Mapping):
        return {}
    result: Dict[str, List[str]] = {}
    for kind, prefixes in parsed.items():
        if kind not in _PRECEDENCE:
            continue
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if not isinstance(prefixes, list):
            continue
        result[kind] = [
            prefix.strip()
            for prefix in prefixes
            if isinstance(prefix, str) and prefix.strip()
        ]
    return result


def _command_patterns(
    environ: Mapping[str, str],
) -> List[List[Tuple[int, str, Pattern[str]]]]:
    """Return user prefixes, then default prefixes, longest first.

    A simple command takes the kind of its longest matching prefix, so
    `make lint` is lint even though `make` alone is build. User prefixes are
    tried before any default, so they can reclassify a command outright.
    """

    extra = _user_command_patterns(environ)
    tiers = []
    for source in (extra, _DEFAULT_COMMAND_KIND_PATTERNS):
        tier = [
            (len(prefix.split()), kind, _compile(prefix))
            for kind in _PRECEDENCE
            for prefix in source.get(kind, ())
        ]
        tier.sort(key=lambda item: -item[0])
        tiers.append(tier)
    return tiers


def _kind_of(simple: str, tiers) -> Optional[str]:
    for tier in tiers:
        for _, kind, pattern in tier:
            if pattern.match(simple):
                return kind
    return None


def _strip_quotes(command: str) -> str:
    """Blank out quoted text so `git commit -m "fix pytest"` is not a test."""

    return re.sub(r"'[^']*'|\"(?:\\.|[^\"\\])*\"", "''", command)


def _tokens(segment: str) -> List[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _unwrap(tokens: List[str]) -> List[str]:
    changed = True
    while tokens and changed:
        changed = False
        while tokens and _ENV_ASSIGNMENT.match(tokens[0]):
            tokens = tokens[1:]
            changed = True
        for wrapper in _WRAPPERS:
            if tuple(tokens[: len(wrapper)]) == wrapper:
                tokens = tokens[len(wrapper) :]
                changed = True
                break
    return tokens


def _simple_commands(command: str) -> Iterable[str]:
    for segment in _SEPARATORS.split(_strip_quotes(command)):
        tokens = _unwrap(_tokens(segment.strip()))
        if tokens:
            yield " ".join(tokens)


def classify_command(
    command: Optional[str], environ: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Label a shell command as test, build, lint, vcs or other.

    Args:
        command: The command line the agent ran.
        environ: Environment to read `TRULENS_COMMAND_KIND_PATTERNS` from.
            Defaults to `os.environ`.

    Returns:
        One of `COMMAND_KINDS`, or None when there is no command to classify.
    """

    if not isinstance(command, str) or not command.strip():
        return None
    tiers = _command_patterns(os.environ if environ is None else environ)
    found = {_kind_of(simple, tiers) for simple in _simple_commands(command)}
    return next((kind for kind in _PRECEDENCE if kind in found), "other")


def _glob_regex(glob: str) -> Pattern[str]:
    parts = []
    index = 0
    while index < len(glob):
        if glob.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif glob.startswith("**", index):
            parts.append(".*")
            index += 2
        elif glob[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif glob[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(glob[index]))
            index += 1
    return re.compile("".join(parts))


def configured_test_globs(
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[str, ...]:
    """Return the globs that mark a path as a test file.

    `TRULENS_TEST_FILE_GLOBS`, a comma-separated list, replaces the defaults
    when set.
    """

    source = os.environ if environ is None else environ
    raw = source.get(TEST_FILE_GLOBS_ENV, "")
    custom = tuple(glob.strip() for glob in raw.split(",") if glob.strip())
    return custom or DEFAULT_TEST_FILE_GLOBS


def is_test_file(
    path: Optional[str], environ: Optional[Mapping[str, str]] = None
) -> Optional[bool]:
    """Report whether an edited path is a test file.

    A glob without a slash is matched against the file name. A glob with one is
    matched against the path and every trailing part of it, so `tests/**`
    matches both `tests/test_x.py` and `/repo/pkg/tests/helpers.py`.

    Args:
        path: The edited file's path, absolute or relative.
        environ: Environment to read `TRULENS_TEST_FILE_GLOBS` from. Defaults
            to `os.environ`.

    Returns:
        True or False, or None when there is no path to classify.
    """

    if not isinstance(path, str) or not path.strip():
        return None
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    name = normalized.rsplit("/", 1)[-1]
    segments = normalized.split("/")
    suffixes = ["/".join(segments[index:]) for index in range(len(segments))]
    for glob in configured_test_globs(environ):
        if glob.startswith("./"):
            glob = glob[2:]
        pattern = _glob_regex(glob)
        if "/" not in glob:
            if pattern.fullmatch(name):
                return True
        elif any(pattern.fullmatch(suffix) for suffix in suffixes):
            return True
    return False

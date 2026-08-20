"""Static checks on the version constraints trulens publishes.

These tests read the `pyproject.toml` files under `src/` and check the
*declared* constraints, not the resolved environment. That distinction is the
point: CI installs from the committed `poetry.lock`, so any resolution the lock
happens to pick will satisfy the declared ranges by construction. A constraint
whose upper bound conflicts with the wider ecosystem is invisible to the lock
and to every runtime test.

The bug that motivated this: `dill = "^0.3.8"` expands to `>=0.3.8,<0.4.0`
because Poetry's caret pins the *minor* on `0.x` versions. That ceiling made
`trulens-core` uninstallable alongside `multiprocess>=0.70.19`, which requires
`dill>=0.4.1` and arrives transitively via HuggingFace `datasets` and `pathos`.
Nothing in the repository was wrong at runtime; only the published metadata
was, so only a metadata test can catch it.
"""

from collections.abc import Iterator
from pathlib import Path
import re
from unittest import TestCase

from packaging.version import InvalidVersion
from packaging.version import Version

try:  # Python 3.11+
    import tomllib
except ImportError:  # Python 3.9, 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:  # poetry vendors tomlkit, so it is normally present
        import tomlkit as tomllib  # type: ignore[no-redef]

# Repository root, from tests/unit/static/test_dependency_constraints.py.
REPO_ROOT = Path(__file__).parents[3]
SRC_ROOT = REPO_ROOT / "src"

# Dependencies allowed to declare a caret constraint on a `0.x` version, i.e.
# to deliberately pin the minor. Add an entry only with a comment explaining
# the known incompatibility that justifies the ceiling, and prefer an explicit
# `>=a.b,<c.d` range so the intent is legible in the published metadata.
ZERO_VERSION_CARET_ALLOWLIST: dict[tuple[str, str], str] = {}

# Dependencies allowed to declare an upper bound that does not fall on a major
# version boundary, keyed by `(pyproject path relative to src/, dep name)`.
#
# This is a ratchet, not a prohibition: narrow ceilings are sometimes correct,
# but each one is a future resolution conflict for downstream users, so it
# should be a deliberate decision with a recorded reason rather than a default.
# The entries below are the ceilings that predate this test. Adding to this list
# is fine; doing so silently is what the test prevents.
NON_MAJOR_CEILING_ALLOWLIST: dict[tuple[str, str], str] = {
    (
        "core/pyproject.toml",
        "dill",
    ): "dill is pre-1.0, so its minor is the compatibility boundary; 0.5 is "
    "unreleased and cannot be vouched for. Verified against 0.4.x.",
    (
        "apps/nemo/pyproject.toml",
        "langchain",
    ): "nemoguardrails constrains the langchain 0.x line it supports.",
    (
        "apps/nemo/pyproject.toml",
        "onnxruntime",
    ): "pinned for nemoguardrails wheel availability.",
    (
        "dashboard/pyproject.toml",
        "streamlit-aggrid",
    ): "grid component API is unstable across patch releases.",
}

# `python` is exempt: an interpreter ceiling like `<3.14` is mandatory, not a
# smell, since wheels genuinely do not exist for unreleased interpreters.
_CEILING_EXEMPT_NAMES = frozenset({"python"})

# Constraints that are not version ranges at all: path/git/url dependencies for
# the in-repo packages, which say nothing about the published metadata.
_NON_VERSION_KEYS = frozenset({"path", "git", "url"})


def _iter_declared_dependencies() -> Iterator[tuple[Path, str, str, str]]:
    """Yield `(pyproject_path, group, dependency_name, constraint)` tuples.

    Covers the main dependency table and every dependency group, since an
    optional-group constraint is published in the wheel's extras metadata and
    conflicts just as hard as a main one.
    """
    for path in sorted(SRC_ROOT.glob("**/pyproject.toml")):
        with path.open("rb") as fh:
            parsed = tomllib.load(fh)

        poetry = parsed.get("tool", {}).get("poetry", {})

        tables = {"main": poetry.get("dependencies", {})}
        for group_name, group in poetry.get("group", {}).items():
            tables[group_name] = group.get("dependencies", {})

        for group_name, deps in tables.items():
            for name, spec in deps.items():
                if isinstance(spec, str):
                    constraint = spec
                elif isinstance(spec, dict):
                    if _NON_VERSION_KEYS.intersection(spec):
                        continue
                    constraint = spec.get("version", "")
                elif isinstance(spec, list):
                    # Multiple constraints (markers); check each.
                    for alternative in spec:
                        if isinstance(alternative, dict) and not (
                            _NON_VERSION_KEYS.intersection(alternative)
                        ):
                            yield (
                                path,
                                group_name,
                                name,
                                alternative.get("version", ""),
                            )
                    continue
                else:
                    continue

                if constraint:
                    yield path, group_name, name, constraint


class TestDependencyConstraints(TestCase):
    """Checks that published constraints do not carry needless ceilings."""

    def test_finds_dependencies_to_check(self):
        """Guard against the traversal silently matching nothing.

        Without this, a refactor that moves the packages or renames the
        dependency tables would turn every other test in this class into a
        vacuous pass.
        """
        declared = list(_iter_declared_dependencies())

        self.assertGreater(
            len(declared),
            20,
            "Found suspiciously few declared dependencies under src/ -- the "
            "pyproject traversal in this test is probably broken, which would "
            "make the constraint checks below pass vacuously.",
        )

        packages = {path for path, _, _, _ in declared}
        self.assertGreater(
            len(packages),
            5,
            f"Only found pyprojects: {sorted(str(p) for p in packages)}",
        )

    def test_no_caret_on_zero_versions(self):
        """`^0.x.y` pins the minor, which is almost never intended.

        Poetry expands `^0.3.8` to `>=0.3.8,<0.4.0`, not `<1.0.0`. For a
        project that publishes libraries this is a trap: the next `0.(x+1)`
        release of the dependency becomes uninstallable alongside trulens, and
        anything that requires it is locked out.
        """
        violations = []

        for path, group, name, constraint in _iter_declared_dependencies():
            if not re.match(r"^\^\s*0\.", constraint):
                continue
            if (group, name) in ZERO_VERSION_CARET_ALLOWLIST:
                continue

            rel = path.relative_to(REPO_ROOT)
            violations.append(
                f'  {rel} [{group}]: {name} = "{constraint}" '
                f"-> resolves to a <0.(minor+1) ceiling"
            )

        self.assertEqual(
            [],
            violations,
            "Caret constraint on a 0.x dependency pins the minor version:\n"
            + "\n".join(violations)
            + "\n\nPoetry expands `^0.3.8` to `>=0.3.8,<0.4.0`, so the next "
            "0.x release of the dependency cannot be installed alongside "
            "trulens. Use an explicit range such as `>=0.3.8,<0.5` (widened "
            "only as far as the API you actually use allows), or add an entry "
            "to ZERO_VERSION_CARET_ALLOWLIST with the reason.",
        )

    def test_upper_bounds_fall_on_major_boundaries(self):
        """A ceiling below the next major is a resolution conflict waiting.

        Catches the same class of problem as the caret check but for bounds
        written out longhand -- `<0.4.0`, `<2.5`, `<=1.4.2` -- which the caret
        pattern above would miss.
        """
        violations = []

        for path, group, name, constraint in _iter_declared_dependencies():
            if name in _CEILING_EXEMPT_NAMES:
                continue

            key = (str(path.relative_to(SRC_ROOT)), name)
            if key in NON_MAJOR_CEILING_ALLOWLIST:
                continue

            for operator, raw in re.findall(
                r"(<=|<)\s*([0-9][0-9A-Za-z.\-+!]*)", constraint
            ):
                try:
                    ceiling = Version(raw)
                except InvalidVersion:
                    # Not a PEP 440 version; nothing to reason about.
                    continue

                # `<N.0.0` (equivalently `<N.0`, `<N`) is the healthy shape: it
                # excludes only the next major. `<=` never has that shape.
                is_major_boundary = operator == "<" and (
                    ceiling.minor == 0
                    and ceiling.micro == 0
                    and ceiling.pre is None
                    and ceiling.post is None
                    and ceiling.dev is None
                )
                if is_major_boundary:
                    continue

                rel = path.relative_to(REPO_ROOT)
                violations.append(
                    f'  {rel} [{group}]: {name} = "{constraint}" '
                    f"-> ceiling `{operator}{raw}` is below the next major"
                )

        self.assertEqual(
            [],
            violations,
            "Upper bound does not fall on a major version boundary:\n"
            + "\n".join(violations)
            + "\n\nA ceiling like `<0.4.0` or `<2.5` makes trulens "
            "uninstallable alongside anything that requires a newer release, "
            "even when trulens is compatible with it. Either widen to the "
            "next major or add an entry to NON_MAJOR_CEILING_ALLOWLIST "
            "documenting the known incompatibility.",
        )

    def test_dill_permits_multiprocess_floor(self):
        """Regression test for the constraint that caused the reported break.

        `multiprocess>=0.70.19` requires `dill>=0.4.1`, and `multiprocess`
        reaches most users transitively through HuggingFace `datasets`. The
        generic checks above cover the pattern; this pins the specific version
        so a future narrowing of the dill bound fails loudly with the reason
        attached.
        """
        from packaging.requirements import Requirement

        core_pyproject = SRC_ROOT / "core" / "pyproject.toml"
        with core_pyproject.open("rb") as fh:
            deps = tomllib.load(fh)["tool"]["poetry"]["dependencies"]

        constraint = deps["dill"]
        if isinstance(constraint, dict):
            constraint = constraint["version"]

        self.assertFalse(
            constraint.startswith("^"),
            f"dill constraint is `{constraint}`; a caret here pins the minor.",
        )

        requirement = Requirement(f"dill{constraint}")
        self.assertIn(
            Version("0.4.1"),
            list(requirement.specifier.filter([Version("0.4.1")])),
            f"trulens-core declares `dill{constraint}`, which excludes dill "
            "0.4.1. That makes trulens-core uninstallable alongside "
            "multiprocess>=0.70.19 (pulled in by HuggingFace datasets and "
            "pathos). trulens only calls dill.dumps(recurse=True) and "
            "dill.loads, both unchanged in 0.4.x.",
        )


class TestDillUsage(TestCase):
    """Functional counterpart to the metadata checks above.

    Widening a constraint is only safe if the code still works against the
    newer release. This exercises the entire dill surface trulens depends on,
    so the same suite that permits dill 0.4.x also proves it runs.
    """

    def test_dill_round_trip_of_app_loader(self):
        """Mirror the `initial_app_loader` calls in core/schema/app.py."""
        import dill

        def initial_app_loader():
            return {"app": "loaded"}

        dump = dill.dumps(initial_app_loader, recurse=True)
        self.assertIsInstance(dump, bytes)
        self.assertEqual({"app": "loaded"}, dill.loads(dump)())

    def test_dill_surface_used_by_trulens_is_unchanged(self):
        """Only `dumps` and `loads` may be relied on.

        If trulens starts using more of dill, this test should be updated
        deliberately -- and the constraint above re-examined, since a wider
        surface has a higher chance of breaking across a minor release.
        """
        import trulens.core.schema.app as app_schema

        source = Path(app_schema.__file__).read_text()
        used = set(re.findall(r"\bdill\.([A-Za-z_]+)", source))

        self.assertEqual(
            {"dumps", "loads"},
            used,
            f"core/schema/app.py now uses dill.{sorted(used)}. Confirm the "
            "new calls behave identically across the declared dill range "
            "before widening or keeping the bound in src/core/pyproject.toml.",
        )

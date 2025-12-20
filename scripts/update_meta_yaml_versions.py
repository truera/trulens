#!/usr/bin/env python3
"""
Script to update meta.yaml files with new versions and SHA256 hashes from PyPI.

Usage:
    python scripts/update_meta_yaml_versions.py --version <new_version>

Example:
    python scripts/update_meta_yaml_versions.py --version 2.5.3
"""

import argparse
from pathlib import Path
import re
import sys

import requests

# Mapping from package name to meta.yaml path (relative to repo root).
PACKAGE_TO_META_YAML: dict[str, str] = {
    "trulens-connectors-snowflake": "src/connectors/snowflake/meta.yaml",
    "trulens-core": "src/core/meta.yaml",
    "trulens-dashboard": "src/dashboard/meta.yaml",
    "trulens-feedback": "src/feedback/meta.yaml",
    "trulens-otel-semconv": "src/otel/semconv/meta.yaml",
    "trulens-providers-cortex": "src/providers/cortex/meta.yaml",
}


def get_repo_root() -> Path:
    """Get the repository root directory."""
    # This script is in scripts/, so repo root is one level up.
    return Path(__file__).parent.parent


def get_sha256_from_pypi(package_name: str, version: str) -> str:
    """
    Fetch the SHA256 hash for a package version from PyPI.

    Args:
        package_name: The name of the package on PyPI
        version: The version to fetch

    Returns:
        The SHA256 hash of the source distribution

    Raises:
        ValueError: If the version is not found or no source distribution exists
    """
    url = f"https://pypi.org/pypi/{package_name}/{version}/json"
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        raise ValueError(
            f"Version {version} of package '{package_name}' not found on PyPI. "
            f"Make sure the package has been released before running this script."
        )
    response.raise_for_status()
    data = response.json()
    # Look for the source distribution (.tar.gz).
    for file_info in data.get("urls", []):
        if file_info.get("packagetype") == "sdist":
            sha256 = file_info.get("digests", {}).get("sha256")
            if sha256:
                return sha256
    raise ValueError(
        f"No source distribution (sdist) found for {package_name} version {version}"
    )


def update_meta_yaml(
    file_path: Path, new_version: str, new_sha256: str
) -> None:
    """
    Update the version and SHA256 in a meta.yaml file.

    Args:
        file_path: Path to the meta.yaml file
        new_version: The new version string
        new_sha256: The new SHA256 hash
    """
    content = file_path.read_text()
    # Update version: {% set version = "X.X.X" %}.
    version_pattern = r'(\{%\s*set\s+version\s*=\s*")[^"]*(".*%\})'
    new_content = re.sub(version_pattern, rf"\g<1>{new_version}\g<2>", content)
    # Update sha256: sha256: <hash>.
    sha256_pattern = r"(sha256:\s*)[a-fA-F0-9]{64}"
    new_content = re.sub(sha256_pattern, rf"\g<1>{new_sha256}", new_content)
    file_path.write_text(new_content)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update meta.yaml files with new versions and SHA256 hashes from PyPI."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="The new version to set (e.g., 2.5.3)",
    )
    args = parser.parse_args()
    new_version: str = args.version
    repo_root = get_repo_root()
    errors: list[str] = []
    updates: list[
        tuple[str, str, str]
    ] = []  # (package_name, meta_yaml_path, sha256)
    print(f"Fetching SHA256 hashes for version {new_version} from PyPI...")
    print()
    # First, fetch all SHA256 hashes to ensure all packages are released.
    for package_name, meta_yaml_rel_path in PACKAGE_TO_META_YAML.items():
        meta_yaml_path = repo_root / meta_yaml_rel_path
        if not meta_yaml_path.exists():
            errors.append(f"meta.yaml not found: {meta_yaml_path}")
            continue
        try:
            sha256 = get_sha256_from_pypi(package_name, new_version)
            updates.append((package_name, meta_yaml_rel_path, sha256))
            print(f"  ✓ {package_name}: {sha256}")
        except ValueError as e:
            errors.append(str(e))
            print(f"  ✗ {package_name}: ERROR - {e}")
    print()
    if errors:
        print("Errors occurred:")
        for error in errors:
            print(f"  - {error}")
        print()
        print("Aborting. Please ensure all packages are released on PyPI.")
        return 1
    # Apply updates.
    print("Updating meta.yaml files...")
    for package_name, meta_yaml_rel_path, sha256 in updates:
        meta_yaml_path = repo_root / meta_yaml_rel_path
        update_meta_yaml(meta_yaml_path, new_version, sha256)
        print(f"  ✓ Updated {meta_yaml_rel_path}")
    print()
    print("Done! All meta.yaml files have been updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

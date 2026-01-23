#!/usr/bin/env python3
"""
Script to update meta.yaml files with version and SHA256 from PyPI.

This script:
1. Finds all meta.yaml files in the repository
2. Extracts the package name from each meta.yaml
3. Reads the version from the corresponding pyproject.toml
4. Fetches the SHA256 hash from PyPI for that package/version
5. Updates the meta.yaml with the new version and SHA256

Usage:
    python update_meta_yaml.py

    # Or via make:
    make update-meta-yaml
"""

from pathlib import Path
import re
import sys

import requests

# Repository root (same directory as this script)
REPO_ROOT = Path(__file__).parent


def find_meta_yaml_files() -> list[Path]:
    """Find all meta.yaml files in the repository."""
    return list(REPO_ROOT.glob("**/meta.yaml"))


def extract_package_name(meta_yaml_path: Path) -> str | None:
    """Extract package name from meta.yaml file.

    Looks for: {% set name = "package-name" %}
    """
    content = meta_yaml_path.read_text()
    match = re.search(r'\{%\s*set\s+name\s*=\s*"([^"]+)"\s*%\}', content)
    if match:
        return match.group(1)
    return None


def extract_version_from_pyproject(pyproject_path: Path) -> str | None:
    """Extract version from pyproject.toml file.

    Looks for: version = "X.Y.Z"
    """
    if not pyproject_path.exists():
        return None
    content = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def fetch_sha256_from_pypi(package_name: str, version: str) -> str | None:
    """Fetch SHA256 hash from PyPI for a specific package version.

    Uses the PyPI JSON API to get the sdist (tar.gz) SHA256.
    """
    url = f"https://pypi.org/pypi/{package_name}/{version}/json"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Find the sdist (source distribution) URL entry
        for url_info in data.get("urls", []):
            if url_info.get("packagetype") == "sdist":
                digests = url_info.get("digests", {})
                return digests.get("sha256")

        print(f"  Warning: No sdist found for {package_name}=={version}")
        return None

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(
                f"  Error: Package {package_name}=={version} not found on PyPI"
            )
            print(
                "         Make sure the version has been released to PyPI first."
            )
        else:
            print(f"  Error fetching from PyPI: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching from PyPI: {e}")
        return None


def update_meta_yaml(meta_yaml_path: Path, version: str, sha256: str) -> bool:
    """Update the meta.yaml file with new version and SHA256.

    Updates:
    - {% set version = "X.Y.Z" %}
    - sha256: <hash>
    """
    content = meta_yaml_path.read_text()
    original_content = content

    # Update version
    content = re.sub(
        r'(\{%\s*set\s+version\s*=\s*")[^"]+("\s*%\})',
        rf"\g<1>{version}\g<2>",
        content,
    )

    # Update sha256
    content = re.sub(r"(sha256:\s*)[a-fA-F0-9]+", rf"\g<1>{sha256}", content)

    if content != original_content:
        meta_yaml_path.write_text(content)
        return True
    return False


def main():
    """Main function to update all meta.yaml files."""
    print(
        "Updating meta.yaml files with versions and SHA256 hashes from PyPI...\n"
    )

    meta_yaml_files = find_meta_yaml_files()

    if not meta_yaml_files:
        print("No meta.yaml files found!")
        sys.exit(1)

    print(f"Found {len(meta_yaml_files)} meta.yaml files\n")

    success_count = 0
    error_count = 0
    skip_count = 0

    for meta_yaml_path in sorted(meta_yaml_files):
        relative_path = meta_yaml_path.relative_to(REPO_ROOT)
        print(f"Processing: {relative_path}")

        # Extract package name
        package_name = extract_package_name(meta_yaml_path)
        if not package_name:
            print(
                f"  Error: Could not extract package name from {relative_path}"
            )
            error_count += 1
            continue
        print(f"  Package: {package_name}")

        # Find corresponding pyproject.toml
        pyproject_path = meta_yaml_path.parent / "pyproject.toml"
        version = extract_version_from_pyproject(pyproject_path)
        if not version:
            print(
                f"  Error: Could not find version in {pyproject_path.relative_to(REPO_ROOT)}"
            )
            error_count += 1
            continue
        print(f"  Version: {version}")

        # Fetch SHA256 from PyPI
        sha256 = fetch_sha256_from_pypi(package_name, version)
        if not sha256:
            error_count += 1
            continue
        print(f"  SHA256: {sha256[:16]}...")

        # Update meta.yaml
        if update_meta_yaml(meta_yaml_path, version, sha256):
            print(f"  ✓ Updated {relative_path}")
            success_count += 1
        else:
            print("  - No changes needed (already up to date)")
            skip_count += 1

        print()

    # Summary
    print("-" * 50)
    print(
        f"Summary: {success_count} updated, {skip_count} unchanged, {error_count} errors"
    )

    if error_count > 0:
        print(
            "\nNote: Errors may occur if the version hasn't been released to PyPI yet."
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

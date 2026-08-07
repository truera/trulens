"""Fail if any LFS-tracked file in the working tree is still a pointer stub.

Every .png, .jpg and .gif in this repository is tracked by Git LFS. A clone whose
smudge filter is disabled -- a reasonable thing for a developer to do, since it
makes checkouts much faster -- leaves those files on disk as ~130 byte text stubs
instead of images.

Nothing downstream objects. The docs build copies them into site/ verbatim, a
deploy pushes them, and the server then answers requests for them with HTTP 200
and Content-Type: image/png, carrying 130 bytes of pointer text. A browser shows
a broken image and no log anywhere records an error. This is how all 55
LFS-tracked files under docs/ came to be broken on trulens.org at once while the
bytes committed to git were perfectly fine the whole time.

So the check runs against the working tree rather than the built site: it costs a
few milliseconds and can fail before anything is built or pushed. Patterns come
from .gitattributes rather than a list held here, so adding a new LFS-tracked
extension does not quietly fall outside the guard.

Usage:
    python tools/check_no_lfs_pointers.py [ROOT ...]     # default: docs
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent

# The first line of the pointer format, which is stable and documented. Matching
# on it is safer than matching on size alone, since a genuinely tiny image is
# possible and a pointer that grew a little should still be caught.
POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"

# A pointer is three short lines. Reading a little more than that is enough to
# identify one without pulling a multi-megabyte image into memory.
SNIFF = 200


def lfs_patterns(gitattributes: Path) -> list[str]:
    """Glob patterns that .gitattributes routes through the LFS filter."""
    patterns = []
    for line in gitattributes.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if "filter=lfs" in fields[1:]:
            patterns.append(fields[0])
    return patterns


def is_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(SNIFF).startswith(POINTER_MAGIC)
    except OSError:
        return False


def display(path: Path) -> str:
    """Shorten a path against the repo when it is inside it, else leave it alone.

    Roots outside the repository are a supported case: CI points this at an
    extracted copy of the gh-pages branch to check what was actually published.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or [REPO / "docs"]

    attributes = REPO / ".gitattributes"
    if not attributes.exists():
        print(f"error: {attributes} not found; cannot tell which files are LFS")
        return 1

    patterns = lfs_patterns(attributes)
    if not patterns:
        print(f"error: no filter=lfs patterns in {attributes}")
        return 1

    checked, pointers = 0, []
    for root in roots:
        if not root.exists():
            print(f"error: {root} does not exist")
            return 1
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if not any(fnmatch.fnmatch(path.name, p) for p in patterns):
                continue
            checked += 1
            if is_pointer(path):
                pointers.append(path)

    where = ", ".join(display(r) for r in roots)
    if pointers:
        print(
            f"{len(pointers)} of {checked} LFS-tracked files under {where} are pointer stubs:"
        )
        for path in pointers:
            print(f"  {display(path)}  ({path.stat().st_size} bytes)")
        print(
            "\nThese would deploy as HTTP 200 responses containing pointer text rather\n"
            "than images. Run `git lfs pull` to materialise the real content, then\n"
            "re-run this check. If the clone has smudge disabled, `git lfs pull` still\n"
            "writes the real bytes and leaves `git status` clean."
        )
        return 1

    print(
        f"ok: {checked} LFS-tracked files under {where}, none are pointer stubs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

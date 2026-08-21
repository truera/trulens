"""Check that a pull request certifies the Developer Certificate of Origin.

The pull request template carries a checkbox. Ticking it certifies the
[DCO](../../DCO) for every commit in the pull request: that you wrote the changes,
or have the right to submit them under this project's license.

The check re-runs whenever commits are pushed, so a passing status always
corresponds to the pull request's current head rather than to whatever it looked
like when the box was ticked.

Usage::

    python check_dco.py --body-file body.md
    python check_dco.py --body "$PR_BODY"
"""

from __future__ import annotations

import argparse
import re
import sys

TERM = "developer certificate of origin"

# A task-list item: "- [ ]", "- [x]", "* [X]", with optional leading whitespace.
ITEM = re.compile(r"^[ \t]*[-*][ \t]*\[(?P<mark>[ xX])\](?P<rest>.*)$")

MISSING = """No Developer Certificate of Origin checkbox found in the pull request
description.

The pull request template includes it. If the section was removed, add this line
back to the description:

  - [x] I certify that I wrote these changes, or have the right to submit them
        under this project's license (the Developer Certificate of Origin)
"""

UNTICKED = """The Developer Certificate of Origin checkbox is not ticked.

Edit the pull request description and change `- [ ]` to `- [x]` on that line. By
ticking it you certify that you wrote these changes, or have the right to submit
them under this project's license. The full text is in DCO at the repository root.
"""


def _items(body: str) -> list[tuple[str, str]]:
    """Task-list items as (mark, text), joining wrapped continuation lines.

    Markdown allows a list item to wrap across lines, and the template's item does.
    Anything after the checkbox that is not itself a new list item, a heading, or a
    blank line belongs to the same item.
    """
    items: list[tuple[str, str]] = []
    mark: str | None = None
    parts: list[str] = []

    def flush() -> None:
        if mark is not None:
            items.append((mark, " ".join(parts)))

    for line in (body or "").splitlines():
        match = ITEM.match(line)
        if match:
            flush()
            mark = match.group("mark")
            parts = [match.group("rest")]
            continue
        if mark is not None:
            if not line.strip() or line.lstrip().startswith("#"):
                flush()
                mark, parts = None, []
            else:
                parts.append(line.strip())
    flush()
    return items


def check(body: str) -> str | None:
    """Return an error message, or None when the DCO is certified."""
    relevant = [
        (mark, text) for mark, text in _items(body) if TERM in text.lower()
    ]
    if not relevant:
        return MISSING
    if all(mark == " " for mark, _ in relevant):
        return UNTICKED
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--body", help="pull request description")
    source.add_argument("--body-file", help="file containing the description")
    args = parser.parse_args()

    body = (
        args.body
        if args.body is not None
        else open(args.body_file, encoding="utf-8").read()
    )

    error = check(body)
    if error:
        print(error)
        return 1
    print("Developer Certificate of Origin certified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

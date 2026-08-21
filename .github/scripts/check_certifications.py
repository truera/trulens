"""Check that a pull request carries the required certifications.

The pull request template has a checkbox for each. Ticking them certifies, for
every commit in the pull request:

- the [Developer Certificate of Origin](../../DCO) -- that you wrote the changes,
  or have the right to submit them under this project's license;
- that the change follows the project's
  [standards](https://www.trulens.org/contributing/standards/).

The check re-runs whenever commits are pushed, so a passing status always
corresponds to the pull request's current head rather than to whatever it looked
like when the boxes were ticked.

Usage::

    python check_certifications.py --body-file body.md
    python check_certifications.py --body "$PR_BODY"
"""

from __future__ import annotations

import argparse
import re
import sys

# Each certification is matched by a phrase that must appear in the checkbox item.
# Matching on a phrase rather than exact text means the template can be reworded or
# reordered without breaking the check.
REQUIRED = {
    "Developer Certificate of Origin": (
        "- [x] I certify that I wrote these changes, or have the right to submit"
        " them under this project's license (the Developer Certificate of Origin)"
    ),
    "TruLens standards": (
        "- [x] This change follows the TruLens standards"
        " (https://www.trulens.org/contributing/standards/)"
    ),
}

# A task-list item: "- [ ]", "- [x]", "* [X]", with optional leading whitespace.
ITEM = re.compile(r"^[ \t]*[-*][ \t]*\[(?P<mark>[ xX])\](?P<rest>.*)$")


def _items(body: str) -> list[tuple[str, str]]:
    """Task-list items as (mark, text), joining wrapped continuation lines.

    Markdown allows a list item to wrap across lines, and the template's items do.
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


def check(body: str) -> list[str]:
    """Return a message for each certification that is missing or unticked."""
    items = _items(body)
    problems: list[str] = []

    for phrase, example in REQUIRED.items():
        matching = [
            mark for mark, text in items if phrase.lower() in text.lower()
        ]
        if not matching:
            problems.append(
                f'No checkbox mentioning "{phrase}" found in the pull request '
                f"description.\nThe template includes one. If the section was "
                f"removed, add this back:\n\n{example}\n"
            )
        elif all(mark == " " for mark in matching):
            problems.append(
                f'The "{phrase}" checkbox is not ticked. Edit the pull request '
                f"description and change `- [ ]` to `- [x]` on that line.\n"
            )
    return problems


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

    problems = check(body)
    if problems:
        for problem in problems:
            print(problem)
        return 1
    print(f"all {len(REQUIRED)} certifications present")
    return 0


if __name__ == "__main__":
    sys.exit(main())

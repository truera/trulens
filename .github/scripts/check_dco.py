"""Check that every commit in a pull request carries a DCO sign-off.

Each commit needs a trailer of the form::

    Signed-off-by: Your Name <your.email@example.com>

with an email matching the commit's author or committer. Adding it certifies the
[Developer Certificate of Origin](../../DCO) -- that you wrote the change, or
have the right to submit it under this project's license.

Only commits introduced by the pull request are checked, so history predating
this policy is unaffected.

Usage::

    python check_dco.py <base-sha> <head-sha>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

SIGNOFF = re.compile(r"^Signed-off-by: .+ <(?P<email>[^<>]+)>\s*$", re.M)

# Bots commit through the API and cannot add trailers.
BOT_SUFFIXES = ("[bot]",)
BOT_EMAILS = ("@users.noreply.github.com",)


def commits(base: str, head: str) -> list[str]:
    """SHAs introduced by head and not present in base, excluding merges."""
    out = subprocess.run(
        ["git", "rev-list", "--no-merges", f"{base}..{head}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.split() if line]


def field(sha: str, fmt: str) -> str:
    return subprocess.run(
        ["git", "show", "-s", f"--format={fmt}", sha],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def is_bot(name: str, email: str) -> bool:
    return name.endswith(BOT_SUFFIXES) and email.endswith(BOT_EMAILS)


def check(sha: str) -> str | None:
    """Return an error message, or None when the commit is signed off."""
    author_name = field(sha, "%an")
    author_email = field(sha, "%ae").lower()
    committer_email = field(sha, "%ce").lower()
    body = field(sha, "%B")

    if is_bot(author_name, author_email):
        return None

    signoffs = {m.group("email").lower() for m in SIGNOFF.finditer(body)}
    if not signoffs:
        return f"{sha[:8]} {field(sha, '%s')[:60]!r} has no Signed-off-by"
    if author_email not in signoffs and committer_email not in signoffs:
        return (
            f"{sha[:8]} signed off by {', '.join(sorted(signoffs))} but authored "
            f"by {author_email}"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="base SHA or ref")
    parser.add_argument("head", help="head SHA or ref")
    args = parser.parse_args()

    shas = commits(args.base, args.head)
    if not shas:
        print("no commits to check")
        return 0

    failures = [msg for sha in shas if (msg := check(sha))]
    if not failures:
        print(f"all {len(shas)} commit(s) signed off")
        return 0

    print(
        f"{len(failures)} of {len(shas)} commit(s) missing a valid sign-off:\n"
    )
    for msg in failures:
        print(f"  {msg}")
    print(
        "\nSign off future commits with `git commit -s`. To fix these, run:\n"
        f"\n    git rebase --signoff {args.base}\n    git push --force-with-lease\n"
        "\nSee CONTRIBUTING.md for details, and DCO for what you are certifying."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

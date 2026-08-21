"""Request advisory reviews from area reviewers listed in area-reviewers.yml.

GitHub's CODEOWNERS only routes to accounts with write access, so Triagers --
who hold the Triage role and no write access -- never get review requests from
it. This script covers that gap: it reads ``.github/area-reviewers.yml``, matches
the paths changed by a pull request, and requests a review from each listed
reviewer.

Requests here are advisory. A Maintainer or Area Reviewer still approves and
merges; see CONTRIBUTOR_LADDER.md.

Run with ``--dry-run FILE...`` to check path matching locally without touching
the GitHub API.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "area-reviewers.yml"
API = "https://api.github.com"
MARKER = "<!-- area-reviewers -->"


def load_areas(path: Path = CONFIG_PATH) -> list[dict[str, Any]]:
    """Read the area routing table."""
    config = yaml.safe_load(path.read_text()) or {}
    return config.get("areas") or []


def matches(pattern: str, changed_path: str) -> bool:
    """Whether ``changed_path`` falls under ``pattern``.

    Patterns are directory prefixes (trailing ``/``) or exact paths. There is
    deliberately no wildcard support -- this table routes review requests to
    people, so predictability matters more than expressiveness.
    """
    if pattern.endswith("/"):
        return changed_path.startswith(pattern)
    return changed_path == pattern


def match_areas(
    areas: list[dict[str, Any]], changed: list[str]
) -> dict[str, set[str]]:
    """Map area name to reviewers, for areas touched by ``changed``.

    A file counts toward an area if it matches one of the area's ``paths`` and
    none of its ``exclude`` patterns. Exclusions let a broad area hand a subtree
    to a different owner -- for example, the providers area excludes the Google
    provider, which has its own reviewer.
    """
    matched: dict[str, set[str]] = {}
    for entry in areas:
        reviewers = {r for r in (entry.get("reviewers") or []) if r}
        if not reviewers:
            continue
        patterns = entry.get("paths") or []
        excluded = entry.get("exclude") or []
        hit = any(
            any(matches(p, c) for p in patterns)
            and not any(matches(x, c) for x in excluded)
            for c in changed
        )
        if hit:
            matched[entry["name"]] = reviewers
    return matched


def _request(
    method: str, url: str, token: str, payload: dict | None = None
) -> Any:
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    if body:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def changed_files(repo: str, pr: int, token: str) -> list[str]:
    """All paths changed by the pull request, following pagination."""
    paths: list[str] = []
    page = 1
    while True:
        batch = _request(
            "GET",
            f"{API}/repos/{repo}/pulls/{pr}/files?per_page=100&page={page}",
            token,
        )
        if not batch:
            break
        paths.extend(f["filename"] for f in batch)
        if len(batch) < 100:
            break
        page += 1
    return paths


def existing_reviewers(repo: str, pr: int, token: str) -> set[str]:
    """Reviewers already requested or already reviewed.

    Avoids re-pinging someone who has been asked, or who has already left a
    review and had their request cleared.
    """
    requested = _request(
        "GET", f"{API}/repos/{repo}/pulls/{pr}/requested_reviewers", token
    )
    users = {u["login"] for u in (requested or {}).get("users", [])}
    reviews = (
        _request(
            "GET", f"{API}/repos/{repo}/pulls/{pr}/reviews?per_page=100", token
        )
        or []
    )
    users |= {r["user"]["login"] for r in reviews if r.get("user")}
    return users


def request_reviews(
    repo: str, pr: int, token: str, reviewers: set[str]
) -> tuple[list[str], list[str]]:
    """Request review from each user, one call each.

    Deliberately not batched: GitHub rejects the whole batch if any single
    reviewer is invalid, and a contributor who has since lost access should not
    block requests to everyone else.
    """
    ok: list[str] = []
    failed: list[str] = []
    for login in sorted(reviewers):
        try:
            _request(
                "POST",
                f"{API}/repos/{repo}/pulls/{pr}/requested_reviewers",
                token,
                {"reviewers": [login]},
            )
            ok.append(login)
        except urllib.error.HTTPError as exc:
            print(f"could not request {login}: {exc.code} {exc.reason}")
            failed.append(login)
    return ok, failed


def comment_body(matched: dict[str, set[str]], failed: list[str]) -> str:
    lines = [
        MARKER,
        "This pull request touches areas with designated reviewers:",
        "",
    ]
    for name, reviewers in sorted(matched.items()):
        mentions = ", ".join(f"@{r}" for r in sorted(reviewers))
        lines.append(f"- **{name}** — {mentions}")
    lines += [
        "",
        "These reviews are advisory and do not block merging. A maintainer or "
        "area reviewer still approves. See "
        "[CONTRIBUTOR_LADDER.md](../blob/main/CONTRIBUTOR_LADDER.md).",
    ]
    if failed:
        names = ", ".join(f"@{f}" for f in failed)
        lines += [
            "",
            f"Could not send a review request to {names} — tagging here "
            "instead. This usually means the account is not a repository "
            "collaborator yet.",
        ]
    return "\n".join(lines)


def upsert_comment(repo: str, pr: int, token: str, body: str) -> None:
    """Post the routing comment, replacing any previous one."""
    comments = (
        _request(
            "GET",
            f"{API}/repos/{repo}/issues/{pr}/comments?per_page=100",
            token,
        )
        or []
    )
    for comment in comments:
        if MARKER in (comment.get("body") or ""):
            _request(
                "PATCH",
                f"{API}/repos/{repo}/issues/comments/{comment['id']}",
                token,
                {"body": body},
            )
            return
    _request(
        "POST",
        f"{API}/repos/{repo}/issues/{pr}/comments",
        token,
        {"body": body},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        nargs="+",
        metavar="PATH",
        help="match these paths and print the result, without calling the API",
    )
    args = parser.parse_args()

    areas = load_areas()

    if args.dry_run:
        matched = match_areas(areas, args.dry_run)
        if not matched:
            print("no areas matched")
            return 0
        for name, reviewers in sorted(matched.items()):
            print(f"{name}: {', '.join(sorted(reviewers))}")
        return 0

    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["REPO"]
    pr = int(os.environ["PR_NUMBER"])
    author = os.environ.get("PR_AUTHOR", "")

    changed = changed_files(repo, pr, token)
    matched = match_areas(areas, changed)
    if not matched:
        print("no areas matched; nothing to do")
        return 0

    reviewers = set().union(*matched.values())
    reviewers.discard(author)
    reviewers -= existing_reviewers(repo, pr, token)
    if not reviewers:
        print("all area reviewers already requested or have reviewed")
        return 0

    print(f"requesting: {', '.join(sorted(reviewers))}")
    _, failed = request_reviews(repo, pr, token, reviewers)
    upsert_comment(repo, pr, token, comment_body(matched, failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())

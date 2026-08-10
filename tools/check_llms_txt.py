"""Validate docs/llms.txt against the llmstxt.org spec and the built sitemap.

The file is hand-authored, which buys editorial control at the cost of drift: a
page can be renamed or removed and nothing today would notice, because lychee
only inspects site/**/*.html and the strict build does not look inside static
files.

Every internal link in llms.txt must therefore appear in the built sitemap.xml.
That is a plain subset assertion, needs no network access, and catches renames,
moves and deletions. External links are counted and skipped.

Usage:
    python tools/check_llms_txt.py [--llms docs/llms.txt] [--sitemap site/sitemap.xml]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from xml.etree import ElementTree

INTERNAL_HOSTS = ("www.trulens.org", "trulens.org")
LINK = re.compile(r"\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)")
SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def parse_llms_txt(text: str) -> tuple[list[str], list[str]]:
    """Return (urls, structural errors) after checking the required spec shape."""
    errors: list[str] = []
    lines = text.splitlines()

    h1s = [i for i, line in enumerate(lines) if line.startswith("# ")]
    if len(h1s) != 1:
        errors.append(f"expected exactly one H1, found {len(h1s)}")
    else:
        preceding = [line for line in lines[: h1s[0]] if line.strip()]
        if preceding:
            errors.append("content appears before the H1")

    body = [line for line in lines if line.strip()]
    if len(body) < 2:
        errors.append("file is too short to contain an H1 and a summary")
    elif not body[1].startswith(">"):
        errors.append("the H1 must be followed by a blockquote summary")

    # Inside an H2 file list, every bullet has to be a real markdown link, or a
    # consumer parsing the file with a regex silently drops the entry.
    in_section = False
    for n, line in enumerate(lines, start=1):
        if line.startswith("## "):
            in_section = True
            continue
        if line.startswith("# "):
            in_section = False
            continue
        if in_section and line.startswith("- ") and not LINK.search(line):
            errors.append(
                f"line {n}: list item is not a markdown link: {line.strip()}"
            )

    urls = [m.group("url") for m in LINK.finditer(text)]
    return urls, errors


def sitemap_urls(path: Path) -> set[str]:
    root = ElementTree.parse(path).getroot()
    return {
        loc.text.strip()
        for loc in root.iter(f"{SITEMAP_NS}loc")
        if loc.text and loc.text.strip()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llms", default="docs/llms.txt", type=Path)
    ap.add_argument("--sitemap", default="site/sitemap.xml", type=Path)
    args = ap.parse_args()

    if not args.llms.exists():
        print(f"FAIL: {args.llms} does not exist")
        return 1
    if not args.sitemap.exists():
        print(
            f"FAIL: {args.sitemap} does not exist. Build the docs first, for example "
            "with `make docs`."
        )
        return 1

    urls, errors = parse_llms_txt(args.llms.read_text(encoding="utf-8"))
    known = sitemap_urls(args.sitemap)
    if not known:
        print(
            f"FAIL: {args.sitemap} lists no URLs. site_url is probably unset in "
            "mkdocs.yml, which makes this check vacuous."
        )
        return 1

    internal = [u for u in urls if any(h in u for h in INTERNAL_HOSTS)]
    external = [u for u in urls if u not in internal]
    missing = [u for u in internal if u not in known]

    print(
        f"{args.llms}: {len(urls)} links ({len(internal)} internal, {len(external)} external)"
    )
    print(f"{args.sitemap}: {len(known)} URLs")

    for e in errors:
        print(f"FAIL: spec: {e}")
    for u in missing:
        print(f"FAIL: not in sitemap: {u}")

    if errors or missing:
        print(
            f"\n{len(errors) + len(missing)} problem(s). External links are not checked here."
        )
        return 1

    print("OK: every internal link resolves and the spec shape is intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

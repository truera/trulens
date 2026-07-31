"""Render the og:image / README banner card at the OG spec size of 1200x630.

The card has text baked into pixels, which means it drifts the moment the tagline
changes and nobody can fix it without redoing the artwork. Keeping the generator
in the repo makes that a one-command job instead.

Built from the home page's own tokens and the same squid-eye mark that closes
band 15, so the card and the page stay one thing.

Usage:
    python tools/make_og_card.py [--out docs/assets/images/og-card.png]
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

WORDMARK = "trulens"
TAGLINE = "Evals and tracing for AI agents"
META = "Open source &middot; OpenTelemetry-native &middot; MIT"

INK = "#0A2C37"
AQUA = "#84DCD7"
SAND = "#F6D881"
MIST = "#E9F2F1"

# The band 15 mark, with the CSS custom properties resolved.
EYE = f"""
<svg viewBox="0 0 320 320" style="width:100%;height:100%">
  <circle cx="160" cy="160" r="150" fill="rgba(255,255,255,.06)"/>
  <circle cx="160" cy="160" r="112" fill="{AQUA}"/>
  <circle cx="160" cy="160" r="84" fill="none" stroke="{INK}" stroke-width="3" opacity=".3"/>
  <circle cx="160" cy="160" r="52" fill="#06202A"/>
  <circle cx="141" cy="141" r="15" fill="#fff" opacity=".92"/>
  <circle cx="268" cy="258" r="28" fill="{SAND}"/>
  <circle cx="268" cy="258" r="11" fill="#fff" opacity=".7"/>
</svg>
"""

# Type is sized for the thumbnail case: unfurls are often rendered at a third of
# 1200x630, where a 112px wordmark survives and body copy does not.
HTML = f"""
<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700
&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  html,body{{width:1200px;height:630px;overflow:hidden}}
  body{{background:{INK};display:flex;align-items:center;justify-content:center;
    position:relative;font-family:'Space Grotesk',sans-serif}}
  .eye{{position:absolute;right:-90px;bottom:-120px;width:520px;height:520px;opacity:.16}}
  .wordmark{{font-size:112px;font-weight:700;letter-spacing:-.02em;color:#fff;line-height:1}}
  .tagline{{font-size:38px;font-weight:500;color:{AQUA};margin-top:22px}}
  .meta{{font-family:'JetBrains Mono',monospace;font-size:16px;text-transform:uppercase;
    letter-spacing:.14em;color:{MIST};margin-top:34px;opacity:.72}}
</style></head>
<body>
  <div class="eye">{EYE}</div>
  <div style="text-align:center;z-index:1;padding:0 72px">
    <div class="wordmark">{WORDMARK}</div>
    <p class="tagline">{TAGLINE}</p>
    <div class="meta">{META}</div>
  </div>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default="docs/assets/images/og-card.png", type=Path
    )
    args = ap.parse_args()

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(
            viewport={"width": 1200, "height": 630}, device_scale_factor=1
        )
        pg.set_content(HTML)
        # Webfonts must land before capture or the card renders in a fallback face.
        pg.wait_for_timeout(2500)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pg.screenshot(path=str(args.out))
        b.close()

    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size / 1024:.0f} KB, 1200x630)")


if __name__ == "__main__":
    main()

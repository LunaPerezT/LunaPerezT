#!/usr/bin/env python3
"""
Generate the animated terminal banner for the LunaPerezT profile README.

Writes assets/banner-dark.svg and assets/banner-light.svg.

The banner is a self-contained SVG: no external fonts, no external requests,
no JavaScript. The typing effect is SMIL (<animate> on a clip rect's width),
which is what renders reliably when an SVG is loaded through an <img> tag --
GitHub serves README images that way, and scripts never run in that context.

Edit LINES below and re-run:  python scripts/generate_banner.py
"""
from pathlib import Path
from xml.sax.saxutils import escape

# ---------------------------------------------------------------- content ---
# Each entry: (kind, [(role, text), ...])
#   kind  'cmd' -> prefixed with a green $ prompt
#         'out' -> indented output line
#   role  key   -> highlighted (accent)
#         num   -> numbers / metrics (second accent)
#         dim   -> secondary text
#         txt   -> normal text
LINES = [
    ("cmd", [("txt", "whoami")]),
    ("out", [("key", "Luna Pérez Troncoso"), ("dim", " · "),
             ("txt", "Data Scientist & AI Engineer"), ("dim", " · Madrid")]),
    ("cmd", [("txt", "cat focus.txt")]),
    ("out", [("txt", "Medical & health AI"), ("dim", " — from Spark pipelines "),
             ("dim", "to deployed decision tools")]),
    ("cmd", [("txt", "ls ~/portfolio | wc -l")]),
    ("out", [("num", "9"), ("dim", " projects · "), ("num", "3"), ("dim", " live apps · "),
             ("num", "109"), ("dim", " passing tests")]),
    ("cmd", [("txt", "python -c \"print(headline)\"")]),
    ("out", [("dim", "ECG macro-F1 "), ("num", "0.9166"), ("dim", "  ·  MRI Dice "),
             ("num", "0.846"), ("dim", "  ·  WRMSSE₇ "), ("num", "0.7717")]),
]

TITLE = "profile.sh --live"

# ----------------------------------------------------------------- themes ---
THEMES = {
    "dark": dict(
        bg="#0D1117", chrome="#161B22", border="#30363D", shadow="#010409",
        title="#8B949E", prompt="#3FB950", txt="#E6EDF3", dim="#8B949E",
        key="#58A6FF", num="#BC8CFF", cursor="#58A6FF",
        dot1="#FF5F57", dot2="#FEBC2E", dot3="#28C840",
    ),
    "light": dict(
        bg="#FFFFFF", chrome="#F6F8FA", border="#D1D9E0", shadow="#D1D9E0",
        title="#59636E", prompt="#1A7F37", txt="#1F2328", dim="#59636E",
        key="#0550AE", num="#6639BA", cursor="#0550AE",
        dot1="#FF5F57", dot2="#FEBC2E", dot3="#28C840",
    ),
}

# ---------------------------------------------------------------- metrics ---
W, BAR = 980, 42            # canvas width, title-bar height
PAD_X, LINE_H = 34, 30      # left padding, line height
FS = 17                     # font size
CH = FS * 0.6               # monospace advance width
GAP_AFTER_OUT = 10          # extra breathing room under each output line
# A command is "typed" at human speed; its output appears almost at once, the
# way a real shell behaves. Whole sequence lands under 5s -- a header nobody
# should have to wait for.
TYPE_CMD = 0.035            # seconds per character while typing a command
TYPE_OUT = 0.006            # seconds per character while printing output
PAUSE = 0.10                # pause between lines

FONT = ("ui-monospace, SFMono-Regular, &apos;SF Mono&apos;, Menlo, Consolas, "
        "&apos;DejaVu Sans Mono&apos;, monospace")

# Applied as presentation attributes rather than a <style> block, so the banner
# does not depend on an embedded stylesheet surviving any sanitiser.
def _font(size: int) -> str:
    return f'font-family="{FONT}" font-size="{size}px"'


def build(theme_name: str) -> str:
    c = THEMES[theme_name]
    rows, y, t0, clips, anim = [], BAR + 34, 0.30, [], []

    for idx, (kind, segs) in enumerate(LINES):
        plain = "".join(s for _, s in segs)
        indent = PAD_X + (0 if kind == "cmd" else CH * 2)
        n_chars = len(plain) + (2 if kind == "cmd" else 0)
        rate = TYPE_CMD if kind == "cmd" else TYPE_OUT
        dur = max(0.18, n_chars * rate)
        width = n_chars * CH + 14

        # one clip per line; its width animates 0 -> width => left-to-right reveal
        cid = f"t{idx}"
        clips.append(
            f'<clipPath id="{cid}"><rect x="{indent - 4:.0f}" y="{y - FS - 4:.0f}" '
            f'height="{FS + 12}" width="0">'
            f'<animate attributeName="width" from="0" to="{width:.0f}" '
            f'begin="{t0:.2f}s" dur="{dur:.2f}s" calcMode="linear" fill="freeze"/>'
            f'</rect></clipPath>')

        spans, x = [], indent
        if kind == "cmd":
            spans.append(f'<tspan fill="{c["prompt"]}">$</tspan>'
                         f'<tspan fill="{c["dim"]}"> </tspan>')
            x += CH * 2
        for role, text in segs:
            fill = {"key": c["key"], "num": c["num"], "dim": c["dim"]}.get(role, c["txt"])
            weight = ' font-weight="600"' if role in ("key", "num") else ""
            spans.append(f'<tspan fill="{fill}"{weight}>{escape(text)}</tspan>')

        rows.append(
            f'<text {_font(FS)} x="{indent:.0f}" y="{y:.0f}" clip-path="url(#{cid})">'
            + "".join(spans) + "</text>")

        t0 += dur + PAUSE
        y += LINE_H + (GAP_AFTER_OUT if kind == "out" else 0)

    # trailing prompt + blinking block cursor
    cur_y = y
    rows.append(f'<text {_font(FS)} x="{PAD_X}" y="{cur_y:.0f}">'
                f'<tspan fill="{c["prompt"]}">$</tspan></text>')
    rows.append(
        f'<rect x="{PAD_X + CH * 2:.0f}" y="{cur_y - FS + 2:.0f}" width="{CH:.0f}" '
        f'height="{FS}" fill="{c["cursor"]}" opacity="0">'
        f'<animate attributeName="opacity" values="0;0;1;1;0" keyTimes="0;0.01;0.02;0.5;0.51" '
        f'dur="1.06s" begin="{t0:.2f}s" repeatCount="indefinite"/></rect>')

    H = cur_y + 30

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}" \
width="{W}" height="{H:.0f}" role="img" \
aria-label="Terminal banner: Luna Perez Troncoso, Data Scientist and AI Engineer, Madrid">
<title>profile.sh --live — Luna Pérez Troncoso</title>
<defs>
{chr(10).join(clips)}
<linearGradient id="edge" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#0981F7"/><stop offset="1" stop-color="#7B2FBE"/>
</linearGradient>
</defs>
<rect x="1" y="1" width="{W-2}" height="{H-2:.0f}" rx="12" fill="{c['bg']}" stroke="{c['border']}"/>
<path d="M1 13 A12 12 0 0 1 13 1 H{W-13} A12 12 0 0 1 {W-1} 13 V{BAR} H1 Z" fill="{c['chrome']}"/>
<line x1="1" y1="{BAR}" x2="{W-1}" y2="{BAR}" stroke="{c['border']}"/>
<rect x="1" y="{H-5:.0f}" width="{W-2}" height="4" fill="url(#edge)" opacity="0.9"/>

<circle cx="24" cy="{BAR/2:.0f}" r="6" fill="{c['dot1']}"/>
<circle cx="45" cy="{BAR/2:.0f}" r="6" fill="{c['dot2']}"/>
<circle cx="66" cy="{BAR/2:.0f}" r="6" fill="{c['dot3']}"/>
<text {_font(13)} x="{W/2:.0f}" y="{BAR/2 + 5:.0f}" text-anchor="middle" \
fill="{c['title']}">{escape(TITLE)}</text>

{chr(10).join(rows)}
</svg>
'''


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "assets"
    out.mkdir(parents=True, exist_ok=True)
    for name in THEMES:
        p = out / f"banner-{name}.svg"
        p.write_text(build(name), encoding="utf-8")
        print(f"wrote {p}  ({p.stat().st_size / 1024:.1f} KB)")

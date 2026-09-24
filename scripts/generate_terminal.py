#!/usr/bin/env python3
"""
Generate the compact typing terminal that sits beside the coding GIF in the
About Me section of the profile README.

    python scripts/generate_terminal.py

Writes assets/terminal-dark.svg and assets/terminal-light.svg.

Same rules as the banner: no external fonts, no network, no JavaScript, and the
typing effect is SMIL on clip rects so it runs when the SVG is loaded as an
<img>. Sized narrow (700px viewBox, 14px type) so it can render at roughly half
the README column beside a 240px image and stay readable.

Edit LINES and re-run.
"""
from pathlib import Path
from xml.sax.saxutils import escape

LINES = [
    ("cmd", [("txt", "whoami")]),
    ("out", [("key", "Luna Pérez Troncoso"), ("dim", " · "),
             ("txt", "Data Scientist & AI Engineer")]),
    ("cmd", [("txt", "cat focus.txt")]),
    ("out", [("txt", "Medical & health AI"), ("dim", " — Spark pipelines to "),
             ("dim", "deployed decision tools")]),
    ("cmd", [("txt", "ls ~/portfolio | wc -l")]),
    ("out", [("num", "9"), ("dim", " projects · "), ("num", "3"), ("dim", " live apps · "),
             ("num", "109"), ("dim", " passing tests")]),
    ("cmd", [("txt", "python -c \"print(headline)\"")]),
    ("out", [("dim", "ECG macro-F1 "), ("num", "0.9166"), ("dim", " · MRI Dice "),
             ("num", "0.846"), ("dim", " · WRMSSE₇ "), ("num", "0.7717")]),
]
TITLE = "profile.sh --live"

THEMES = {
    "dark": dict(bg="#0D1117", chrome="#161B22", border="#30363D", title="#8B949E",
                 prompt="#3FB950", txt="#E6EDF3", dim="#8B949E", key="#58A6FF",
                 num="#BC8CFF", cursor="#58A6FF",
                 d1="#FF5F57", d2="#FEBC2E", d3="#28C840"),
    "light": dict(bg="#FFFFFF", chrome="#F6F8FA", border="#D1D9E0", title="#59636E",
                  prompt="#1A7F37", txt="#1F2328", dim="#59636E", key="#0550AE",
                  num="#6639BA", cursor="#0550AE",
                  d1="#FF5F57", d2="#FEBC2E", d3="#28C840"),
}

W, BAR = 700, 36
PAD_X, LINE_H, FS = 26, 25, 14
CH = FS * 0.6
GAP_AFTER_OUT = 8
TYPE_CMD, TYPE_OUT, PAUSE = 0.035, 0.006, 0.10
FONT = ("ui-monospace, SFMono-Regular, &apos;SF Mono&apos;, Menlo, Consolas, "
        "&apos;DejaVu Sans Mono&apos;, monospace")


def _f(size):
    return f'font-family="{FONT}" font-size="{size}px"'


def build(theme: str) -> str:
    c = THEMES[theme]
    rows, clips, y, t0 = [], [], BAR + 30, 0.30

    for idx, (kind, segs) in enumerate(LINES):
        plain = "".join(t for _, t in segs)
        indent = PAD_X + (0 if kind == "cmd" else CH * 2)
        n = len(plain) + (2 if kind == "cmd" else 0)
        dur = max(0.18, n * (TYPE_CMD if kind == "cmd" else TYPE_OUT))
        cid = f"t{idx}"
        clips.append(
            f'<clipPath id="{cid}"><rect x="{indent - 4:.0f}" y="{y - FS - 4:.0f}" '
            f'height="{FS + 11}" width="0">'
            f'<animate attributeName="width" from="0" to="{n * CH + 12:.0f}" '
            f'begin="{t0:.2f}s" dur="{dur:.2f}s" fill="freeze"/></rect></clipPath>')

        spans = []
        if kind == "cmd":
            spans.append(f'<tspan fill="{c["prompt"]}">$</tspan>'
                         f'<tspan fill="{c["dim"]}"> </tspan>')
        for role, text in segs:
            fill = {"key": c["key"], "num": c["num"], "dim": c["dim"]}.get(role, c["txt"])
            wt = ' font-weight="600"' if role in ("key", "num") else ""
            spans.append(f'<tspan fill="{fill}"{wt}>{escape(text)}</tspan>')
        rows.append(f'<text {_f(FS)} x="{indent:.0f}" y="{y:.0f}" '
                    f'clip-path="url(#{cid})">' + "".join(spans) + "</text>")

        t0 += dur + PAUSE
        y += LINE_H + (GAP_AFTER_OUT if kind == "out" else 0)

    cur_y = y
    rows.append(f'<text {_f(FS)} x="{PAD_X}" y="{cur_y:.0f}">'
                f'<tspan fill="{c["prompt"]}">$</tspan></text>')
    rows.append(
        f'<rect x="{PAD_X + CH * 2:.0f}" y="{cur_y - FS + 2:.0f}" width="{CH:.0f}" '
        f'height="{FS}" fill="{c["cursor"]}" opacity="0">'
        f'<animate attributeName="opacity" values="0;0;1;1;0" '
        f'keyTimes="0;0.01;0.02;0.5;0.51" dur="1.06s" begin="{t0:.2f}s" '
        f'repeatCount="indefinite"/></rect>')

    H = cur_y + 24
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}" \
width="{W}" height="{H:.0f}" role="img" \
aria-label="Terminal: Luna Perez Troncoso, Data Scientist and AI Engineer">
<title>profile.sh --live</title>
<defs>
{chr(10).join(clips)}
<linearGradient id="e" x1="0" y1="0" x2="1" y2="0">
  <stop offset="0" stop-color="#0981F7"/><stop offset="1" stop-color="#7B2FBE"/>
</linearGradient>
</defs>
<rect x="1" y="1" width="{W-2}" height="{H-2:.0f}" rx="10" fill="{c['bg']}" stroke="{c['border']}"/>
<path d="M1 11 A10 10 0 0 1 11 1 H{W-11} A10 10 0 0 1 {W-1} 11 V{BAR} H1 Z" fill="{c['chrome']}"/>
<line x1="1" y1="{BAR}" x2="{W-1}" y2="{BAR}" stroke="{c['border']}"/>
<rect x="1" y="{H-4:.0f}" width="{W-2}" height="3" fill="url(#e)" opacity="0.9"/>
<circle cx="22" cy="{BAR//2}" r="5" fill="{c['d1']}"/>
<circle cx="40" cy="{BAR//2}" r="5" fill="{c['d2']}"/>
<circle cx="58" cy="{BAR//2}" r="5" fill="{c['d3']}"/>
<text {_f(11)} x="{W//2}" y="{BAR//2 + 4}" text-anchor="middle" fill="{c['title']}">{escape(TITLE)}</text>
{chr(10).join(rows)}
</svg>
'''


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "assets"
    out.mkdir(parents=True, exist_ok=True)
    for name in THEMES:
        p = out / f"terminal-{name}.svg"
        p.write_text(build(name), encoding="utf-8")
        print(f"wrote {p}  ({p.stat().st_size/1024:.1f} KB)")

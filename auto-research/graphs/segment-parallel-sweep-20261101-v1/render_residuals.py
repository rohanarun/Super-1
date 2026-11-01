"""Render residual-vs-sweep curves from RESULTS.json as a standalone SVG (standard library only).

Usage: python3 render_residuals.py RESULTS.json out.svg
"""
import json
import math
import sys

results = json.load(open(sys.argv[1]))
SERIES = [  # (row id, label, colour, dash)
    ("trained_uniform_T1024", "trained 120M, T=1024 (N=32)", "#1f5fae", ""),
    ("trained_uniform_T2048", "trained 120M, T=2048 (N=64)", "#1f5fae", "6 4"),
    ("default_w688_s1", "default init, T=1024", "#2e8b57", ""),
    ("longmem_w688_s1", "long-memory stress, T=1024", "#c77c0e", ""),
    ("scale3_w688_s1", "saturated x3, T=1024", "#b03030", ""),
]
rows = {r["id"]: r for r in results["rows"]}
W, H, ML, MR, MT, MB = 800, 440, 70, 250, 40, 55
x_max, y_lo, y_hi = 65, -13, 1
px = lambda k: ML + (k / x_max) * (W - ML - MR)
py = lambda lv: MT + (y_hi - lv) / (y_hi - y_lo) * (H - MT - MB)
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       'font-family="Helvetica, Arial, sans-serif" font-size="12" role="img" '
       'aria-label="Boundary-input residual per sweep for each condition">',
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
       f'<text x="{ML}" y="22" font-size="14" font-weight="bold">Segment-parallel sweeps: residual per sweep (FP64, CPU)</text>']
for lv in range(y_lo, y_hi + 1):
    y = py(lv)
    out.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" stroke="#e6e6e6"/>')
    if lv % 2 == 1 or lv == y_hi:
        out.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">1e{lv}</text>')
for k in range(0, x_max + 1, 8):
    x = px(k)
    out.append(f'<line x1="{x:.1f}" y1="{H - MB}" x2="{x:.1f}" y2="{H - MB + 4}" stroke="#555"/>')
    out.append(f'<text x="{x:.1f}" y="{H - MB + 18}" text-anchor="middle">{k}</text>')
out.append(f'<line x1="{ML}" y1="{H - MB}" x2="{W - MR}" y2="{H - MB}" stroke="#555"/>')
out.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H - MB}" stroke="#555"/>')
out.append(f'<text x="{(ML + W - MR) / 2:.1f}" y="{H - 12}" text-anchor="middle">sweep k</text>')
out.append(f'<text x="18" y="{(MT + H - MB) / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {(MT + H - MB) / 2:.1f})">max change of guessed boundary inputs</text>')
y6 = py(-6)
out.append(f'<line x1="{ML}" y1="{y6:.1f}" x2="{W - MR}" y2="{y6:.1f}" stroke="#777" stroke-dasharray="3 3"/>')
out.append(f'<text x="{W - MR - 4}" y="{y6 - 5:.1f}" text-anchor="end" fill="#555">primary tolerance 1e-6</text>')
for k, lab, dy in ((16, "break-even T=1024", 12), (32, "break-even T=2048", 28)):
    x = px(k)
    out.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{H - MB}" stroke="#999" stroke-dasharray="2 4"/>')
    out.append(f'<text x="{x + 4:.1f}" y="{MT + dy}" fill="#555">{lab}</text>')
for i, (rid, label, colour, dash) in enumerate(SERIES):
    hist = rows[rid]["history"]
    pts = []
    for h in hist:
        r = h["residual"]
        lv = max(math.log10(r), y_lo) if r > 0 else y_lo   # an exactly-zero residual is drawn at the floor
        pts.append(f"{px(h['sweep']):.1f},{py(lv):.1f}")
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2"{dash_attr} points="{" ".join(pts)}"/>')
    ly = MT + 30 + 22 * i
    lx = W - MR + 16
    out.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="{colour}" stroke-width="2"{dash_attr}/>')
    out.append(f'<text x="{lx + 32}" y="{ly + 4}">{label}</text>')
out.append(f'<text x="{W - MR + 16}" y="{MT + 30 + 22 * len(SERIES) + 10}" fill="#555" font-size="11">'
           'Zero residual (exact at sweep N)</text>')
out.append(f'<text x="{W - MR + 16}" y="{MT + 30 + 22 * len(SERIES) + 24}" fill="#555" font-size="11">'
           'is drawn at 1e-13.</text>')
out.append(f'<text x="{W - MR + 16}" y="{MT + 30 + 22 * len(SERIES) + 44}" fill="#555" font-size="11">'
           'Synthetic token streams; mechanics only.</text>')
out.append("</svg>")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("wrote", sys.argv[2])

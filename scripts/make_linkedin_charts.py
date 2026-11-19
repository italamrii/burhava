"""Three minimal white LinkedIn images (same look as the project figures).

Numbers come from results/summaries/*.json. Usage:
    python scripts/make_linkedin_charts.py
then render report/linkedin/*.html to PNG with Playwright (1080x1080, scale 2).
"""

import json
from pathlib import Path

S = Path("results/summaries")
OUT = Path("report/linkedin")
a = json.loads((S / "stageA_summary.json").read_text())["mean_loss_b2"]
p2 = json.loads((S / "phase2_summary.json").read_text())["P2B_table"]

INK, INK2, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"

HEAD = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600&family=Inter:wght@400;600&display=swap">
<style>*{{box-sizing:border-box;margin:0}}
body{{width:1080px;height:1080px;background:{BG};color:{INK};direction:rtl;
font-family:'IBM Plex Sans Arabic','Inter',sans-serif;padding:90px;
display:flex;flex-direction:column;gap:22px}}
h1{{font-size:34px;font-weight:600}}
.s{{font-size:20px;color:{INK2}}}
.f{{margin-top:auto;font-size:17px;color:{MUTED}}}
svg{{width:100%;height:auto;direction:ltr}}
svg text{{font-family:'IBM Plex Sans Arabic','Inter',sans-serif}}
.legend{{display:flex;gap:26px;font-size:19px;color:{INK2}}}
.legend span{{display:inline-flex;align-items:center;gap:8px}}
.legend i{{display:inline-block;width:26px;height:3px;border-radius:2px}}
.bars{{display:flex;flex-direction:column;gap:26px;margin-top:30px}}
.lab{{font-size:20px;color:{INK2};margin-bottom:6px}}
.track{{display:flex;align-items:center;gap:12px}}
.bar{{height:40px;border-radius:4px}}
.v{{font-family:'Inter',sans-serif;font-size:20px;color:{INK};direction:ltr}}
.zones{{display:flex;flex-direction:column;gap:14px;margin-top:20px}}
.z{{display:flex;align-items:baseline;gap:18px;font-size:22px;padding:18px 0;
border-top:1px solid {GRID}}}
.z b{{font-family:'Inter',sans-serif;font-weight:600;min-width:150px;
direction:ltr;text-align:right}}
.dot{{width:12px;height:12px;border-radius:50%;display:inline-block;flex:none}}
.big{{font-family:'Inter',sans-serif;font-size:96px;font-weight:600;color:{AQUA};direction:ltr;
text-align:right;line-height:1}}
</style></head><body>"""
FOOT = '<div class="f">برهوة · simulation</div></body></html>'

# ---------------- 1: summary card
cut = round((1 - a["a10_voi2"]["mean"] / a["a03_lowconf"]["mean"]) * 100)
card = (
    HEAD
    + f"""
<h1>أي خطوة يتحقق منها الـ AI أولًا؟</h1>
<div class="big">−{cut}%</div>
<div class="s">أخطاء أقل عند الفحص حسب خريطة الاستدلال، مقارنة بـ lowest confidence</div>
<div class="zones">
<div class="z"><span class="dot" style="background:{AQUA}"></span><b>≤ 2%</b>
<span>أخطاء الخريطة: افحص حسب الخريطة</span></div>
<div class="z"><span class="dot" style="background:#eda100"></span><b>5–10%</b>
<span>robust أو self-consistency</span></div>
<div class="z"><span class="dot" style="background:{ORANGE}"></span><b>&gt; 10%</b>
<span>self-consistency فقط</span></div>
</div>
{FOOT}"""
)
(OUT / "card.html").write_text(card)

# ---------------- 2: bars
bars = [
    ("a01_none", "بدون فحص", GRAY),
    ("a03_lowconf", "lowest confidence", ORANGE),
    ("a10_voi2", "reasoning graph", BLUE),
]
rows = []
for arm, label, col in bars:
    v = a[arm]["mean"]
    rows.append(
        f'<div><div class="lab">{label}</div><div class="track">'
        f'<div class="bar" style="width:{v / 0.25 * 80:.1f}%;background:{col}"></div>'
        f'<span class="v">{v:.3f}</span></div></div>'
    )
chart1 = (
    HEAD
    + f"""
<h1>متوسط الخسارة (loss)</h1>
<div class="s">ميزانية فحصين · الأقل أفضل</div>
<div class="bars">{"".join(rows)}</div>
{FOOT}"""
)
(OUT / "chart1.html").write_text(chart1)

# ---------------- 3: lines vs graph error (ECE 0.10, verifier 5%)
tab = [r for r in p2 if abs(r["ece"] - 0.10) < 1e-9]
series = [
    ("a10_voi2", "reasoning graph", BLUE, ""),
    ("a06_selfcons", "self-consistency", AQUA, ""),
    ("a01_none", "بدون فحص", GRAY, "6 5"),
]
W, H, L, R, T, B = 900, 620, 56, 20, 16, 80
y0, y1 = 0.20, 0.56


def px(e):
    return L + e / 0.25 * (W - L - R)


def py(v):
    return T + (y1 - v) / (y1 - y0) * (H - T - B)


g = [f'<svg viewBox="0 0 {W} {H}">']
for v in (0.2, 0.3, 0.4, 0.5):
    g.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{W - R}" y2="{py(v):.1f}" stroke="{GRID}"/>')
    g.append(
        f'<text x="{L - 10}" y="{py(v) + 6:.1f}" text-anchor="end" font-size="17" '
        f'fill="{MUTED}">{v:.1f}</text>'
    )
for e in (0, 0.05, 0.10, 0.25):
    g.append(
        f'<text x="{px(e):.1f}" y="{H - 48}" text-anchor="middle" font-size="17" '
        f'fill="{MUTED}">{round(e * 100)}%</text>'
    )
g.append(
    f'<text x="{(L + W - R) / 2:.1f}" y="{H - 12}" text-anchor="middle" font-size="18" '
    f'fill="{INK2}" direction="rtl">نسبة الأخطاء في الخريطة (graph errors)</text>'
)
for arm, _, col, dash in series:
    pts = [(px(r["edge"]), py(r["loss"][arm])) for r in tab]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    g.append(
        f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="3" '
        f'stroke-dasharray="{dash}"/>'
    )
    for x, y in pts:
        g.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{col}" stroke="{BG}" stroke-width="2"/>'
        )
g.append("</svg>")
legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for _, n, c, _ in series)
chart2 = (
    HEAD
    + f"""
<h1>لما تكون الخريطة غلط</h1>
<div class="s">متوسط الخسارة (loss) · الأقل أفضل</div>
<div class="legend">{legend}</div>
{"".join(g)}
{FOOT}"""
)
(OUT / "chart2.html").write_text(chart2)
print("ok", cut)

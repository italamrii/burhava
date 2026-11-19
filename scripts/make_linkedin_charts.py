"""Two simple Arabic/English LinkedIn charts from results/summaries (no hand-typed numbers).

Usage: python scripts/make_linkedin_charts.py   (then render with Playwright, see report/linkedin)
"""

import json
from pathlib import Path

S = Path("results/summaries")
OUT = Path("report/linkedin")
a = json.loads((S / "stageA_summary.json").read_text())["mean_loss_b2"]
p2 = json.loads((S / "phase2_summary.json").read_text())["P2B_table"]

HEAD = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600;700&family=Inter:wght@500;700;800&display=swap">
<style>*{box-sizing:border-box;margin:0}
body{width:1080px;height:1080px;background:#0f1b2a;color:#f2f5f8;
font-family:'IBM Plex Sans Arabic',sans-serif;direction:rtl;padding:80px;
display:flex;flex-direction:column;gap:26px}
.bars{display:flex;flex-direction:column;gap:34px;margin-top:20px}
.row{display:flex;flex-direction:column;gap:10px}
.row .lab{font-size:28px;color:#e6edf3}
.track{display:flex;align-items:center;gap:18px}
.bar{height:64px;border-radius:8px}
.row .v{font-family:'Inter',sans-serif;font-size:34px;font-weight:800;direction:ltr}
.k{color:#8fb4d6;font-size:26px;font-weight:600}
h1{font-size:54px;line-height:1.25;font-weight:700}
.sub{font-size:28px;color:#c9d4df;line-height:1.5}
svg{width:100%;height:auto;direction:ltr}
svg text{font-family:'IBM Plex Sans Arabic','Inter',sans-serif}
.f{margin-top:auto;color:#8fa3b8;font-size:22px}
.legend{display:flex;gap:30px;font-size:24px;color:#d5dee8;flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:12px}
.legend i{display:inline-block;width:34px;height:6px;border-radius:3px}
</style></head><body>"""

# ---------------- chart 1: bars
bars = [
    ("a01_none", "بدون أي فحص", "#5f6f80"),
    ("a03_lowconf", "الأضعف ثقة أولًا (lowest confidence)", "#c98a3a"),
    ("a10_voi2", "حسب خريطة الاستدلال (reasoning graph)", "#6fd3a4"),
]
vmax = 0.25
svg = ['<div class="bars">']
for arm, label, col in bars:
    v = a[arm]["mean"]
    pct = v / vmax * 82  # % of the track width
    svg.append(
        f'<div class="row"><div class="lab">{label}</div><div class="track">'
        f'<div class="bar" style="width:{pct:.1f}%;background:{col}"></div>'
        f'<span class="v">{v:.3f}</span></div></div>'
    )
svg.append("</div>")
cut = 1 - a["a10_voi2"]["mean"] / a["a03_lowconf"]["mean"]
html1 = (
    HEAD
    + f"""
<div class="k">برهوة · simulation · ميزانية فحصين لكل جواب</div>
<h1>نفس عدد الفحوص… أخطاء أقل بـ {round(cut * 100)}%</h1>
<div class="sub">متوسط الخسارة (loss) لكل طريقة اختيار — <b>الأقل أفضل</b></div>
{"".join(svg)}
<div class="sub">الفرق ليس في عدد الفحوص، بل في <b>أي خطوة</b> تختار تفحصها.</div>
<div class="f">64 ألف حالة لكل نوع سلسلة · فواصل ثقة 95% ضيقة جدًا (±0.002)</div>
</body></html>"""
)
(OUT / "chart1.html").write_text(html1)

# ---------------- chart 2: lines vs graph error (ECE 0.10, verifier 5%)
rows = [r for r in p2 if abs(r["ece"] - 0.10) < 1e-9]
series = [
    ("a10_voi2", "الاعتماد على الخريطة", "#6fd3a4", ""),
    ("a06_selfcons", "self-consistency", "#7fb2ff", ""),
    ("a01_none", "بدون أي فحص", "#8a97a6", "8 8"),
]
W2, H2, L, R, T, B = 920, 600, 70, 30, 20, 100
ymin, ymax = 0.20, 0.56


def px(e):
    return L + (e / 0.25) * (W2 - L - R)


def py(v):
    return T + (ymax - v) / (ymax - ymin) * (H2 - T - B)


s2 = [f'<svg viewBox="0 0 {W2} {H2}">']
for v in (0.2, 0.3, 0.4, 0.5):
    s2.append(f'<line x1="{L}" y1="{py(v):.1f}" x2="{W2 - R}" y2="{py(v):.1f}" stroke="#25364a"/>')
    s2.append(
        f'<text x="{L - 12}" y="{py(v) + 8:.1f}" text-anchor="end" font-size="22" '
        f'fill="#8fa3b8">{v:.1f}</text>'
    )
# danger zone shading beyond 5%
s2.append(
    f'<rect x="{px(0.05):.1f}" y="{T}" width="{px(0.25) - px(0.05):.1f}" '
    f'height="{H2 - T - B}" fill="#6a2626" opacity="0.25"/>'
)
for e in (0, 0.05, 0.10, 0.25):
    s2.append(
        f'<text x="{px(e):.1f}" y="{H2 - 62}" text-anchor="middle" font-size="24" '
        f'fill="#c9d4df">{round(e * 100)}%</text>'
    )
s2.append(
    f'<text x="{(L + W2 - R) / 2:.1f}" y="{H2 - 14}" text-anchor="middle" font-size="24" '
    f'fill="#8fa3b8" direction="rtl">نسبة الأخطاء في خريطة الاستدلال (graph errors)</text>'
)
for arm, _, col, dash in series:
    pts = [(px(r["edge"]), py(r["loss"][arm])) for r in rows]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    s2.append(
        f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="6" '
        f'stroke-linejoin="round" stroke-dasharray="{dash}"/>'
    )
    for x, y in pts:
        s2.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{col}" '
            f'stroke="#0f1b2a" stroke-width="3"/>'
        )
s2.append("</svg>")
legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for _, n, c, _ in series)
g0 = next(r for r in rows if r["edge"] == 0)["loss"]
g25 = next(r for r in rows if abs(r["edge"] - 0.25) < 1e-9)["loss"]
html2 = (
    HEAD
    + f"""
<div class="k">برهوة · simulation · الفخ اللي ما توقعناه</div>
<h1>لما تكون الخريطة غلط… تصير الأداة ضارة</h1>
<div class="sub">متوسط الخسارة (loss) — <b>الأقل أفضل</b>.
المنطقة الحمراء: الخريطة فيها 5% أخطاء أو أكثر.</div>
<div class="legend">{legend}</div>
{"".join(s2)}
<div class="sub">خريطة صحيحة: الأفضل ({g0["a10_voi2"]:.2f}).
خريطة فيها 25% أخطاء: الأسوأ ({g25["a10_voi2"]:.2f})،
أسوأ حتى من عدم الفحص ({g25["a01_none"]:.2f}).</div>
<div class="f">24 ألف حالة لكل نقطة · ثقة غير دقيقة (ECE 0.10) ومُحقِّق يخطئ 5%</div>
</body></html>"""
)
(OUT / "chart2.html").write_text(html2)
print("cut", round(cut * 100), "g0", g0["a10_voi2"], "g25", g25["a10_voi2"], g25["a01_none"])

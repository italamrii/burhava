"""Build report/burhava_report.html (Arabic summary page) from results/summaries/*.json.

Usage: python scripts/make_report_page.py
"""

import json
from pathlib import Path

S = Path("results/summaries")
a = json.loads((S / "stageA_summary.json").read_text())
b = json.loads((S / "stageB_summary.json").read_text())
p2 = json.loads((S / "phase2_summary.json").read_text())

AR = {
    "a01_none": "بلا تحقق",
    "a02_random": "عشوائي",
    "a03_lowconf": "الأضعف ثقة أولًا",
    "a04_entropy": "الأكثر غموضًا",
    "a05_degree": "الأكثر استخدامًا",
    "a06_selfcons": "الاتساق الذاتي",
    "a07_topo": "بالترتيب",
    "a08_propagation": "التأثير على النتيجة",
    "a09_voi1": "قيمة المعلومة (خطوة)",
    "a10_voi2": "قيمة المعلومة (خطوتان)",
    "a10_voi2_indep": "قيمة المعلومة بلا ارتباط",
}


def f3(x):
    return f"{x:.3f}"


# ---------- chart 1: mean loss by arm, Stage A budget 2 (dot plot with CI)
rows = sorted(a["mean_loss_b2"].items(), key=lambda kv: kv[1]["mean"])
rows = [r for r in rows if r[0] != "a10_voi2_indep"]
W, H, L, R, T, RH = 640, 40 + 30 * len(rows), 200, 30, 20, 30
lo_x, hi_x = 0.10, 0.25


def sx(v):
    return L + (v - lo_x) / (hi_x - lo_x) * (W - L - R)


svg1 = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="متوسط الخسارة لكل طريقة">']
for t in (0.10, 0.15, 0.20, 0.25):
    x = sx(t)
    svg1.append(f'<line x1="{x:.1f}" y1="{T - 6}" x2="{x:.1f}" y2="{H - 24}" class="grid"/>')
    svg1.append(f'<text x="{x:.1f}" y="{H - 6}" class="tick" text-anchor="middle">{t:.2f}</text>')
for k, (arm, r) in enumerate(rows):
    y = T + k * RH + 8
    hl = arm in ("a10_voi2", "a08_propagation")
    cls = "dot hl" if hl else "dot"
    svg1.append(
        f'<text x="{L - 12}" y="{y + 4}" class="lab{" hlt" if hl else ""}" '
        f'text-anchor="start" direction="rtl">{AR[arm]}</text>'
    )
    svg1.append(
        f'<line x1="{sx(r["lo"]):.1f}" y1="{y}" x2="{sx(r["hi"]):.1f}" y2="{y}" '
        f'class="ci{" hl" if hl else ""}"/>'
    )
    svg1.append(
        f'<circle cx="{sx(r["mean"]):.1f}" cy="{y}" r="6" class="{cls}">'
        f"<title>{AR[arm]}: {f3(r['mean'])} (95%: {f3(r['lo'])}–{f3(r['hi'])})</title>"
        f"</circle>"
    )
    svg1.append(f'<text x="{sx(r["hi"]) + 8:.1f}" y="{y + 4}" class="val">{f3(r["mean"])}</text>')
svg1.append("</svg>")

# ---------- chart 2: loss vs edge error (phase 2 B, ECE 0.10)
tab = [t for t in p2["P2B_table"] if abs(t["ece"] - 0.10) < 1e-9]
series = [
    ("a03_lowconf", "s2"),
    ("a06_selfcons", "s3"),
    ("a10_voi2", "s1"),
    ("a12_robust_voi", "s4"),
    ("a01_none", "s5"),
]
names = dict(AR, a12_robust_voi="قيمة المعلومة المتينة (جديد)")
W2, H2, L2, R2, T2, B2 = 640, 330, 52, 190, 20, 44
xs = [t["edge"] for t in tab]
ymin, ymax = 0.20, 0.56


def px(e):
    return L2 + (e / 0.25) * (W2 - L2 - R2)


def py(v):
    return T2 + (ymax - v) / (ymax - ymin) * (H2 - T2 - B2)


svg2 = [f'<svg viewBox="0 0 {W2} {H2}" role="img" aria-label="الخسارة مقابل أخطاء الخريطة">']
for v in (0.2, 0.3, 0.4, 0.5):
    svg2.append(f'<line x1="{L2}" y1="{py(v):.1f}" x2="{W2 - R2}" y2="{py(v):.1f}" class="grid"/>')
    svg2.append(
        f'<text x="{L2 - 8}" y="{py(v) + 4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>'
    )
for e in xs:
    svg2.append(
        f'<text x="{px(e):.1f}" y="{H2 - 22}" class="tick" text-anchor="middle">'
        f"{int(round(e * 100))}%</text>"
    )
svg2.append(
    f'<text x="{(L2 + W2 - R2) / 2:.1f}" y="{H2 - 4}" class="axis" text-anchor="middle">'
    f"نسبة الأخطاء في خريطة الاستدلال</text>"
)
ends = []
for arm, cls in series:
    pts = [(px(t["edge"]), py(t["loss"][arm])) for t in tab]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    svg2.append(f'<polyline points="{d}" class="line {cls}"/>')
    for (x, y), t in zip(pts, tab, strict=True):
        svg2.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="pt {cls}"><title>'
            f"{names[arm]} — أخطاء {int(round(t['edge'] * 100))}%: "
            f"{f3(t['loss'][arm])}</title></circle>"
        )
    ends.append([pts[-1][1], names[arm], cls])
ends.sort()
for i in range(1, len(ends)):
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 16)
for y, nm, cls in ends:
    svg2.append(
        f'<text x="{W2 - R2 + 10}" y="{y + 4:.1f}" class="lab end {cls}t" '
        f'direction="rtl" text-anchor="end" dx="{R2 - 14}">{nm}</text>'
    )
svg2.append("</svg>")
legend = "".join(
    f'<span class="key"><i class="sw {cls}"></i>{names[arm]}</span>' for arm, cls in series
)
svg2.append(f'<div class="legend">{legend}</div>')


def ci(r, d=3):
    return f"{r['mean']:+.{d}f} <span class='ci'>[{r['lo']:+.{d}f}, {r['hi']:+.{d}f}]</span>"


H1 = a["H1"]["per_structure"]
hyp = [
    (
        "H1",
        "استخدام الخريطة يتفوّق على «افحص الأضعف ثقة أولًا»",
        True,
        f"في السلاسل المختلطة: {ci(H1['mixed'])}",
    ),
    ("H2", "النظر خطوتين للأمام يفيد في سلاسل «و»", True, ci(a["H2"], 4) + " (أثر صغير)"),
    ("H3", "نمذجة ارتباط المصادر مهمة", False, "الفرق 4.1% فقط (المطلوب 5%)"),
    ("H4", "الفائدة تصمد أمام أخطاء الثقة", True, ci(b["H4"]["G_B_ece10_ver5_matched"])),
    ("H5", "النتائج تنتقل إلى نماذج لغة حقيقية", None, "لم تُشغَّل: تنتظر الموافقة على الميزانية"),
    (
        "H6a",
        "«تحقّق ثم اصرف الميزانية» يصلح التوقف المبكر",
        True,
        ci(p2["H6a_arm11_vs_arm10_b3_5"], 4),
    ),
    (
        "H6c",
        "حساب الخسارة عند الاختيار يضيف شيئًا",
        False,
        ci(p2["H6c_arm11_vs_arm8"], 4) + " — لا يضيف",
    ),
    ("H7a", "النسخة المتينة تتفوّق عند 10% أخطاء خريطة", True, ci(p2["H7a_arm12_vs_arm10_edge10"])),
    (
        "H7d",
        "النسخة المتينة غير ضارة عند 25% أخطاء",
        False,
        ci(p2["H7d_arm12_vs_none_edge25"]) + " — ما زالت ضارة",
    ),
]
chip = {True: ("ok", "مدعومة"), False: ("no", "مرفوضة"), None: ("wait", "معلّقة")}
hyp_rows = "\n".join(
    f'<tr><td class="id">{i}</td><td>{t}</td>'
    f'<td><span class="chip {chip[v][0]}">{chip[v][1]}</span>'
    f'</td><td class="num">{n}</td></tr>'
    for i, t, v, n in hyp
)

html = Path("scripts/report_template.html").read_text()
html = (
    html.replace("{{SVG1}}", "".join(svg1))
    .replace("{{SVG2}}", "".join(svg2))
    .replace("{{HYP}}", hyp_rows)
    .replace("{{LOSS10}}", f3(a["mean_loss_b2"]["a10_voi2"]["mean"]))
    .replace("{{LOSS3}}", f3(a["mean_loss_b2"]["a03_lowconf"]["mean"]))
    .replace("{{LOSS1}}", f3(a["mean_loss_b2"]["a01_none"]["mean"]))
)
Path("report/burhava_report.html").write_text(html)
print("wrote report/burhava_report.html")

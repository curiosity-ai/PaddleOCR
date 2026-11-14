"""Render one synthetic A4 page (title, paragraphs, table) at several DPIs.

Layout is computed once in points at a 720-dpi reference, so every DPI has the same line
breaks and cell geometry; only the pixel density changes. Text is re-rasterised at each DPI
(as a PDF viewer would), not downsampled from a large image.
Writes pages/page_<dpi>.png and truth.json.
"""
import json, sys, os
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
DPIS = [int(x) for x in sys.argv[2].split(",")]
FD = "/usr/share/fonts/truetype/liberation/"
REF = 10.0  # px per point for layout measurement
A4 = (595.276, 841.89)  # pt
M = 56.7  # 20 mm

TITLE = "Quarterly Operations Report"
P1 = ("The regional logistics network handled a record volume of parcels during the third quarter, "
      "driven mainly by seasonal demand in the northern distribution centres. Average delivery time "
      "fell from 3.4 days to 2.9 days after the introduction of the new routing software, while "
      "customer complaints decreased by eleven percent compared with the previous period.")
P2 = ("Operating costs remained stable despite higher fuel prices. The finance team attributes this "
      "to improved vehicle utilisation and a renegotiated contract with the largest carrier. "
      "The table below summarises shipments, revenue and on-time performance for each region.")
TABLE = [["Region", "Shipments", "Revenue (EUR)", "On-time"],
         ["North", "12,480", "1,254,300", "96.2%"],
         ["South", "9,315", "874,950", "93.8%"],
         ["East", "7,022", "648,120", "91.5%"],
         ["West", "10,764", "1,018,600", "95.1%"],
         ["Central", "14,203", "1,389,775", "97.4%"],
         ["Total", "53,784", "5,185,745", "94.8%"]]
P3 = ("Looking ahead, management plans to open two additional hubs in the east and to extend the "
      "evening collection service to all urban areas. A detailed investment proposal will be "
      "presented to the board at the end of the fiscal year.")

def font(name, pt, scale):
    return ImageFont.truetype(FD + name, pt * scale)

def wrap(text, name, pt, width_pt):
    f = font(name, pt, REF)
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.getlength(t) / REF <= width_pt: cur = t
        else: lines.append(cur); cur = w
    lines.append(cur)
    return lines

# ---- layout in points -------------------------------------------------------------------
items = []  # (kind, x, y, text, fontname, pt)
y = M
items.append(("text", M, y, TITLE, "LiberationSerif-Bold.ttf", 20)); y += 20 * 1.5 + 14
BODY, LEAD = 11, 11 * 1.45
W = A4[0] - 2 * M
for para in (P1, P2):
    for ln in wrap(para, "LiberationSerif-Regular.ttf", BODY, W):
        items.append(("text", M, y, ln, "LiberationSerif-Regular.ttf", BODY)); y += LEAD
    y += 8
y += 6
# table
colw = [W * f for f in (0.28, 0.24, 0.28, 0.20)]
rowh, tpt = 24, 10.5
tx0, ty0 = M, y
for r, row in enumerate(TABLE):
    cx = tx0
    for c, cell in enumerate(row):
        fn = "LiberationSans-Bold.ttf" if r in (0, len(TABLE) - 1) else "LiberationSans-Regular.ttf"
        f = font(fn, tpt, REF); tw = f.getlength(cell) / REF
        x = cx + 8 if c == 0 else cx + colw[c] - 8 - tw
        items.append(("text", x, ty0 + r * rowh + (rowh - tpt * 1.2) / 2 - 1, cell, fn, tpt))
        cx += colw[c]
y = ty0 + len(TABLE) * rowh + 18
for ln in wrap(P3, "LiberationSerif-Regular.ttf", BODY, W):
    items.append(("text", M, y, ln, "LiberationSerif-Regular.ttf", BODY)); y += LEAD
assert y < A4[1] - M, y

truth = {"title": TITLE, "paragraphs": [P1, P2, P3], "table": TABLE}
os.makedirs(OUT, exist_ok=True)
json.dump(truth, open(os.path.join(OUT, "truth.json"), "w"), indent=1)

for dpi in DPIS:
    s = dpi / 72.0
    W_px, H_px = round(A4[0] * s), round(A4[1] * s)
    img = Image.new("RGB", (W_px, H_px), "white"); d = ImageDraw.Draw(img)
    for _, x, yy, text, fn, pt in items:
        d.text((x * s, yy * s), text, fill="black", font=font(fn, pt, s))
    lw = max(1, round(0.75 * s))
    xs = [tx0]; [xs.append(xs[-1] + w) for w in colw]
    for r in range(len(TABLE) + 1):
        yy = round((ty0 + r * rowh) * s); d.line([(round(xs[0] * s), yy), (round(xs[-1] * s), yy)], fill="black", width=lw)
    for x in xs:
        xx = round(x * s); d.line([(xx, round(ty0 * s)), (xx, round((ty0 + len(TABLE) * rowh) * s))], fill="black", width=lw)
    img.save(os.path.join(OUT, f"page_{dpi:03d}.png"))
    print(dpi, W_px, H_px)

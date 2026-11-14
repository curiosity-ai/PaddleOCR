"""Compare each variant's markdown with the bf16 baseline, page by page.

Usage: score_variants.py <root> <truth.json> variant [variant ...]
  <root>/base/*.md is the baseline; <root>/<variant>/*.md are the variants.
Reports, per variant: pages byte-identical to the baseline, character similarity, words changed,
and for the synthetic pages the text CER and table cells against ground truth.
"""
import difflib, glob, html, json, os, re, sys
from html.parser import HTMLParser

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def norm(s): return re.sub(r"\s+", " ", re.sub(r"[#*]", "", s)).strip()

class T(HTMLParser):
    def __init__(s): super().__init__(); s.rows = []; s.cell = None; s.row = None
    def handle_starttag(s, t, a):
        if t == "tr": s.row = []
        elif t in ("td", "th"): s.cell = ""
    def handle_endtag(s, t):
        if t in ("td", "th") and s.cell is not None: s.row.append(norm(s.cell)); s.cell = None
        elif t == "tr" and s.row is not None: s.rows.append(s.row); s.row = None
    def handle_data(s, d):
        if s.cell is not None: s.cell += d

def synthetic_score(md, truth):
    m = re.search(r"<table.*?</table>", md, re.S | re.I)
    rows = None
    if m:
        p = T(); p.feed(html.unescape(m.group(0))); rows = p.rows
    body = re.sub(r"<table.*?</table>", " ", md, flags=re.S | re.I)
    got = norm(re.sub(r"<[^>]+>", " ", html.unescape(body)))
    ref = norm(" ".join([truth["title"]] + truth["paragraphs"]))
    cells = sum(1 for r, row in enumerate(truth["table"]) for c, v in enumerate(row)
                if rows and r < len(rows) and c < len(rows[r]) and rows[r][c].replace(" ", "") == v.replace(" ", ""))
    return lev(ref, got) / len(ref), cells

def words(s): return re.findall(r"\S+", s)

def main():
    root, truth_path, variants = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3:]
    pages = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(root, "base", "*.md")))
    base = {p: open(os.path.join(root, "base", p + ".md"), encoding="utf-8").read() for p in pages}
    print(f"{'variant':16s} {'identical':>9s} {'min char sim':>12s} {'words changed':>13s} | " + " | ".join(f"{p[:11]:>11s}" for p in pages))
    for v in variants:
        sims, changed, ident, cells = [], 0, 0, ""
        row = []
        for p in pages:
            f = os.path.join(root, v, p + ".md")
            if not os.path.exists(f): row.append("missing"); continue
            md = open(f, encoding="utf-8").read()
            same = md == base[p]
            ident += same
            sim = 1.0 if same else difflib.SequenceMatcher(None, base[p], md, autojunk=False).ratio()
            sims.append(sim)
            wa, wb = words(base[p]), words(md)
            nchg = 0 if same else sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes() if tag != "equal")
            changed += nchg
            row.append("=" if same else f"{sim:.4f}/{nchg}w")
        print(f"{v:16s} {ident:>6d}/{len(pages)} {min(sims) if sims else 0:12.4f} {changed:13d} | " + " | ".join(f"{x:>11s}" for x in row))
    print("\nsynthetic pages against ground truth (text CER / table cells of 28):")
    for v in ["base"] + variants:
        out = []
        for p in pages:
            if not p.startswith("synth"): continue
            f = os.path.join(root, v, p + ".md")
            if os.path.exists(f):
                cer, cells = synthetic_score(open(f, encoding="utf-8").read(), truth_path)
                out.append(f"{p}: {cer*100:.2f}% / {cells}")
        print(f"  {v:16s} " + "   ".join(out))

main()

"""Score parse output against truth.json. Usage: score.py <out_root> <truth.json> [times.txt]"""
import json, re, sys, os, html
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
    def __init__(s): super().__init__(); s.rows=[]; s.cell=None; s.row=None
    def handle_starttag(s, t, a):
        if t == "tr": s.row = []
        elif t in ("td", "th"): s.cell = ""
    def handle_endtag(s, t):
        if t in ("td", "th") and s.cell is not None: s.row.append(norm(s.cell)); s.cell = None
        elif t == "tr" and s.row is not None: s.rows.append(s.row); s.row = None
    def handle_data(s, d):
        if s.cell is not None: s.cell += d

def parse_table(md):
    m = re.search(r"<table.*?</table>", md, re.S | re.I)
    if not m: return None
    p = T(); p.feed(html.unescape(m.group(0))); return p.rows

truth = json.load(open(sys.argv[2]))
text_truth = norm(" ".join([truth["title"]] + truth["paragraphs"]))
times = {}
if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
    for l in open(sys.argv[3]):
        p = l.split()
        if len(p) == 2 and p[0].isdigit(): times[int(p[0])] = float(p[1])

rows = []
for d in sorted(os.listdir(sys.argv[1])):
    mdp = os.path.join(sys.argv[1], d, "page_%s.md" % d)
    if not os.path.exists(mdp): continue
    md = open(mdp).read()
    tbl = parse_table(md)
    body = re.sub(r"<table.*?</table>", " ", md, flags=re.S | re.I)
    body = re.sub(r"<[^>]+>", " ", html.unescape(body))
    got = norm(body)
    cer = lev(text_truth, got) / len(text_truth)
    tw, gw = text_truth.split(), got.split()
    wer = lev(tw, gw) / len(tw) if False else None
    # word-level accuracy via set-multiset overlap
    from collections import Counter
    ov = sum((Counter(tw) & Counter(gw)).values()) / len(tw)
    # table
    tt = truth["table"]; flat_t = [c for r in tt for c in r]
    if tbl:
        shape = f"{len(tbl)}x{max(len(r) for r in tbl)}"
        cells_ok = sum(1 for r, row in enumerate(tt) for c, v in enumerate(row)
                       if r < len(tbl) and c < len(tbl[r]) and tbl[r][c].replace(" ", "") == v.replace(" ", ""))
        flat_g = " | ".join(c for r in tbl for c in r); flat_ref = " | ".join(flat_t)
        tcer = lev(flat_ref, flat_g) / len(flat_ref)
    else:
        shape, cells_ok, tcer = "none", 0, 1.0
    rows.append(dict(dpi=int(d), cer=cer, word_recall=ov, table_shape=shape, cells_ok=cells_ok,
                     cells_total=len(flat_t), table_cer=tcer, seconds=times.get(int(d))))
print(json.dumps(rows, indent=1))

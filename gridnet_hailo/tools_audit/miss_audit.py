"""usage: miss_audit.py UNITS.jsonl PAGES_DIR. per page: share of text-ink mass (components >= 12 px) not covered by any box. usage: miss_audit.py OUT_DIR"""
import sys, json, glob, os
import numpy as np, cv2
sys.path.insert(0, "/mnt/storage/lib_project/models/Dolphin/wordlid_vert")
from scan_units import clean, text_ink, text_region
U = [json.loads(l) for l in open(sys.argv[1])]
pdir = sys.argv[2]
rows = []
for p in U:
    p["boxes_px"] = p["units"]; p["n"] = len(p["units"])
    g = cv2.imread(f"{pdir}/p{p['page']:03d}.jpg", 0)
    g = clean(g); ink = text_ink(g) > 0
    reg = text_region(g)
    if reg is None: continue
    X0, Y0, X1, Y1 = reg
    m = np.zeros_like(ink)
    m[Y0:Y1, X0:X1] = ink[Y0:Y1, X0:X1]
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    cov = np.zeros_like(ink)
    for x0, y0, x1, y1, _ in p["boxes_px"]:
        cov[int(y0):int(np.ceil(y1)) + 1, int(x0):int(np.ceil(x1)) + 1] = True
    big = st[1:, 4] >= 12
    tot = st[1:, 4][big].sum()
    inside = np.bincount(lab[cov & m], minlength=n)[1:]
    miss = (st[1:, 4] - inside)[big].clip(0).sum()
    rows.append((p["page"], p["dir"], p["n"], round(miss / max(tot, 1), 4)))
json.dump(rows, open(sys.argv[1] + ".miss.json", "w"), default=float)
r = np.array([x[3] for x in rows])
for d in "hv":
    v = np.array([x[3] for x in rows if x[1] == d])
    if not len(v): continue
    print(d, len(v), "median miss %.4f  pages >2%%: %d  >10%%: %d  >50%%: %d" % (np.median(v), (v > .02).sum(), (v > .1).sum(), (v > .5).sum()))
print(" mean miss h %.4f v %.4f" % tuple(np.mean([x[3] for x in rows if x[1] == d] or [0]) for d in "hv"), "worst:", [(x[0], round(float(x[3]), 2)) for x in sorted(rows, key=lambda x: -x[3])[:8]])

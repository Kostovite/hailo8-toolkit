"""Word cuts vs the white runs of each line. A word space = white column run >= 0.22 x line ink height.
merged = a space with no cut in it; split = a cut not inside any space; offset = cut - space centre."""
import json, sys, cv2, numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Dolphin/wordlid_vert")
from scan_units import clean, text_ink
U = [json.loads(l) for l in open(sys.argv[1])]
sp_n = merged = cuts_n = split = 0; off = []; ex = []
for r in U:
    if r["dir"] != "h" or len(r["units"]) < 20:
        continue
    ink = text_ink(clean(cv2.imread(f"{sys.argv[2]}/p{r['page']:03d}.jpg", 0)))
    lines = {}
    for u in r["units"]:
        lines.setdefault(u[4], []).append(u)
    for ln, us in lines.items():
        us.sort(key=lambda u: u[0])
        y0, y1 = int(us[0][1]), int(us[0][3]); x0, x1 = int(us[0][0]), int(us[-1][2])
        rows = np.nonzero(ink[y0:y1, x0:x1].any(1))[0]
        if len(rows) < 5:
            continue
        hgt = rows[-1] - rows[0] + 1
        col = ink[y0:y1, x0:x1].any(0)
        w = np.nonzero(~col)[0]
        runs = [q for q in np.split(w, np.nonzero(np.diff(w) > 1)[0] + 1) if len(q) and q[0] > 0 and q[-1] < len(col) - 1]
        spaces = [(q[0] + x0, q[-1] + 1 + x0) for q in runs if len(q) >= 0.22 * hgt]
        inner = [u[2] for u in us[:-1]]
        for a, b in spaces:
            sp_n += 1
            c = [x for x in inner if a - 1 <= x <= b + 1]
            if not c:
                merged += 1; ex.append(("merge", r["page"], ln))
            else:
                off.append(c[0] - (a + b) / 2)
        for x in inner:
            cuts_n += 1
            if not any(a - 1 <= x <= b + 1 for a, b in spaces):
                split += 1; ex.append(("split", r["page"], ln))
off = np.array(off)
print(f"spaces {sp_n}: merged {merged / sp_n:.1%}; cuts {cuts_n}: outside any space {split / cuts_n:.1%}; "
      f"offset from space centre median {np.median(off):+.1f} px (p10 {np.percentile(off, 10):+.1f}, p90 {np.percentile(off, 90):+.1f})")
import collections; print(collections.Counter(e[0] for e in ex), ex[:6])

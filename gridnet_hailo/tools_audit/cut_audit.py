"""Real-page audit of GridNet cuts vs ink. Row cuts: offset of each line boundary from the centre of the white gap
between the two lines (ink profile over the line's x-range); word cuts: does the cut column cross ink inside the line."""
import json, sys, cv2, numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Dolphin/wordlid_vert")
from scan_units import clean, text_ink
U = [json.loads(l) for l in open(sys.argv[1])]
pages_dir = sys.argv[2]
off, gapw, rowink, wcross, wn = [], [], [], 0, 0
for r in U:
    if r["dir"] != "h" or len(r["units"]) < 20:
        continue
    g = clean(cv2.imread(f"{pages_dir}/p{r['page']:03d}.jpg", 0)); ink = text_ink(g)
    lines = {}
    for x0, y0, x1, y1, ln in r["units"]:
        lines.setdefault(ln, []).append((x0, y0, x1, y1))
    L = [lines[k] for k in sorted(lines)]
    for a, b in zip(L[:-1], L[1:]):
        ya, yb = a[0][3], b[0][1]
        if abs(ya - yb) > 1:            # not adjacent (a dropped line between)
            continue
        xs0 = int(max(min(u[0] for u in a), min(u[0] for u in b))); xs1 = int(min(max(u[2] for u in a), max(u[2] for u in b)))
        ta, tb = int(a[0][1]), int(b[0][3])                          # top of line a, bottom of line b
        prof = ink[ta:tb, xs0:xs1].sum(1)
        c = int(round(ya)) - ta
        # white run containing / nearest the cut: rows with ink <= 1% of the line's peak
        thr = max(1, prof.max() * 0.01)
        white = prof <= thr
        idx = np.nonzero(white)[0]
        if len(idx) == 0:
            continue
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        run = min(runs, key=lambda q: min(abs(q[0] - c), abs(q[-1] - c)))
        mid = (run[0] + run[-1] + 1) / 2
        off.append(c - mid); gapw.append(len(run)); rowink.append(prof[min(c, len(prof) - 1)] > thr)
    for ln in L:
        for (x0, y0, x1, y1), nxt in zip(ln[:-1], ln[1:]):
            x = int(round(x1)); wn += 1
            wcross += int(ink[int(y0) + 1:int(y1) - 1, x].sum() > 0)
off = np.array(off)
print(f"row cuts {len(off)}: offset from gap centre (px, +=down) median {np.median(off):+.1f}, mean {off.mean():+.2f}, "
      f"p10 {np.percentile(off,10):+.1f} p90 {np.percentile(off,90):+.1f}; median white gap {np.median(gapw):.0f} px; "
      f"cut row touches ink {np.mean(rowink):.1%}")
print(f"word cuts {wn}: column crosses ink {wcross / wn:.1%}")

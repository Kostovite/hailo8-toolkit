"""usage: python cmp_units.py A/units.jsonl B/units.jsonl -- per-page agreement of two unit runs (B vs reference A)."""
import json, sys
import numpy as np


def iou(a, b):
    a, b = np.asarray(a)[:, None, :4], np.asarray(b)[None, :, :4]
    w = (np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0])).clip(0)
    h = (np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1])).clip(0)
    i = w * h
    ar = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return i / (ar(a) + ar(b) - i)


A = {r["page"]: r for r in map(json.loads, open(sys.argv[1]))}
B = {r["page"]: r for r in map(json.loads, open(sys.argv[2]))}
tot = {"h": [0, 0, 0, 0], "v": [0, 0, 0, 0]}
worst, flips = [], []
for p in sorted(A.keys() & B.keys()):
    a, b = A[p], B[p]
    if a["dir"] != b["dir"]:
        flips.append(p); continue
    if not a["units"] or not b["units"]:
        continue
    m = iou(a["units"], b["units"])
    same = int((m.max(1) >= 0.9).sum())                   # reference boxes reproduced at IoU >= .9
    t = tot[a["dir"]]; t[0] += len(a["units"]); t[1] += len(b["units"]); t[2] += same; t[3] += 1
    worst.append((same / len(a["units"]), p, len(a["units"]), len(b["units"])))
for d, (na, nb, s, n) in tot.items():
    print(f"{d}: {n} pages, ref units {na}, test units {nb}, ref boxes reproduced (IoU>=.9) {s / max(na, 1):.4f}")
print("direction flips", flips)
print("worst pages (agree, page, ref n, test n)", sorted(worst)[:8])

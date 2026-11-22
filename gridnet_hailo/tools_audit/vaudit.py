"""vertical pages: tall boxes (>1.6x the page's median char height = likely 2+ chars) and uncovered character-sized ink"""
import sys, json, numpy as np, cv2, collections
sys.path.insert(0, "/mnt/storage/lib_project/models/Dolphin/wordlid_vert")
from scan_units import clean, text_ink, text_region
U = [json.loads(l) for l in open(sys.argv[1])]; P = sys.argv[2]
tot = tall = 0; per = []; miss_chars = 0
for r in U:
    if r["dir"] != "v" or len(r["units"]) < 30: continue
    b = np.array([u[:4] for u in r["units"]]); h = b[:, 3] - b[:, 1]; w = b[:, 2] - b[:, 0]
    med = np.median(h); t = (h > 1.6 * med).sum()
    # uncovered character-sized blobs: dilate ink to glue a character's strokes, count blobs with no box overlap
    g = clean(cv2.imread(f"{P}/p{r['page']:03d}.jpg", 0)); ink = text_ink(g)
    X0, Y0, X1, Y1 = r["region"]; m = np.zeros_like(ink); m[Y0:Y1, X0:X1] = ink[Y0:Y1, X0:X1]
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    cov = np.zeros(m.shape, bool)
    for x0, y0, x1, y1 in b: cov[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
    hit = np.bincount(lab[cov], minlength=n)
    mw = np.median(w)
    mc = sum(1 for k in range(1, n) if hit[k] == 0 and st[k, 2] > 0.4 * mw and st[k, 3] > 0.3 * med and st[k, 4] > 40)
    tot += len(h); tall += t; miss_chars += mc; per.append((r["page"], int(t), mc, len(h)))
print(f"vertical pages {len(per)}: boxes {tot}, tall (2+ chars?) {tall} = {tall/tot:.2%}, uncovered char-sized blobs {miss_chars}")
print("worst tall:", sorted(per, key=lambda x: -x[1])[:10]); print("worst missing:", sorted(per, key=lambda x: -x[2])[:10])
json.dump(per, open(sys.argv[1] + ".vaudit.json", "w"))

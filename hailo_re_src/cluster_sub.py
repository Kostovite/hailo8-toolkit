"""cluster_sub.py: inside a cluster window (0x11000000 + cl*0x02000000), which sub-blocks get written and how their
byte counts relate to the cluster's units (sc, l2, apu, ia, lcu) and to the conv's weight bytes.

Grid conv dirs are named conv_c=..,o=..,h=..,w=..,k=..[,s=..][,q=16]; weights = c*o*k*k * (2 if q=16 else 1).
Usage: python cluster_sub.py 'GLOB'   -> per sub-block: n, median bytes, Pearson r vs each unit count / weights
"""
import collections
import glob
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _p in ("..", "."):  # hwlayout.py: ../probe locally (hefre/), ./probe in the repo
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), _p, "probe"))
from ccwdec import records  # noqa: E402
from hwlayout import layout  # noqa: E402


def corr(x, y):
    if len(x) < 5 or st.pstdev(x) == 0 or st.pstdev(y) == 0:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) * st.pstdev(x) * st.pstdev(y))


rows = collections.defaultdict(list)  # sub-block -> [(bytes, sc, l2, apu, ia, lcu, weights_share)]
tot = []                               # (weights, sum of 0x40/0x41 bytes over clusters)
for f in sorted(glob.glob(sys.argv[1])):
    m = dict(re.findall(r"([a-z]+)=(\d+)", os.path.basename(os.path.dirname(f))))
    w = int(m["c"]) * int(m["o"]) * int(m["k"]) ** 2 * (2 if m.get("q") == "16" else 1) if "o" in m and "k" in m else None
    R = records(f)
    U = {}
    for c in layout(f):
        U.update(c["clusters"])
    per = collections.defaultdict(lambda: collections.Counter())
    for r in R:
        t = r["addr"] >> 24
        if 0x11 <= t <= 0x1f and t % 2:
            cl, off = (t - 0x11) // 2, r["addr"] & 0x00ffffff
            per[cl][off >> 16] += len(r["data"])
    wsum = 0
    for cl, cnt in per.items():
        u = U.get(cl)
        if not u:
            continue
        for blk, nb in cnt.items():
            rows[blk].append((nb, len(u["sc"]), len(u["l2"]), len(u["apu"]), len(u["ia"]), len(u["lcu"])))
            if blk in (0x40, 0x41, 0x42, 0x43):
                wsum += nb
    if w:
        tot.append((w, wsum))
print("sub-block (offset>>16): n  median_B   r(sc)  r(l2)  r(apu)  r(ia)  r(lcu)")
for blk in sorted(rows):
    v = rows[blk]
    if len(v) < 5:
        continue
    b = [x[0] for x in v]
    rs = [corr(b, [x[i] for x in v]) for i in range(1, 6)]
    print(f"  +0x{blk:02x}xxxx {len(v):5d} {st.median(b):9.0f}  " + "  ".join(f"{r:5.2f}" for r in rs))
if tot:
    print(f"conv weights vs bytes in +0x40..0x43 (all clusters): r = {corr([a for a, _ in tot], [b for _, b in tot]):.3f}, "
          f"median ratio bytes/weights = {st.median(b / a for a, b in tot if a):.3f}, n = {len(tot)}")

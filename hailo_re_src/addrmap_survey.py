"""addrmap_survey.py: test address -> cluster hypotheses over many HEFs.

For each HEF: the clusters the sequencer enables (hwlayout) vs the CCW address top bytes (ccwdec).
Prints, per top byte (addr >> 24) and per 0x00X00000 sub-block, how often the HEF also enables the hypothesised
cluster (top - 0x11) / 2, plus which top bytes appear in HEFs that use exactly one cluster.
Usage: python addrmap_survey.py 'GLOB'
"""
import collections
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _p in ("..", "."):  # hwlayout.py: ../probe locally (hefre/), ./probe in the repo
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), _p, "probe"))
from ccwdec import records  # noqa: E402
from hwlayout import layout  # noqa: E402

files = sorted(glob.glob(sys.argv[1]))
top = collections.Counter()          # top byte -> HEFs seen
hit = collections.Counter()          # top byte -> HEFs where hypothesised cluster is enabled
sub = collections.Counter()          # (top & 1, (addr >> 20) & 0xf) -> count (offset within a cluster window)
single = collections.defaultdict(collections.Counter)  # enabled cluster (single-cluster HEFs) -> top bytes
bad = 0
for f in files:
    try:
        R = records(f)
        cl = set()
        for c in layout(f):
            cl |= set(c["clusters"])
    except Exception as e:  # noqa: BLE001
        bad += 1
        continue
    tops = {r["addr"] >> 24 for r in R}
    for t in tops:
        top[t] += 1
        if 0x11 <= t <= 0x1f and (t - 0x11) % 2 == 0 and (t - 0x11) // 2 in cl:
            hit[t] += 1
    for r in R:
        sub[(r["addr"] >> 24 & 0xf1 if 0x10 <= r["addr"] >> 24 <= 0x1f else r["addr"] >> 24, (r["addr"] >> 20) & 0xf)] += 1
    if len(cl) == 1:
        single[next(iter(cl))].update(tops)
print(f"{len(files)} HEFs, {bad} unreadable")
print("top byte : HEFs  hypothesis-hit")
for t in sorted(top):
    print(f"  0x{t:02x}   {top[t]:5d}  {hit[t] if 0x11 <= t <= 0x1f else '-'}")
print("single-cluster HEFs: cluster -> top bytes seen")
for c in sorted(single):
    print(f"  cl {c}: " + " ".join(f"0x{t:02x}:{n}" for t, n in sorted(single[c].items())))

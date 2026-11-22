"""Firmware runtime-data profile (hailortcli ... collect-runtime-data JSON): per context the span and where the time
goes (gap before each action, summed by action type). python rtprof.py A.json [B.json ...]"""
import json
import sys
from collections import Counter

for f in sys.argv[1:]:
    d = json.load(open(f))
    mhz = d["clock_cycle_MHz"]
    ng = d["network_groups"][0][0] if isinstance(d["network_groups"][0], list) else d["network_groups"][0]
    print(f"== {f}  activation {ng['mean_activation_time_ms']} ms")
    tot = 0
    for c in ng["contexts"]:
        a = c["actions"]
        if not a:
            continue
        span = (a[-1]["timestamp"] - a[0]["timestamp"]) / mhz / 1000
        tot += span
        gap = Counter()
        for p, q in zip(a, a[1:]):
            t = q["type"] if q["type"] != "repeated_action" else "rep:" + q["data"].get("sub_action_type", "")
            gap[t] += (q["timestamp"] - p["timestamp"]) / mhz / 1000
        print(f" {c['context_name']:12s} {span:6.3f} ms  n={len(a):4d}  " + "  ".join(f"{k}={v:.3f}" for k, v in gap.most_common(4)))
    print(f" total {tot:.3f} ms")

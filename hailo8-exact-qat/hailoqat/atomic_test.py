"""Unit-test atomic.py against every captured instance: recorded inputs -> our op -> compare with recorded output.

Usage: python atomic_test.py CAPTURE_DIR [CAPTURE_DIR ...] [--types A,B] [--show N]
"""
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch

import atomic as A

caps = [a for a in sys.argv[1:] if not a.startswith("--")]
types = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--types=")), None)
show = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--show=")), 2))
stat = defaultdict(lambda: [0, 0, 0])  # exact, inexact, error
shown = defaultdict(int)


def nchw(a):
    a = np.asarray(a, np.float64)
    return torch.as_tensor(a).permute(0, 3, 1, 2) if a.ndim == 4 else torch.as_tensor(a)


for cap in caps:
    d = pickle.load(open(f"{cap}/_all_ops_np.pkl", "rb"))
    for k, r in d.items():
        t = r["type"]
        if (t not in A.OPS and "perm" not in r) or (types and t not in types) or "out0" not in r \
                or (A.OPS.get(t) is A.gather and "perm" not in r) or t == "ConvStrippedOp":
            continue
        ins = [nchw(r[f"in{i}"]) for i in range(8) if f"in{i}" in r]
        try:
            f = A.gather if "perm" in r else A.OPS[t]
            if f is A.gather and t in A.OPS and A.OPS[t] is not A.gather:  # validate like Chain does
                try:
                    if not bool((A.gather(r, *ins) == nchw(r["out0"])).all()):
                        f = A.OPS[t]
                except Exception:  # noqa: BLE001
                    f = A.OPS[t]
            y = f(r, *ins) if t != "ConstOp" else f(r)
            ref = nchw(r["out0"])
            ok = y.shape == ref.shape and bool((y == ref).all())
            stat[t][0 if ok else 1] += 1
            if not ok and shown[t] < show:
                shown[t] += 1
                if y.shape != ref.shape:
                    print(f"  SHAPE {k}: got {tuple(y.shape)} ref {tuple(ref.shape)}")
                else:
                    bad = (y != ref)
                    print(f"  DIFF {k}: {int(bad.sum())}/{ref.numel()} max {float((y - ref).abs().max()):.3g} "
                          f"e.g. got {y[bad][:4].tolist()} ref {ref[bad][:4].tolist()}")
        except Exception as e:  # noqa: BLE001
            stat[t][2] += 1
            if shown[t] < show:
                shown[t] += 1
                print(f"  ERROR {k}: {type(e).__name__}: {str(e)[:150]}")
for t, (ok, bad, err) in sorted(stat.items()):
    print(f"{t:28s} exact {ok:5d}  inexact {bad:5d}  error {err:5d}")

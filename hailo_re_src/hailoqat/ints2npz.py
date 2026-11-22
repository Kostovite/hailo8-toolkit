"""QATNet.export_ints() checkpoint -> writeback.py INTS_NPZ (capture HWIO kernel layout, accumulator biases).

Inverts graph.conv_params: dense OIHW (co, ci, kh, kw) -> HWIO (kh, kw, ci, co); depthwise (ci*mult, 1, kh, kw) ->
(kh, kw, ci, mult), with ci/mult from the capture record.

Usage: python ints2npz.py INTS_PT CAPTURE_DIR OUT_NPZ
"""
import os
import sys

import numpy as np
import torch

ints_pt, cap, out = sys.argv[1:4]
ints = torch.load(ints_pt)
arrs = {}
for name, (k, b) in ints.items():
    r = np.load(os.path.join(cap, name.replace("/", "__") + ".npz"))
    k = k.double()
    if bool(r["depthwise"]):
        kh, kw, ci, mult = r["k"].shape
        hwio = k.reshape(ci, mult, kh, kw).permute(2, 3, 0, 1)
    else:
        hwio = k.permute(2, 3, 1, 0)
    assert tuple(hwio.shape) == tuple(r["k"].shape), (name, hwio.shape, r["k"].shape)
    arrs[f"{name}/k"] = hwio.numpy()
    arrs[f"{name}/b"] = b.double().reshape(-1).numpy()
np.savez(out, **arrs)
print(f"{len(ints)} layers -> {out}")

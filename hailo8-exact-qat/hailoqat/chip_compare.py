"""Bit-level comparison of Hailo-8 chip head outputs (chip_dump.py) with the exact qatfast executor on the same pages.

Usage: python chip_compare.py CHIP_NPZ HN CAPTURE_DIR
"""
import json
import sys

import cv2
import numpy as np
import torch

import graph26
import qatfast

chip_path, hn_path, cap = sys.argv[1:4]
z = np.load(chip_path)
files = list(z["files"])
dev = torch.device("cuda")
hn = json.load(open(hn_path))
net = qatfast.QATNet(hn, graph26.Params(cap, dev), dev).to(dev).eval()
names = [k for k in z.files if k != "files"]
tot = {k: [0, 0, 0] for k in names}  # mismatching elements, elements, max |diff|
for i in range(0, len(files), 4):
    imgs = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files[i:i + 4]]).astype(np.float32)
    with torch.no_grad():
        out = net(torch.from_numpy(imgs).to(dev))
    for k in names:
        ours = out[k.replace("__", "/")].permute(0, 2, 3, 1).cpu().numpy().astype(np.int64)
        chip = z[k][i:i + 4].astype(np.int64)
        d = np.abs(ours - chip)
        tot[k][0] += int((d != 0).sum())
        tot[k][1] += d.size
        tot[k][2] = max(tot[k][2], int(d.max()))
for k, (bad, n, mx) in tot.items():
    print(f"{k.replace('__', '/'):14s} {bad}/{n} elements differ, max |diff| {mx}")
print("CHIP == EXACT EXECUTOR" if all(v[0] == 0 for v in tot.values()) else "CHIP DIFFERS")

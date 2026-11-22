"""mAP from raw Hailo-8 chip head outputs (chip_dump.py), decoded exactly like evaluate.py's YOLO26 path.

Usage: python chip_map.py CHIP_NPZ HN CAPTURE_DIR PT
"""
import json
import sys

import numpy as np
import torch

import evaluate as E
from train_qat import head_names

chip_path, hn_path, cap, pt = sys.argv[1:5]
z = np.load(chip_path)
files = list(z["files"])
dev = "cuda"
hn = json.load(open(hn_path))
from ultralytics import YOLO  # noqa: E402
det = YOLO(pt).model.model[-1].to(dev).eval()
names = head_names(hn)


def deq(name, q):
    r = np.load(f"{cap}/{name.replace('/', '__')}.npz")
    sc = torch.as_tensor(np.asarray(r["out_scale"], np.float64)).to(dev).reshape(1, -1, 1, 1)
    zp = torch.as_tensor(np.asarray(r["out_zp"], np.float64)).to(dev).reshape(1, -1, 1, 1)
    return ((q - zp) * sc).float()


met = E.Metric()
labels = {}
for f in files:
    _, ls = next(E.batches([f], (896, 896), 1))
    labels[f] = ls[0]
for i in range(0, len(files), 8):
    sl = slice(i, i + 8)
    t = lambda n: torch.from_numpy(z[n.replace("/", "__")][sl].astype(np.float64)).permute(0, 3, 1, 2).to(dev)
    bx = [deq(b, t(b)) for b, _ in names]
    cl = [deq(c, t(c)) for _, c in names]
    bs = bx[0].shape[0]
    x = {"boxes": torch.cat([v.view(bs, 4, -1) for v in bx], -1),
         "scores": torch.cat([v.view(bs, det.nc, -1) for v in cl], -1), "feats": bx}
    with torch.no_grad():
        preds = det.postprocess(det._inference(x).permute(0, 2, 1))
    for j, p in enumerate(preds):
        met.add(p[p[:, 4] > 0.001], labels[files[i + j]], i + j)
print(json.dumps({"chip": chip_path, **met.result(), "n": len(files)}))

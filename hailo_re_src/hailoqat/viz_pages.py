"""Side-by-side page renders: ground truth | FP model | Hailo-8 chip output (raw chip_dump.py integers, decoded like
evaluate.py). One PNG per page plus a contact sheet.

Usage: python viz_pages.py CHIP_NPZ HN CAPTURE_DIR PT OUT_DIR [CONF]
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
import yaml

import evaluate as E
from train_qat import head_names

chip_path, hn_path, cap, pt, out_dir = sys.argv[1:6]
conf = float(sys.argv[6]) if len(sys.argv) > 6 else 0.4
names = yaml.safe_load(open("/mnt/storage/lib_project/models/Hailo_model/hailo_re/data_v14sec/data.yaml"))["names"]
os.makedirs(out_dir, exist_ok=True)
z = np.load(chip_path)
files = list(z["files"])
dev = "cuda"
from ultralytics import YOLO  # noqa: E402
fp = YOLO(pt).model.to(dev).eval().float()
det = YOLO(pt).model.model[-1].to(dev).eval()
heads = head_names(json.load(open(hn_path)))
rng = np.random.default_rng(3)
COL = {i: tuple(int(c) for c in rng.integers(40, 230, 3)) for i in range(len(names))}


def deq(name, q):
    r = np.load(f"{cap}/{name.replace('/', '__')}.npz")
    sc = torch.as_tensor(np.asarray(r["out_scale"], np.float64)).to(dev).reshape(1, -1, 1, 1)
    zp = torch.as_tensor(np.asarray(r["out_zp"], np.float64)).to(dev).reshape(1, -1, 1, 1)
    return ((q - zp) * sc).float()


def chip_preds(i):
    t = lambda n: torch.from_numpy(z[n.replace("/", "__")][i:i + 1].astype(np.float64)).permute(0, 3, 1, 2).to(dev)
    bx = [deq(b, t(b)) for b, _ in heads]
    cl = [deq(c, t(c)) for _, c in heads]
    x = {"boxes": torch.cat([v.view(1, 4, -1) for v in bx], -1),
         "scores": torch.cat([v.view(1, det.nc, -1) for v in cl], -1), "feats": bx}
    with torch.no_grad():
        return det.postprocess(det._inference(x).permute(0, 2, 1))[0]


def draw(img, boxes, title):
    im = img.copy()
    for x1, y1, x2, y2, c, s in boxes:
        col = COL[int(c)]
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        lab = names[int(c)] + (f" {s:.2f}" if s is not None else "")
        cv2.putText(im, lab, (int(x1) + 2, max(12, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(im, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return im


sheets = []
for i, f in enumerate(files):
    rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, labs = next(E.batches([f], (896, 896), 1))
    gt = [(*l[1:5], l[0], None) for l in labs[0]]
    with torch.no_grad():
        y = fp(torch.from_numpy(rgb[None]).to(dev).float().permute(0, 3, 1, 2) / 255.0)
    fpp = (y[0] if isinstance(y, (list, tuple)) else y)[0]
    fpb = [tuple(float(v) for v in r[:4]) + (int(r[5]), float(r[4])) for r in fpp.cpu().numpy() if r[4] >= conf]
    cp = chip_preds(i).cpu().numpy()
    chb = [tuple(float(v) for v in r[:4]) + (int(r[5]), float(r[4])) for r in cp if r[4] >= conf]
    row = np.concatenate([draw(bgr, gt, f"Ground truth ({len(gt)})"),
                          draw(bgr, fpb, f"FP model ({len(fpb)})"),
                          draw(bgr, chb, f"Hailo-8 chip int8 ({len(chb)})")], 1)
    name = os.path.splitext(os.path.basename(f))[0]
    cv2.imwrite(os.path.join(out_dir, f"{name}.png"), row)
    sheets.append(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2)))
    print(name, f"gt {len(gt)}  fp {len(fpb)}  chip {len(chb)}")
cv2.imwrite(os.path.join(out_dir, "contact_sheet.jpg"), np.concatenate(sheets, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("wrote", out_dir)

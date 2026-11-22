"""Precompute the FP teacher's one2one head outputs for every training page (the data has no augmentation, so they are
fixed). Removes the teacher forward (~20% of a QAT step) from every epoch of every run.

Layout in OUT_DIR: files.txt (Pages order), boxes.f32 (n, 4, A) float32, scores.f16 (n, nc, A) float16 logits,
meta.json. Each process fills its interleaved share of the same memmaps.

Usage: python teacher_cache.py OUT_DIR --shard k/n   (run n processes, one per GPU; the first creates the files)
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"


def open_cache(out, n, nc, a, mode):
    boxes = np.memmap(os.path.join(out, "boxes.f32"), np.float32, mode, shape=(n, 4, a))
    scores = np.memmap(os.path.join(out, "scores.f16"), np.float16, mode, shape=(n, nc, a))
    return boxes, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", default=f"{ROOT}/data_v14sec/train/images")
    ap.add_argument("--pt", default=f"{ROOT}/models/v14sec/best.pt")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=0, help="resize the (square) pages to the model input size")
    a = ap.parse_args()
    k, nsh = map(int, a.shard.split("/"))
    files = sorted(glob.glob(os.path.join(a.train, "*")))
    dev = torch.device("cuda")
    from ultralytics import YOLO
    tmodel = YOLO(a.pt).model.to(dev).float().eval()
    det = tmodel.model[-1]
    feats = {}
    det.register_forward_pre_hook(lambda m, inp: feats.__setitem__("x", inp[0]))

    def teacher(imgs):
        with torch.no_grad():
            tmodel(imgs.permute(0, 3, 1, 2) / 255.0)
            return det.forward_head(feats["x"], **det.one2one)

    def load(f):
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        if a.imgsz and im.shape[:2] != (a.imgsz, a.imgsz):
            im = cv2.resize(im, (a.imgsz, a.imgsz), interpolation=cv2.INTER_LINEAR)
        return im

    img0 = load(files[0])
    t0 = teacher(torch.from_numpy(img0[None]).to(dev).float())
    nc, na = t0["scores"].shape[1], t0["scores"].shape[2]
    os.makedirs(a.out, exist_ok=True)
    if k == 0:
        with open(os.path.join(a.out, "files.txt"), "w") as fh:
            fh.write("\n".join(files) + "\n")
        json.dump({"n": len(files), "nc": nc, "anchors": na, "pt": a.pt, "shape": list(img0.shape)},
                  open(os.path.join(a.out, "meta.json"), "w"))
        open_cache(a.out, len(files), nc, na, "w+")  # create both files at full size
        del t0
    else:
        while not os.path.exists(os.path.join(a.out, "meta.json")):
            time.sleep(1)
        time.sleep(3)
    boxes, scores = open_cache(a.out, len(files), nc, na, "r+")
    mine = list(range(k, len(files), nsh))
    t = time.time()
    for j in range(0, len(mine), a.bs):
        idx = mine[j:j + a.bs]
        imgs = np.stack([load(files[i]) for i in idx])
        o = teacher(torch.from_numpy(imgs).to(dev).float())
        boxes[idx] = o["boxes"].float().cpu().numpy()
        scores[idx] = o["scores"].float().cpu().numpy().astype(np.float16)
        if (j // a.bs) % 100 == 0:
            print(f"shard {k}: {j + len(idx)}/{len(mine)} {(j + len(idx)) / (time.time() - t):.1f} img/s", flush=True)
    boxes.flush()
    scores.flush()
    print(f"shard {k} DONE {len(mine)} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())

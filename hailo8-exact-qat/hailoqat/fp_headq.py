"""FP model with ONLY its one2one head outputs fake-quantized to int8, to measure what the head output quantization
costs on its own. Modes: l0 = level-0 HAR scales/zero points; clip:LO:HI = class logits clipped to [LO, HI] over 255
levels (boxes keep the L0 grid); box16 = boxes at 16 bits (L0 range), classes as l0.

Usage: python fp_headq.py MODE [--shard k/n --stats F]   (stats combine with evaluate.py --combine)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E  # noqa: E402
from train_qat import head_names  # noqa: E402

ROOT = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"


def fq(x, scale, zp, bits=8):
    q = torch.clamp(torch.round(x / scale + zp), 0, 2 ** bits - 1)
    return (q - zp) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--stats", default="")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    dev = "cuda"
    from ultralytics import YOLO
    model = YOLO(f"{ROOT}/models/v14sec/best.pt").model.to(dev).eval().float()
    det = model.model[-1]
    hn = json.load(open(f"{ROOT}/yolo26m_v14sec_L0.hn"))
    q = {}
    for i, (b, c) in enumerate(head_names(hn)):
        for kind, n in (("box", b), ("cls", c)):
            r = np.load(f"{ROOT}/capture_y26g/{n.replace('/', '__')}.npz")
            q[(kind, i)] = (float(np.asarray(r["out_scale"]).ravel()[0]), float(np.asarray(r["out_zp"]).ravel()[0]))

    def hook(kind, i):
        def f(m, inp, out):
            sc, zp = q[(kind, i)]
            bits = 8
            if a.mode.startswith("clip:") and kind == "cls":
                lo, hi = map(float, a.mode[5:].split(":"))
                sc, zp = (hi - lo) / 255.0, round(-lo / ((hi - lo) / 255.0))
            if a.mode == "box16" and kind == "box":
                lo, hi = (0 - zp) * sc, (255 - zp) * sc
                sc, zp, bits = (hi - lo) / 65535.0, round(-lo / ((hi - lo) / 65535.0)), 16
            if a.mode == "none":
                return out
            return fq(out, sc, zp, bits)
        return f

    for i in range(3):
        det.one2one_cv2[i][2].register_forward_hook(hook("box", i))
        det.one2one_cv3[i][2].register_forward_hook(hook("cls", i))
    k, nsh = (int(v) for v in a.shard.split("/"))
    files = E.val_files(0)[k::nsh]
    met, done, t0 = E.Metric(), 0, time.time()
    with torch.no_grad():
        for imgs, labs in E.batches(files, (896, 896), a.bs):
            y = model(torch.as_tensor(imgs, device=dev).permute(0, 3, 1, 2).float() / 255.0)
            preds = y[0] if isinstance(y, (list, tuple)) else y
            for d, l in zip(preds, labs):
                met.add(d[d[:, 4] > 0.001], l, k + nsh * done)
                done += 1
    if a.stats:
        np.savez(a.stats, **{key: np.concatenate(v, 0) for key, v in met.stats.items()})
    if nsh == 1:
        print(json.dumps({"mode": a.mode, **met.result(), "sec": round(time.time() - t0)}))


if __name__ == "__main__":
    main()

"""Validation mAP of the exact Hailo-8 integer model (hailoqat) and of the FP teacher, through one harness.

Same letterbox, decode (Ultralytics Detect: DFL + anchors), NMS and AP code for both, so numbers are comparable.

Usage:
  python evaluate.py hailo HN PARAM_DIR [--n N] [--bs B] [--conf 0.001] [--iou 0.7]
  python evaluate.py fp   PT_PATH H W    [--n N] [--bs B]
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from letterbox import letterbox  # noqa: E402

from ultralytics.engine.validator import BaseValidator  # noqa: E402
from ultralytics.utils.metrics import ap_per_class, box_iou  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402
from ultralytics.utils.tal import dist2bbox, make_anchors  # noqa: E402

VAL = os.environ.get("VAL_DIR", "/mnt/storage/lib_project/models/Hailo_model/hailo_re/data_v14sec/val/images")
IOUV = torch.linspace(0.5, 0.95, 10)
NC = int(os.environ.get("NC", "14"))
REG_MAX = 16  # YOLOv8 only; YOLO26 (reg_max=1, end2end) decodes through the checkpoint's own Detect head


def val_files(n):
    files = sorted(glob.glob(os.path.join(VAL, "*")))
    if n and n < len(files):
        files = sorted(random.Random(0).sample(files, n))
    return files


def load_labels(img_path, r, pad, w0, h0):
    lp = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(lp):
        return np.zeros((0, 5), np.float32)
    a = np.loadtxt(lp, ndmin=2).astype(np.float32)
    if a.size == 0:
        return np.zeros((0, 5), np.float32)
    xyxy = xywh2xyxy(torch.as_tensor(a[:, 1:5])).numpy()
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * w0 * r + pad[0]
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * h0 * r + pad[1]
    return np.concatenate([a[:, :1], xyxy], 1)


def decode(heads, strides):
    """heads: list of (reg (B,64,H,W), cls_prob (B,NC,H,W)) floats -> (B, 4+NC, A) like Detect inference output."""
    feats = [torch.cat([r, c], 1) for r, c in heads]
    anchors, stride_t = (t.transpose(0, 1) for t in make_anchors(feats, torch.tensor(strides, dtype=torch.float32), 0.5))
    x = torch.cat([f.flatten(2) for f in feats], 2)
    box, cls = x.split((4 * REG_MAX, NC), 1)
    b, _, a = box.shape
    proj = torch.arange(REG_MAX, dtype=box.dtype, device=box.device)
    dist = box.view(b, 4, REG_MAX, a).softmax(2).transpose(1, 2).mul(proj.view(1, REG_MAX, 1, 1)).sum(1)
    dbox = dist2bbox(dist, anchors.unsqueeze(0).to(box.device), xywh=True, dim=1) * stride_t.to(box.device)
    return torch.cat([dbox, cls], 1)


class Metric:
    def __init__(self):
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], img=[], rank=[])
        self.matcher = SimpleNamespace(iouv=IOUV)

    def add(self, det, labels, img_id):
        cls_t = torch.as_tensor(labels[:, 0])
        n = det.shape[0]
        self.stats["target_cls"].append(cls_t.numpy())
        self.stats["img"].append(np.full(n, img_id, np.int64))
        self.stats["rank"].append(np.arange(n, dtype=np.int64))
        if n == 0:
            self.stats["tp"].append(np.zeros((0, 10), bool))
            self.stats["conf"].append(np.zeros(0))
            self.stats["pred_cls"].append(np.zeros(0))
            return
        det = det.cpu().float()
        if len(labels):
            iou = box_iou(torch.as_tensor(labels[:, 1:5]), det[:, :4])
            tp = BaseValidator.match_predictions(self.matcher, det[:, 5], cls_t, iou)
        else:
            tp = np.zeros((n, 10), bool)
        self.stats["tp"].append(np.asarray(tp))
        self.stats["conf"].append(det[:, 4].numpy())
        self.stats["pred_cls"].append(det[:, 5].numpy())

    def result(self):
        s = {k: np.concatenate(v, 0) for k, v in self.stats.items()}
        # canonical prediction order: ap_per_class sorts by conf with an unstable sort, so tied confidences (int8 head
        # outputs have 256 levels) ranked in concatenation order and the mAP depended on the shard count
        # (L0: 0.8547 on 2 shards, 0.8521 on 4; FP-first 0.8453). Ties are broken by a fixed hash of (image, rank).
        tie = (s["img"] * 2654435761 + s["rank"] * 40503 + 12345) % (2 ** 32)
        o = np.lexsort([tie, -s["conf"]])
        s["tp"], s["conf"], s["pred_cls"] = s["tp"][o], s["conf"][o], s["pred_cls"][o]
        res = ap_per_class(s["tp"], s["conf"], s["pred_cls"], s["target_cls"])
        ap = res[5]  # (nc, 10)
        return dict(mAP50=float(ap[:, 0].mean()), mAP50_95=float(ap.mean()), P=float(res[2].mean()), R=float(res[3].mean()))


def batches(files, shape, bs):
    for i in range(0, len(files), bs):
        chunk = files[i:i + bs]
        imgs, labs = [], []
        for f in chunk:
            im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            h0, w0 = im.shape[:2]
            lb, r, pad = letterbox(im, shape)
            imgs.append(lb)
            labs.append(load_labels(f, r, pad, w0, h0))
        yield np.stack(imgs), labs


def y26_heads(hn, params, dev, pt, ints=None, impl="qatfast", apu=None):
    """YOLO26 end2end: dequantized one2one heads -> checkpoint's Detect._inference + top-k postprocess.
    params: graph26.Params. ints: optional {layer: (kernel, bias)} from QATNet.export_ints() overriding the HAR.
    impl: qatfast (fused fp32/int32 sm_52 kernels, verified bit-exact against graph26) or graph26 (float64 reference)."""
    import graph26
    from ultralytics import YOLO
    det = YOLO(pt).model.model[-1].to(dev).eval()
    layers = hn["layers"]

    def find(b, i):
        tag = f"/model.23/one2one_cv{b}.{i}/one2one_cv{b}.{i}.2/Conv"
        hits = [k for k, v in layers.items() if tag in v.get("original_names", [])]
        assert len(hits) == 1, (tag, hits)
        return hits[0]

    names = [(find(2, i), find(3, i)) for i in range(3)]

    def deq(name, q):
        r = params.conv[name]
        sc = torch.as_tensor(np.asarray(r["out_scale"], np.float64)).to(dev).reshape(1, -1, 1, 1)
        zp = torch.as_tensor(np.asarray(r["out_zp"], np.float64)).to(dev).reshape(1, -1, 1, 1)
        return ((q - zp) * sc).float()

    net = None
    if impl == "qatfast":
        import qatfast
        torch.backends.cudnn.benchmark = True
        net = qatfast.QATNet(hn, params, dev).to(dev).eval()
        if ints is not None:
            for k, m in net.mac.items():
                kk, bb = ints[k.replace("__", "/")]
                m.set_ints(kk, bb)
        if apu is not None:
            net.load_apu(apu)
    elif apu is not None:
        raise ValueError("--apu needs --impl qatfast")

    def run(imgs):
        if net is not None:
            out = {k: v.double() for k, v in net(torch.from_numpy(imgs).to(dev).float()).items()}
            if net.flag_events:
                raise RuntimeError(f"qatfast exactness flags {net.flag_events}: rerun with --impl graph26")
        else:
            out, _ = graph26.run(hn, params, imgs.astype(np.float64), dev, check=False, ints=ints)
        bx = [deq(b, out[b]) for b, _ in names]
        cl = [deq(c, out[c]) for _, c in names]
        bs = bx[0].shape[0]
        x = {"boxes": torch.cat([t.view(bs, 4, -1) for t in bx], -1),
             "scores": torch.cat([t.view(bs, det.nc, -1) for t in cl], -1), "feats": bx}
        y = det._inference(x)
        return det.postprocess(y.permute(0, 2, 1))  # (B, max_det, 6) xyxy, conf, cls

    return run


def hailo_heads(hn, params, dev):
    """Return a fn(images uint8 NHWC) -> decoded preds, running the exact integer graph."""
    import graph
    pp = [v for v in hn["layers"].values() if v["type"] == "postprocess"][0]
    names = pp["input"]  # reg, cls per stride: conv57, conv58, conv70, conv71, conv82, conv83
    in_h = hn["layers"][[k for k, v in hn["layers"].items() if v["type"] == "input_layer"][0]]["input_shapes"][0][1]

    def deq(name, q):
        r = params[name]
        sc = torch.as_tensor(np.asarray(r["out_scale"], np.float64)).to(dev).reshape(1, -1, 1, 1)
        zp = torch.as_tensor(np.asarray(r["out_zp"], np.float64)).to(dev).reshape(1, -1, 1, 1)
        return ((q - zp) * sc).float()

    def run(imgs):
        out, _ = graph.run(hn, params, imgs.astype(np.float64), dev, check=False)
        heads = [(deq(names[i], out[names[i]]), deq(names[i + 1], out[names[i + 1]])) for i in range(0, 6, 2)]
        strides = [in_h // h[0].shape[2] for h in heads]
        return decode(heads, strides)

    return run


def fp_heads(pt, dev):
    from ultralytics import YOLO
    model = YOLO(pt).model.to(dev).eval().float()

    def run(imgs):
        x = torch.as_tensor(imgs, device=dev).permute(0, 3, 1, 2).float() / 255.0
        y = model(x)
        return y[0] if isinstance(y, (list, tuple)) else y

    return run, bool(getattr(model.model[-1], "end2end", False))


def combine(paths):
    m = Metric()
    for p_ in paths:
        z = np.load(p_)
        for key in m.stats:
            m.stats[key].append(z[key])
    return m.result()


def main():
    if "--combine" in sys.argv:
        paths = sys.argv[sys.argv.index("--combine") + 1:]
        print(json.dumps({**combine(paths), "shards": len(paths)}))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["hailo", "fp"])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("c", nargs="?")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--out", default="")
    ap.add_argument("--shard", default="0/1", help="k/n: evaluate files[k::n]; with n>1 raw stats go to --stats")
    ap.add_argument("--stats", default="", help="write raw matching stats (npz) for later --combine")
    ap.add_argument("--combine", nargs="*", default=None, help="merge stats npz files and print mAP")
    ap.add_argument("--ints", default=None, help="QATNet.export_ints() checkpoint (torch.save) overriding the HAR ints")
    ap.add_argument("--apu", default=None, help="QATNet.export_apu() checkpoint: trained APU tables")
    ap.add_argument("--impl", default="qatfast", choices=["qatfast", "graph26"], help="exact executor for YOLO26")
    ap.add_argument("--pt", default=None, help="YOLO26 checkpoint whose Detect head decodes the Hailo outputs")
    args = ap.parse_args()
    dev = args.dev
    if args.kind == "hailo":
        import graph
        import graph26
        hn = json.load(open(args.a))
        e2e = args.pt is not None
        params = graph26.Params(args.b, dev) if e2e else graph.load_params(args.b, dev)
        ints = None
        if args.ints:
            ints = {k: (a.to(dev).double(), b.to(dev).double()) for k, (a, b) in torch.load(args.ints).items()}
        apu = torch.load(args.apu) if args.apu else None
        fn = y26_heads(hn, params, dev, args.pt, ints=ints, impl=args.impl, apu=apu) if e2e else hailo_heads(hn, params, dev)
        in_l = [v for v in hn["layers"].values() if v["type"] == "input_layer"][0]
        shape = tuple(in_l["input_shapes"][0][1:3])
    else:
        fn, e2e = fp_heads(args.a, dev)
        shape = (int(args.b), int(args.c))
    k, nsh = (int(v) for v in args.shard.split("/"))
    files = val_files(args.n)[k::nsh]
    met, t0, done = Metric(), time.time(), 0
    with torch.no_grad():
        for imgs, labs in batches(files, shape, args.bs):
            preds = fn(imgs)
            # no time cap: the pure-torch NMS fallback (torchvision::nms is broken in these envs) otherwise
            # silently truncates detections and collapses recall
            if e2e:  # YOLO26 one2one head: already top-k selected, no NMS; drop below conf like Ultralytics val
                dets = [p[p[:, 4] > args.conf] for p in preds]
            else:
                dets = non_max_suppression(preds, args.conf, args.iou, max_det=300, nc=NC, max_time_img=1e9)
            for d, l in zip(dets, labs):
                met.add(d, l, k + nsh * done)  # global val index: files = val_files[k::nsh]
                done += 1
            if done % 100 < args.bs:
                print(f"{done}/{len(files)} {done / (time.time() - t0):.2f} img/s", flush=True)
    if args.stats:
        np.savez(args.stats, **{key: np.concatenate(v, 0) for key, v in met.stats.items()})
        print("wrote stats", args.stats, flush=True)
        if nsh > 1:
            return
    res = met.result()
    res.update(kind=args.kind, model=args.a, shape=shape, n=len(files), sec=round(time.time() - t0))
    print(json.dumps(res))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

"""Per-layer output-rounding sensitivity on the exact executor (picks layers for 16-bit outputs).

A layer's int8 output carries rounding noise of variance 1/12 per element in its own integer units. To first order it
raises the QAT loss by E[g^2]/12 summed over the layer's elements, g = dLoss/d(integer output). A 16-bit output divides
that noise by ~65,000, so the score is also the expected gain from moving the layer to a8_w8_a16.

Usage: python sens.py --hn HN --cap CAP --pt PT --teacher_cache DIR --imgsz 640 [--n 256] [--ints CKPT] --out OUT.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph26  # noqa: E402
from qatfast import QATNet  # noqa: E402
from train_qat import ROOT, Pages, collate, dequant, head_names  # noqa: E402

QUANT_TYPES = {"conv", "dw", "normalization", "ew_add", "ew_sub", "activation", "matmul", "reduce_sum", "ew_mult"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hn", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--teacher_cache", required=True)
    ap.add_argument("--train", default=f"{ROOT}/data_v14sec/train/images")
    ap.add_argument("--imgsz", type=int, default=0)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ints", default=None, help="QATNet.export_ints() checkpoint to score instead of the HAR ints")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    hn = json.load(open(args.hn))
    P = graph26.Params(args.cap, dev)
    student = QATNet(hn, P, dev).to(dev)
    if args.ints:
        for k, (kk, bb) in torch.load(args.ints).items():
            m = student.mac[k.replace("/", "__")]
            m.set_ints(kk, bb)
    heads = head_names(hn)
    from types import SimpleNamespace

    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    tmodel = YOLO(args.pt).model.to(dev).float().eval()
    det = tmodel.model[-1]
    a = tmodel.args if isinstance(getattr(tmodel, "args", None), dict) else {}
    tmodel.args = SimpleNamespace(box=a.get("box", 7.5), cls=a.get("cls", 0.5), dfl=a.get("dfl", 1.5), epochs=40)
    crit = v8DetectionLoss(tmodel, tal_topk=7, tal_topk2=1)

    ds = Pages(args.train, args.n, seed=1, cache=args.teacher_cache, imgsz=args.imgsz)
    dl = torch.utils.data.DataLoader(ds, args.bs, shuffle=False, num_workers=4, collate_fn=collate)
    layers = hn["layers"]
    score, numel, seen = {}, {}, 0
    for imgs, batch in dl:
        imgs = imgs.to(dev).float()
        batch = {k: v.to(dev) for k, v in batch.items()}
        t1 = {"boxes": batch.pop("t_boxes"), "scores": batch.pop("t_scores")}
        out = student(imgs)
        keep = {n: y for n, y in out.items() if layers[n]["type"] in QUANT_TYPES and y.requires_grad}
        for y in keep.values():
            y.retain_grad()
        bs = imgs.shape[0]
        bx = [dequant(P, b, out[b], dev) for b, _ in heads]
        cl = [dequant(P, c, out[c], dev) for _, c in heads]
        s_boxes = torch.cat([t.reshape(bs, 4, -1) for t in bx], -1)
        s_scores = torch.cat([t.reshape(bs, det.nc, -1) for t in cl], -1)
        l_gt, _ = crit.loss({"boxes": s_boxes, "scores": s_scores, "feats": bx}, batch)
        l_gt = l_gt.sum() / bs if l_gt.dim() else l_gt
        t_prob = t1["scores"].sigmoid()
        l_cls = F.binary_cross_entropy_with_logits(s_scores, t_prob, reduction="mean") * det.nc
        w = t_prob.amax(1, keepdim=True)
        l_box = ((s_boxes - t1["boxes"]).abs() * w).sum() / (w.sum() * 4 + 1e-6)
        student.zero_grad(set_to_none=True)
        (l_gt + l_cls + l_box).backward()
        for n, y in keep.items():
            if y.grad is not None:  # per-image sum over elements of g^2 / 12
                score[n] = score.get(n, 0.0) + float(y.grad.double().pow(2).sum()) / 12.0
                numel[n] = y[0].numel()
        seen += bs
        print(f"{seen}/{len(ds)}", flush=True)
    a16 = {c for pair in heads for c in pair}
    rows = sorted(({"layer": n, "type": layers[n]["type"], "score": s / seen, "numel": numel[n], "a16": n in a16}
                   for n, s in score.items()), key=lambda r: -r["score"])
    tot = sum(r["score"] for r in rows if not r["a16"])
    json.dump({"n": seen, "total_int8": tot, "rows": rows}, open(args.out, "w"), indent=1)
    acc = 0.0
    for r in rows[:25]:
        if not r["a16"]:
            acc += r["score"]
        print(f"{r['layer']:40s} {r['type']:14s} {r['score']:.4e}  cum {acc / tot:.2%}{'  (a16)' if r['a16'] else ''}")


if __name__ == "__main__":
    main()

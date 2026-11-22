"""Bottleneck harness for the QAT network on one GPU (llama.cpp-style: measure, patch one thing, re-verify).

  exact : every layer of the student forward == graph26 exact executor (the SDK-verified reference), on real pages,
          with the level-0 ints and with trained ints (--ints) so different kernels/overflow patterns are exercised.
  grads : gradients of a fixed loss, saved (--save) or compared (--cmp) against a saved baseline.
  time  : s/step for train (fwd+bwd) and eval (no_grad fwd), peak memory.
  prof  : torch.profiler CUDA-time table for one train step.

Usage: python bench_qat.py MODE [--impl qatnet] [--bs 2] [--n 4] [--ints PT] [--save F | --cmp F]
"""
import argparse
import glob
import importlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph26  # noqa: E402
from train_qat import head_names  # noqa: E402

ROOT = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"


SHAPE = None  # (H, W) of the hn input: pages are letterboxed to it when it differs from the stored size


def load_imgs(n, off=0):
    fs = sorted(glob.glob(f"{ROOT}/data_v14sec/val/images/*"))[off:off + n]
    ims = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in fs]
    if SHAPE is not None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from letterbox import letterbox
        ims = [im if im.shape[:2] == SHAPE else letterbox(im, SHAPE)[0] for im in ims]
    return np.stack(ims).astype(np.float32)


def set_ints(net, ints):
    for k, m in net.mac.items():
        kk, bb = ints[k.replace("__", "/")]
        m.set_ints(kk, bb)


def fixed_loss(out, heads, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    tot = 0.0
    for b, c in heads:
        for n in (b, c):
            w = torch.randn(out[n].shape[1:], generator=g).to(out[n].device)
            tot = tot + (out[n] * w).mean()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["exact", "grads", "time", "prof", "apufd"])
    ap.add_argument("--impl", default="qatnet")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--ints", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--cmp", default="")
    ap.add_argument("--cudnn_bench", action="store_true")
    ap.add_argument("--apu_train", action="store_true", help="enable trainable APU tables (qatfast)")
    ap.add_argument("--hn", default=f"{ROOT}/yolo26m_v14sec_L0.hn")
    ap.add_argument("--cap", default=f"{ROOT}/capture_y26g")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = a.cudnn_bench
    dev = torch.device("cuda")
    hn = json.load(open(a.hn))
    global SHAPE
    SHAPE = tuple([v for v in hn["layers"].values() if v["type"] == "input_layer"][0]["input_shapes"][0][1:3])
    P = graph26.Params(a.cap, dev)
    mod = importlib.import_module(a.impl)
    net = mod.QATNet(hn, P, dev).to(dev)
    if a.apu_train:
        net.enable_apu_training()
    ints = None
    if a.ints:
        ints = {k: (x.to(dev), y.to(dev)) for k, (x, y) in torch.load(a.ints).items()}
        set_ints(net, ints)
    heads = head_names(hn)

    if a.mode == "exact":
        bad_layers, tot_layers = set(), 0
        for off in range(0, a.n, a.bs):
            imgs = load_imgs(a.bs, off)
            with torch.no_grad():
                ref, _ = graph26.run(hn, P, imgs.astype(np.float64), dev, check=False, ints=ints)
                got = net(torch.from_numpy(imgs).to(dev))
            for k, y in got.items():
                if k not in ref:
                    continue
                tot_layers += 1
                d = int((y.double() != ref[k]).sum())
                if d:
                    bad_layers.add(k)
                    print(f"MISMATCH img{off} {k}: {d}/{y.numel()} max {float((y.double() - ref[k]).abs().max()):.0f}")
            del ref, got
        print(f"EXACT {tot_layers - len(bad_layers) * 0}/{tot_layers} layer-checks, {len(bad_layers)} bad layers "
              f"({a.n} imgs, ints={'trained' if ints else 'L0'})")
        sys.exit(1 if bad_layers else 0)

    imgs = torch.from_numpy(load_imgs(a.bs)).to(dev)
    if a.mode == "apufd":  # APU-table gradient vs finite difference on chosen layers (+dm mantissa / +do offset)
        net.enable_apu_training()
        mods = net.apu_modules()
        out = net(imgs)
        fixed_loss(out, heads).backward()
        for key in ("best__conv71", "best__conv74", "best__conv100", "best__conv5"):
            apu = mods[key]
            for what, lat, d in (("mant", apu.mant_lat, 4.0), ("offs", apu.offs_lat, 64.0)):
                seg = int(lat.grad.abs().argmax())
                g = float(lat.grad[seg])
                with torch.no_grad():
                    base = float(fixed_loss(net(imgs), heads))
                    lat[seg] += d
                    up = float(fixed_loss(net(imgs), heads))
                    lat[seg] -= d
                print(f"APUFD {key} {what}[{seg}] grad*d {g * d:+.3e}  finite diff {up - base:+.3e}")
        return
    if a.mode == "grads":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        out = net(imgs)
        fixed_loss(out, heads).backward()
        g = {k: (m.kernel.grad.detach().cpu(), m.bias.grad.detach().cpu()) for k, m in net.mac.items()}
        if a.save:
            torch.save(g, a.save)
            print("saved", a.save)
        if a.cmp:
            ref = torch.load(a.cmp)
            rels, num, den = [], 0.0, 0.0
            for k, (gk, gb) in g.items():
                for x, y in ((gk, ref[k][0]), (gb, ref[k][1])):
                    rels.append((float((x - y).norm() / (y.norm() + 1e-30)), k, float(y.norm())))
                    num += float((x - y).norm()) ** 2
                    den += float(y.norm()) ** 2
            rels.sort()
            med = rels[len(rels) // 2][0]
            print(f"GRADS global rel L2 {(num / den) ** 0.5:.3g}  median {med:.3g}  worst {rels[-1][0]:.3g} "
                  f"({rels[-1][1]}, |g|={rels[-1][2]:.3g})  n>1e-3: {sum(r > 1e-3 for r, _, _ in rels)}/{len(rels)}")
        return

    def train_step():
        net.zero_grad(set_to_none=True)
        out = net(imgs)
        fixed_loss(out, heads).backward()

    def eval_step():
        with torch.no_grad():
            net(imgs)

    if a.mode == "prof":
        train_step()
        torch.cuda.synchronize()
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            train_step()
            torch.cuda.synchronize()
        print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=60))
        return

    for name, fn in (("train", train_step), ("eval", eval_step)):
        fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        for _ in range(a.iters):
            fn()
        torch.cuda.synchronize()
        dt = (time.time() - t) / a.iters
        if getattr(net, "flag_events", None):
            print("FLAG EVENTS", net.flag_events)
        print(f"TIME {name} bs{a.bs}: {dt:.3f} s/step  {a.bs / dt:.2f} img/s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()

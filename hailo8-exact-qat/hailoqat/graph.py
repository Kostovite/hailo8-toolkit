"""Run a Hailo .hn network end-to-end with hailoqat's bit-exact integer ops.

Usage: python graph.py HN_PATH PARAM_DIR IMG [device]
  PARAM_DIR holds per-layer capture records (capture.py); every captured layer output is checked as the graph runs,
  each layer fed only by this executor's own upstream results (no reference tensors are injected).
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from intops import conv_layer, ew_add_layer


def load_params(param_dir, dev):
    params = {}
    for f in glob.glob(os.path.join(param_dir, "*.npz")):
        r = np.load(f)
        params[os.path.basename(f)[:-4].replace("__", "/")] = r
    return params


def to_t(r, dev, skip=("conv_in", "conv_out", "bias_in", "bias_out", "act_in", "act_out", "ew_in0", "ew_in1", "ew_out")):
    return {k: torch.as_tensor(r[k]).to(dev) for k in r.files if r[k].dtype.kind != "U" and k not in skip}


def conv_params(r, dev):
    p = to_t(r, dev)
    if bool(r["depthwise"]):
        # HWIO (kh, kw, Cin, mult) -> (Cin*mult, 1, kh, kw), one group per input channel
        kh, kw, ci, mult = r["k"].shape
        p["k"] = p["k"].permute(2, 3, 0, 1).reshape(ci * mult, 1, kh, kw)
        p["groups"] = torch.tensor(ci)
    else:
        p["k"] = p["k"].permute(3, 2, 0, 1)
    return p


def maxpool(x, layer):
    prm = layer["params"]
    kh, kw = prm["kernel_shape"][1:3]
    sh, sw = prm["strides"][1:3]
    # SAME padding; inputs are unsigned so a 0 pad never wins the max
    ph, pw = (kh - 1), (kw - 1)
    x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=0.0)
    return F.max_pool2d(x, (kh, kw), (sh, sw))


def run(hn, params, img, dev, check=True):
    layers = hn["layers"]
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for i in layers[n]["input"]:
            visit(i)
        order.append(n)

    for n in layers:
        visit(n)

    out, report = {}, []
    for n in order:
        L = layers[n]
        t = L["type"]
        if t in ("postprocess", "output_layer"):
            continue
        ins = [out[i] for i in L["input"]]
        if t == "input_layer":
            y = torch.as_tensor(img, dtype=torch.float64, device=dev).permute(0, 3, 1, 2)
        elif t in ("conv", "normalization"):
            y = conv_layer(ins[0], conv_params(params[n], dev))[2]
        elif t == "ew_add":
            y = ew_add_layer(ins[0], ins[1], to_t(params[n], dev))[2]
        elif t == "concat":
            y = torch.cat(ins, dim=1)
        elif t == "maxpool":
            y = maxpool(ins[0], L)
        elif t == "resize":
            prm = L["params"]
            y = ins[0].repeat_interleave(int(prm["resize_h_ratio_list"][0]), 2)
            y = y.repeat_interleave(int(prm["resize_w_ratio_list"][0]), 3)
        else:
            raise NotImplementedError(f"{n}: layer type {t}")
        out[n] = y
        if check and n in params and "act_out" in params[n].files:
            ref = torch.as_tensor(params[n]["act_out"]).to(dev).permute(0, 3, 1, 2).to(y.dtype)
            bad = int((y != ref).sum())
            report.append((n, t, bad, ref.numel()))
            if bad:
                print(f"MISMATCH {n} ({t}): {bad}/{ref.numel()}", flush=True)
    return out, report


if __name__ == "__main__":
    hn_path, param_dir, img_path = sys.argv[1:4]
    dev = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    hn = json.load(open(hn_path))
    params = load_params(param_dir, dev)
    img = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (896, 896))[None].astype(np.float64)
    with torch.no_grad():
        out, report = run(hn, params, img, dev)
    exact = sum(b == 0 for _, _, b, _ in report)
    print(f"{exact}/{len(report)} layers exact when chained from raw pixels")
    heads = hn["layers"][[k for k, v in hn["layers"].items() if v["type"] == "postprocess"][0]]["input"]
    for h in heads:
        r = [x for x in report if x[0] == h][0]
        print(f"  head {h}: {'EXACT' if r[2] == 0 else f'{r[2]} differ'}")

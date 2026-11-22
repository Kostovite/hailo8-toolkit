"""qatfast (training emulator) vs graph26 (verified exact executor), layer by layer, on one captured input; then one
backward pass to confirm every trainable part receives a gradient.

Usage: python qat_check.py HN CAPTURE_DIR IMG [cuda|cpu]
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

import graph26 as G
import qatfast as Q

hn_path, cap, img_path = sys.argv[1:4]
dev = sys.argv[4] if len(sys.argv) > 4 else "cuda"
hn = json.load(open(hn_path))
in_l = [v for v in hn["layers"].values() if v["type"] == "input_layer"][0]
h, w = in_l["input_shapes"][0][1:3]
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
if img.shape[:2] != (h, w):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from letterbox import letterbox
    img = letterbox(img, (h, w))[0]
x = img[None].astype(np.float64)

P = G.Params(cap, "cpu")
with torch.no_grad():
    ref, report = G.run(hn, P, x, "cpu", check=False, cont=True)
P2 = G.Params(cap, dev)
net = Q.QATNet(hn, P2, dev)
print(f"QATNet: {len(net.mac)} MAC modules, {len(net.apus)} APUs, {len(net.chains)} atomic chains")
with torch.no_grad():
    out = net(torch.as_tensor(x, dtype=torch.float32, device=dev))
ok = bad = miss = 0
by_t = {}
for n, L in hn["layers"].items():
    if L["type"] in ("output_layer", "postprocess") or n not in ref or n not in out:
        miss += n in ref and L["type"] not in ("output_layer", "postprocess")
        continue
    a, b = ref[n], out[n]
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    good = all(x_.shape == y_.shape and bool((x_.cpu().double() == y_.cpu().double()).all()) for x_, y_ in zip(a, b))
    by_t.setdefault(L["type"], [0, 0])
    by_t[L["type"]][0 if good else 1] += 1
    if good:
        ok += 1
    else:
        bad += 1
        if bad <= 5:
            d = [(float((x_.cpu().double() - y_.cpu().double()).abs().max()) if x_.shape == y_.shape else f"{tuple(x_.shape)} vs {tuple(y_.shape)}") for x_, y_ in zip(a, b)]
            print(f"  DIFF {n} ({L['type']}): {d}")
print(f"qatfast == graph26 on {ok}/{ok + bad} layers ({miss} not produced by qatfast)")
print({t: f"{g}/{g + b_}" for t, (g, b_) in by_t.items()})
# gradient check
net.enable_apu_training() if hasattr(net, "enable_apu_training") else None
outs = net(torch.as_tensor(x, dtype=torch.float32, device=dev))
last = [n for n, L in hn["layers"].items() if L["type"] in ("output_layer", "postprocess")]
srcs = {i for o in last for i in hn["layers"][o]["input"] if i in outs and hn["layers"][i]["type"] != "postprocess"}
first = lambda v: v[0] if isinstance(v, list) else v  # noqa: E731
loss = sum(first(outs[i]).float().mean() for i in srcs)
loss.backward()
with_g = sum(1 for p in net.parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0)
print(f"backward ok: {with_g}/{sum(1 for _ in net.parameters())} parameter tensors received a non-zero gradient")

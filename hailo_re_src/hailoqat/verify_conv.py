"""Check intops.conv_layer against captured Hailo bit-exact tensors, stage by stage.

Usage: python verify_conv.py CAPTURE_DIR [device]
"""
import glob
import os
import sys

import numpy as np
import torch

from graph import conv_params
from intops import conv_add_layer, conv_layer, ew_add_layer

cap_dir = sys.argv[1]
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
STAGES = [("accumulator", "conv_out"), ("bias", "bias_out"), ("apu/output", "act_out")]
n_ok = n_all = 0
for f in sorted(glob.glob(os.path.join(cap_dir, "*.npz"))):
    r = np.load(f)
    is_ew = "ew_out" in r.files
    if not is_ew and "act_out" not in r.files:
        print(f"SKIP {os.path.basename(f)} (no act)")
        continue
    t = lambda k: torch.as_tensor(r[k]).to(dev)
    p = {k: t(k) for k in r.files if r[k].dtype.kind != "U"}
    nchw = lambda k: p[k].permute(0, 3, 1, 2)
    if is_ew:
        stages = [("ew_acc", "ew_out"), ("bias", "bias_out"), ("apu/output", "act_out")]
        outs = ew_add_layer(nchw("ew_in0"), nchw("ew_in1"), p)
    else:
        stages = STAGES
        p = {**p, **conv_params(r, dev)}  # HWIO -> OIHW, depthwise -> grouped
        if "ca_l3" in r.files:  # HailoConvAdd: residual folded into the accumulator before the bias
            stages = [("accumulator", "conv_out"), ("bias_in(+res)", "bias_in"), ("bias", "bias_out"), ("apu/output", "act_out")]
            outs = conv_add_layer(nchw("conv_in"), nchw("ca_l3"), p)
        else:
            outs = conv_layer(nchw("conv_in"), p)
    msgs, ok = [], True
    for (name, key), got in zip(stages, outs):
        ref = p[key].permute(0, 3, 1, 2).to(got.dtype)
        bad = int((got != ref).sum())
        ok &= bad == 0
        msgs.append(f"{name}: {'EXACT' if bad == 0 else f'{bad}/{ref.numel()} differ (max {float((got - ref).abs().max()):.0f})'}")
    n_all += 1
    n_ok += ok
    print(f"{'OK ' if ok else 'BAD'} {os.path.basename(f)[:-4]:40s} s={int(r['ew_shift'] if is_ew else r['pre_acc_shift'])} {r['act_name']:8s} | "
          + " | ".join(msgs), flush=True)
print(f"\n{n_ok}/{n_all} layers bit-exact at every stage")

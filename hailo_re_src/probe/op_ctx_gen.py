"""Boundary pseudo-models for layer types the sweep families can't express (torch-exported, as a user gets them):
2 3x3 convs -> OP -> 2 3x3 convs, OUT_DIR/op_<name>/m.onnx, for bnd.py as `@DIR` cases.

Usage (general env, torch): python op_ctx_gen.py OUT_DIR [NAME ...]
  NAME = a key of OPS (C=64, 20x20) or T_<case> = zoo/ops/torch_cases.py CASES[<case>] at the op survey's own size
  (C=16, 24x24, where e.g. LayerNorm / softmax / ViT fuse into single layer_normalization / softmax layers)
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../zoo/ops"))
import torch_cases as tc  # noqa: E402

C, H = 64, 20


class Fn(nn.Module):
    def __init__(self, fn, *mods):
        super().__init__()
        self.fn, self.mods = fn, nn.ModuleList(mods)

    def forward(self, x):
        return self.fn(x, *self.mods)


def cbs(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, 1, 1), nn.SiLU())


def gn1_hw_first(x):  # spatial mean per channel first, then over channels (same value: equal group sizes)
    m = x.mean((2, 3), keepdim=True).mean(1, keepdim=True)
    d = x - m
    return d * torch.rsqrt((d * d).mean((2, 3), keepdim=True).mean(1, keepdim=True) + 1e-5)


def dup_conv(ci, co):  # 1x1 averaging conv: every output channel = mean of the inputs (frozen)
    c = nn.Conv2d(ci, co, 1, bias=False)
    c.weight.data.fill_(1.0 / ci)
    return c


def gn1_dup(x, a, b):  # whole-tensor stats computed on 4 identical channels (the GroupNorm(4) shape that works)
    m = b(a(x).mean((2, 3), keepdim=True))  # [1,4,1,1] -> back to [1,C,1,1]
    d = x - m
    v = b(a(d * d).mean((2, 3), keepdim=True))
    return d * torch.rsqrt(v + 1e-5)


def gn1_gap(x):  # GlobalAveragePool form of the same statistics
    g = nn.functional.adaptive_avg_pool2d
    m = g(x, 1).mean(1, keepdim=True)
    d = x - m
    return d * torch.rsqrt(g(d * d, 1).mean(1, keepdim=True) + 1e-5)


OPS = {  # name: (module, output channels)
    "instnorm": (nn.InstanceNorm2d(C, affine=True), C),
    "groupnorm": (nn.GroupNorm(4, C), C),
    "groupnorm1": (nn.GroupNorm(1, C), C),
    "pow2": (Fn(lambda x: x ** 2), C),
    "unshuffle": (nn.PixelUnshuffle(2), 4 * C),
    "argmax": (Fn(lambda x: x.argmax(1, keepdim=True).float()), 1),
    "gap": (Fn(lambda x, l: l(x.mean((2, 3))).reshape(1, C, 1, 1), nn.Linear(C, C)), C),
    "mean_c": (Fn(lambda x: x.mean(1, keepdim=True)), 1),
    # dense workaround: "No format for fc1 -> conv3" (Linear feeding a conv) -> the same Linear as a 1x1 conv
    "gap_conv": (Fn(lambda x, l: l(x.mean((2, 3), keepdim=True)), nn.Conv2d(C, C, 1)), C),
    "gap_fc_bcast": (Fn(lambda x, l: x * l(x.mean((2, 3))).reshape(1, C, 1, 1), nn.Linear(C, C)), C),
    # GroupNorm(1) silent-wrong hunt (RULES 18): which piece breaks, and exact rewrites that avoid it
    "gm_sub": (Fn(lambda x: x - x.mean((1, 2, 3), keepdim=True)), C),
    "gm_rms": (Fn(lambda x: x * torch.rsqrt(x.pow(2).mean((1, 2, 3), keepdim=True) + 1e-5)), C),
    "gm_mean_c": (Fn(lambda x: x - x.mean(1, keepdim=True)), C),
    "gm_mean_hw": (Fn(lambda x: x - x.mean((2, 3), keepdim=True)), C),
    "gn1_hw_first": (Fn(lambda x: gn1_hw_first(x)), C),
    "gn1_dup": (Fn(lambda x, a, b: gn1_dup(x, a, b), dup_conv(C, 4), dup_conv(4, C)), C),
    "gn1_gap": (Fn(lambda x: gn1_gap(x)), C),
}
TC_C, TC_H = tc.C, tc.H  # the survey's modules were built at import with this width


class Net(nn.Module):
    def __init__(self, op, co, c=C):
        super().__init__()
        self.pre, self.op, self.post = nn.Sequential(cbs(c, c), cbs(c, c)), op, nn.Sequential(cbs(co, co), cbs(co, co))

    def forward(self, x):
        return self.post(self.op(self.pre(x)))


if __name__ == "__main__":
    for nm in sys.argv[2:] or OPS:
        torch.manual_seed(0)
        if nm.startswith("T_"):
            case = tc.CASES[nm[2:]]
            op, co, c, h = case[0], (case[2] if len(case) > 2 else TC_C), TC_C, TC_H
        else:
            (op, co), c, h = OPS[nm], C, H
        d = f"{sys.argv[1]}/op_{nm}"
        os.makedirs(d, exist_ok=True)
        torch.onnx.export(Net(op, co, c).eval(), torch.randn(1, c, h, h), f"{d}/m.onnx", input_names=["x"],
                          output_names=["y"], opset_version=17, dynamo=False)
        print("@" + os.path.abspath(d))

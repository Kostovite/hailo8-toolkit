"""Random pseudo-models for crash hunting: a chain of 4-7 random blocks (shape-preserving, [1,C,H,W] in and out),
exported by torch as a user would get them. fuzz.py compiles each with stock and with the own pipeline at random cuts.

Usage (general env, torch): python fuzz_gen.py OUT_DIR SEED0 N   -> OUT_DIR/fz_<seed>/m.onnx, prints @DIR lines
"""
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def cbs(ci, co, k=3, s=1, g=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, s, k // 2, groups=g), nn.SiLU())


class Res(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class Fn(nn.Module):
    def __init__(self, fn, *mods):
        super().__init__()
        self.fn, self.mods = fn, nn.ModuleList(mods)

    def forward(self, x):
        return self.fn(x, *self.mods)


# yolo PSA attention (4-D view + split). The 5-D reshape(b,3,heads,d,n).unbind(1) form fails stock parsing
# (UnsupportedShuffleLayerError in the Reshape): fz1 fz_1001/1004/1005
from attn_gen import Attention  # noqa: E402


def block(name, c, h):
    return {
        "conv3": lambda: cbs(c, c),
        "conv1": lambda: cbs(c, c, 1),
        "conv5": lambda: cbs(c, c, 5),
        "dw3": lambda: cbs(c, c, 3, 1, c),
        "dw5": lambda: cbs(c, c, 5, 1, c),
        "dw7": lambda: cbs(c, c, 7, 1, c),
        "res": lambda: Res(nn.Sequential(cbs(c, c), cbs(c, c))),
        "gate": lambda: Fn(lambda x, a: x * torch.sigmoid(a(x)), nn.Conv2d(c, c, 1)),
        "cat": lambda: Fn(lambda x, a, b: b(torch.cat([x, a(x)], 1)), cbs(c, c), cbs(2 * c, c, 1)),
        "c2f": lambda: Fn(lambda x, a, b, d: d(torch.cat([*a(x).chunk(2, 1), b(a(x).chunk(2, 1)[1])], 1)),
                          cbs(c, c, 1), cbs(c // 2, c // 2), cbs(c + c // 2, c, 1)),
        "maxpool": lambda: Fn(lambda x: F.max_pool2d(x, 5, 1, 2)),
        "sppf": lambda: Fn(lambda x, a: a(torch.cat([x, F.max_pool2d(x, 5, 1, 2),
                                                     F.max_pool2d(F.max_pool2d(x, 5, 1, 2), 5, 1, 2)], 1)),
                           cbs(3 * c, c, 1)),
        "avgpool": lambda: Fn(lambda x: F.avg_pool2d(x, 3, 1, 1)),
        "down_up": lambda: Fn(lambda x, a: F.interpolate(a(x), scale_factor=2, mode="nearest"), cbs(c, c, 3, 2)),
        "up_down": lambda: Fn(lambda x, a: a(F.interpolate(x, scale_factor=2, mode="nearest")), cbs(c, c, 3, 2)),
        "bilinear": lambda: Fn(lambda x, a: F.interpolate(a(x), scale_factor=2, mode="bilinear",
                                                          align_corners=False), cbs(c, c, 3, 2)),
        "fpn": lambda: Fn(lambda x, a, b: b(torch.cat([x, F.interpolate(a(x), scale_factor=2)], 1)),
                          cbs(c, c, 3, 2), cbs(2 * c, c, 1)),
        "convt": lambda: Fn(lambda x, a, b: b(a(x)), cbs(c, c, 3, 2), nn.ConvTranspose2d(c, c, 2, 2)),
        "shuffle": lambda: Fn(lambda x: F.pixel_unshuffle(F.pixel_shuffle(x, 2), 2)) if c % 4 == 0 else cbs(c, c),
        "d2s": lambda: Fn(lambda x, a, b: b(F.pixel_shuffle(a(x), 2)), cbs(c, 4 * c, 1), cbs(c, c, 3, 2)),
        "layernorm": lambda: Fn(lambda x, n: n(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), nn.LayerNorm(c)),
        "groupnorm": lambda: nn.GroupNorm(4, c) if c % 4 == 0 else cbs(c, c),
        "softmax_c": lambda: Fn(lambda x: x.softmax(1)),
        "attn": lambda: Res(Attention(c, 2 if c % 4 == 0 else 1)),
        "split2": lambda: Fn(lambda x, a, b: torch.cat([a(x[:, :c // 2]), b(x[:, c // 2:])], 1),
                             cbs(c // 2, c // 2), cbs(c - c // 2, c - c // 2, 1)),
        "sigmoid": lambda: nn.Sigmoid(),
        "hswish": lambda: nn.Hardswish(),
        "gelu": lambda: nn.GELU(),
        "se": lambda: Fn(lambda x, a, b: x * torch.sigmoid(b(F.relu(a(x.mean((2, 3), keepdim=True))))),
                         nn.Conv2d(c, max(1, c // 4), 1), nn.Conv2d(max(1, c // 4), c, 1)),
        "mul2": lambda: Fn(lambda x, a, b: a(x) * b(x), cbs(c, c, 1), cbs(c, c)),
        "sub": lambda: Fn(lambda x, a: x - a(x), cbs(c, c)),
        "pad_conv": lambda: Fn(lambda x, a: a(F.pad(x, (1, 1, 1, 1))), nn.Conv2d(c, c, 3)),
    }[name]()


NAMES = ["conv3", "conv1", "conv5", "dw3", "dw5", "dw7", "res", "gate", "cat", "c2f", "maxpool", "sppf", "avgpool",
         "down_up", "up_down", "bilinear", "fpn", "convt", "shuffle", "d2s", "layernorm", "groupnorm", "softmax_c",
         "attn", "split2", "sigmoid", "hswish", "gelu", "se", "mul2", "sub", "pad_conv"]

if __name__ == "__main__":
    out, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    for seed in range(s0, s0 + n):
        r = random.Random(seed)
        c, h = r.choice([16, 32, 48, 64, 96, 128]), r.choice([16, 20, 24, 32, 40])
        if c >= 96 and h >= 32:
            h = 20
        names = [r.choice(NAMES) for _ in range(r.randint(4, 7))]
        if "attn" in names and h * h > 576:  # attention tokens beyond the single-context range
            h = 20
        torch.manual_seed(seed)
        m = nn.Sequential(cbs(c, c), *[block(b, c, h) for b in names], cbs(c, c)).eval()
        d = f"{out}/fz_{seed}"
        os.makedirs(d, exist_ok=True)
        open(f"{d}/spec.txt", "w").write(f"c={c} h={h} " + " ".join(names) + "\n")
        torch.onnx.export(m, torch.randn(1, c, h, h), f"{d}/m.onnx", input_names=["x"], output_names=["y"],
                          opset_version=17, dynamo=False)
        print("@" + os.path.abspath(d))

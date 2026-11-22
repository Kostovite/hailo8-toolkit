"""Pseudo-models for the multi-context attention crash (RULES.md 15): e pre convs -> PSA (attention + FFN, both
residual) -> p post 3x3 convs [-> optional depthwise tail]. Cut into contexts before/after the PSA by mapat.py.

Usage (general env, torch): python attn_ctx_gen.py OUT_DIR c,h,heads,e,p,dw [...]
  -> OUT_DIR/ac_c{c}_h{h}_g{heads}_e{e}_p{p}_dw{dw}/m.onnx
"""
import os
import sys

import torch
import torch.nn as nn

from attn_gen import Attention


def cbs(ci, co, k=3, g=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, 1, k // 2, groups=g), nn.SiLU())


class Net(nn.Module):
    def __init__(self, c, heads, e, p, dw, n=1):
        super().__init__()
        self.pre = nn.Sequential(*[cbs(c, c) for _ in range(e)])
        self.attn = Attention(c, heads)
        self.ffn = nn.Sequential(cbs(c, 2 * c, 1), nn.Conv2d(2 * c, c, 1))
        self.n = n
        if n == 2:  # a second PSA behind 2 convs (yolo26m has two attention contexts)
            self.mid = nn.Sequential(cbs(c, c), cbs(c, c))
            self.attn2 = Attention(c, heads)
            self.ffn2 = nn.Sequential(cbs(c, 2 * c, 1), nn.Conv2d(2 * c, c, 1))
        self.post = nn.Sequential(*[cbs(c, c) for _ in range(p)])
        self.tail = nn.Sequential(*[cbs(c, c, 3, c) for _ in range(dw)])

    def forward(self, x):
        y = self.pre(x)
        y = y + self.attn(y)
        y = y + self.ffn(y)
        if self.n == 2:
            y = self.mid(y)
            y = y + self.attn2(y)
            y = y + self.ffn2(y)
        return self.tail(self.post(y))


if __name__ == "__main__":
    for spec in sys.argv[2:]:
        c, h, g, e, p, dw, n = ([int(x) for x in spec.split(",")] + [1])[:7]
        d = f"{sys.argv[1]}/ac_c{c}_h{h}_g{g}_e{e}_p{p}_dw{dw}" + (f"_n{n}" if n > 1 else "")
        os.makedirs(d, exist_ok=True)
        torch.manual_seed(0)
        m = Net(c, g, e, p, dw, n).eval()
        torch.onnx.export(m, torch.randn(1, c, h, h), f"{d}/m.onnx", input_names=["x"], output_names=["y"],
                          opset_version=17, dynamo=False)
        print(d)

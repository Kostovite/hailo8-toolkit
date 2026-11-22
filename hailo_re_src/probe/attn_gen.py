"""Pseudo-models of the Ultralytics PSA Attention block (the block yolo26m's attention contexts fail on), exported
to ONNX at any size: x(1,c,h,w) -> e convs -> Attention(c, heads, attn_ratio) -> 1x1 conv.

  e = 0: the block is fed straight from the network input (what a context-entry sub-network sees)
  e > 0: e 3x3 convs in front (the block embedded inside a context)

Usage (general env, torch): python attn_gen.py OUT_DIR c,h,heads,e [c,h,heads,e ...]
  -> OUT_DIR/attn_c{c}_h{h}_g{heads}_e{e}/m.onnx ; sweep.py runs them as cases "@OUT_DIR/attn_..."
"""
import os
import sys

import torch
import torch.nn as nn


class Attention(nn.Module):  # ultralytics.nn.modules.block.Attention (Conv = conv + BN, act=False -> folded)
    def __init__(self, dim, num_heads, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        self.qkv = nn.Conv2d(dim, dim + nh_kd * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.pe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        q, k, v = self.qkv(x).view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class Net(nn.Module):
    def __init__(self, c, heads, e):
        super().__init__()
        self.pre = nn.Sequential(*[nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.SiLU()) for _ in range(e)])
        self.attn = Attention(c, heads)
        self.post = nn.Conv2d(c, c, 1)

    def forward(self, x):
        y = self.pre(x)
        return self.post(y + self.attn(y))  # PSA: residual around the attention


def export(out, c, h, heads, e):
    d = f"{out}/attn_c{c}_h{h}_g{heads}_e{e}"
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(0)
    m = Net(c, heads, e).eval()
    torch.onnx.export(m, torch.randn(1, c, h, h), f"{d}/m.onnx", input_names=["x"], output_names=["y"],
                      opset_version=17, dynamo=False)
    return d


if __name__ == "__main__":
    for spec in sys.argv[2:]:
        c, h, g, e = (int(x) for x in spec.split(","))
        print(export(sys.argv[1], c, h, g, e))

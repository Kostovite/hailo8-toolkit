"""Micro-models of one YOLO26 C2PSA block (model.10) for fast Hailo layout experiments.

V0: the original block.
V1: token-tail. pe(v) is computed in spatial layout from a V-only 1x1 conv and converted INTO token layout
    (B, C, 1, N); proj, the attention residual and the ffn run on (B, C, 1, N) (all pointwise); one token->spatial
    reshape at the end feeds only concat + 1x1 cv2. Identical math.

Usage: python psa_micro.py PT OUT_DIR
"""
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from ultralytics import YOLO


class TokenTail(nn.Module):
    def __init__(self, c2psa):
        super().__init__()
        self.b = c2psa
        blk = c2psa.m[0]
        at = blk.attn
        per = at.key_dim * 2 + at.head_dim
        rows = torch.cat([torch.arange(h * per + 2 * at.key_dim, h * per + 2 * at.key_dim + at.head_dim)
                          for h in range(at.num_heads)])
        w, b = fuse(at.qkv)
        self.pe_v = nn.Conv2d(w.shape[1], len(rows), 1)
        self.pe_v.weight.data, self.pe_v.bias.data = w[rows].clone(), b[rows].clone()

    def forward(self, x):
        c = self.b
        blk = c.m[0]
        at = blk.attn
        a, t = c.cv1(x).split((c.c, c.c), dim=1)
        B, C, H, W = t.shape
        N = H * W
        qkv = at.qkv(t)
        q, k, v = qkv.view(B, at.num_heads, at.key_dim * 2 + at.head_dim, N).split(
            [at.key_dim, at.key_dim, at.head_dim], dim=2)
        attn = ((q * at.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        o = (v @ attn.transpose(-2, -1)).reshape(B, C, 1, N)            # stays in token layout
        pe = at.pe(self.pe_v(t)).reshape(B, C, 1, N)                     # spatial 3x3, then INTO tokens
        y = t.reshape(B, C, 1, N) + at.proj(o + pe)
        y = y + blk.ffn(y)
        return c.cv2(torch.cat((a, y.reshape(B, C, H, W)), 1))


def fuse(conv):
    w = conv.conv.weight
    bn = conv.bn
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return (w * s[:, None, None, None]).detach(), (bn.bias - bn.running_mean * s).detach()


def export(mod, path, x):
    torch.onnx.export(mod, x, path, input_names=["x"], output_names=["y"], opset_version=13, do_constant_folding=True)


def main():
    pt, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    model = YOLO(pt).model.float().eval()
    c2 = model.model[10]
    x = torch.randn(1, c2.cv1.conv.in_channels, 28, 28)
    with torch.no_grad():
        export(c2, f"{out}/v0.onnx", x)
        export(TokenTail(c2).eval(), f"{out}/v1.onnx", x)
    ref = ort.InferenceSession(f"{out}/v0.onnx").run(None, {"x": x.numpy()})[0]
    for v in ("v1",):
        y = ort.InferenceSession(f"{out}/{v}.onnx").run(None, {"x": x.numpy()})[0]
        print(v, "max rel diff vs v0", float(np.abs(y - ref).max() / np.abs(ref).max()))


if __name__ == "__main__":
    main()

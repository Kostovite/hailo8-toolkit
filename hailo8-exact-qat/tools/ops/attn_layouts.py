"""Which attention tensor layouts does the DFC parser accept? Torch modules exported like torch_cases.py (x -> f ->
identity conv), probed with probe.py run.

Usage (general env): python attn_layouts.py OUT_DIR   -> T_<name>/m.onnx + dom.txt, prints names
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_cases import C, H, W, L

N = H * W


class Proj(nn.Module):
    def __init__(self, heads, sdpa=False, layout="bhnd", scale_first=False):
        super().__init__()
        self.h, self.sdpa, self.layout, self.scale_first = heads, sdpa, layout, scale_first
        self.qkv, self.proj = nn.Conv2d(C, 3 * C, 1), nn.Conv2d(C, C, 1)

    def forward(self, x):
        b, c, hh, ww = x.shape
        h, d = self.h, c // self.h
        q, k, v = self.qkv(x).chunk(3, 1)  # each (b, c, H, W)
        if self.layout == "bhnd":  # (b, h, N, d) like timm / HF
            f = lambda t: t.reshape(b, h, d, hh * ww).transpose(-1, -2)  # noqa: E731
            q, k, v = f(q), f(k), f(v)
            if self.sdpa:
                o = F.scaled_dot_product_attention(q, k, v)
            else:
                qs = q * d ** -0.5 if self.scale_first else q
                a = qs @ k.transpose(-1, -2)
                a = (a if self.scale_first else a * d ** -0.5).softmax(-1)
                o = a @ v
            o = o.transpose(-1, -2).reshape(b, c, hh, ww)
        elif self.layout == "bhdn":  # (b, h, d, N): YOLO26 / ultralytics PSA style, softmax over keys
            f = lambda t: t.reshape(b, h, d, hh * ww)  # noqa: E731
            q, k, v = f(q), f(k), f(v)
            a = (q.transpose(-1, -2) @ k * d ** -0.5).softmax(-1)  # (b, h, Nq, Nk)
            o = (v @ a.transpose(-1, -2)).reshape(b, c, hh, ww)
        elif self.layout == "bnc":  # tokens (b, N, C), one head, Linear-style
            t = lambda z: z.flatten(2).transpose(1, 2)  # noqa: E731
            q, k, v = t(q), t(k), t(v)
            a = (q @ k.transpose(1, 2) * c ** -0.5).softmax(-1)
            o = (a @ v).transpose(1, 2).reshape(b, c, hh, ww)
        return self.proj(o)


CASES = {
    "Attn_bhnd_h1": Proj(1), "Attn_bhnd_h2": Proj(2), "Attn_bhnd_h4": Proj(4),
    "Attn_bhnd_h2_scalefirst": Proj(2, scale_first=True), "Attn_sdpa_h2": Proj(2, sdpa=True),
    "Attn_sdpa_h1": Proj(1, sdpa=True), "Attn_bhdn_h2": Proj(2, layout="bhdn"), "Attn_bhdn_h1": Proj(1, layout="bhdn"),
    "Attn_bnc_h1": Proj(1, layout="bnc"),
    "MHA_torch": None,  # nn.MultiheadAttention (filled below)
}


class MHAWrap(nn.Module):
    def __init__(self, heads=2):
        super().__init__()
        self.m = nn.MultiheadAttention(C, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        return self.m(t, t, t, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w)


CASES["MHA_torch"] = MHAWrap(2)
CASES["MHA_torch_h4"] = MHAWrap(4)

if __name__ == "__main__":
    out = sys.argv[1]
    torch.manual_seed(0)
    for name, mod in CASES.items():
        d = os.path.join(out, "T_" + name)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "dom.txt"), "w").write("normal")
        torch.onnx.export(L(mod.eval()).eval(), torch.randn(1, C, H, W), os.path.join(d, "m.onnx"), opset_version=17,
                          input_names=["x"], output_names=["y"], dynamo=False)
        print("T_" + name)

"""Minimal PyTorch probes for the Hailo-8 silent-wrong reports. Each case exports DIR/m.onnx (opset 17).

Usage: python models.py OUT_DIR CASE [CASE ...]      (any env with torch + onnx; no Hailo software needed)
       python models.py OUT_DIR all
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynMatmul(nn.Module):
    """#1  x -> conv3x3 -> relu -> {conv1x1 -> relu = Q, conv1x1 -> relu = K} -> Q^T K (100 x 100) -> conv1x1."""
    def __init__(self, c=32, h=10):
        super().__init__()
        self.h = h
        self.stem = nn.Conv2d(c, c, 3, 1, 1)
        self.q, self.k = nn.Conv2d(c, c, 1), nn.Conv2d(c, c, 1)
        self.out = nn.Conv2d(h * h, c, 1)

    def forward(self, x):
        y = F.relu(self.stem(x))
        b, c, h, w = y.shape
        q = F.relu(self.q(y)).reshape(b, c, h * w).transpose(1, 2)  # (1, N, C)
        k = F.relu(self.k(y)).reshape(b, c, h * w)                  # (1, C, N): non-negative weights operand
        a = q @ k                                                    # (1, N, N)
        return self.out(a.transpose(1, 2).reshape(b, h * w, h, w))


class GN(nn.Module):
    """#2  x -> GroupNorm(groups, C) -> conv1x1."""
    def __init__(self, c=64, groups=1):
        super().__init__()
        self.gn = nn.GroupNorm(groups, c)
        with torch.no_grad():
            self.gn.weight.uniform_(0.5, 1.5)
            self.gn.bias.normal_(0, 0.1)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.out(self.gn(x))


class GN1Rewrite(GN):
    """#2 workaround: the same function, statistics spatial-first, then over channels."""
    def forward(self, x):
        m = x.mean((2, 3), keepdim=True).mean(1, keepdim=True)
        d = x - m
        v = (d * d).mean((2, 3), keepdim=True).mean(1, keepdim=True)
        y = d * torch.reciprocal(torch.sqrt(v + self.gn.eps))
        return self.out(y * self.gn.weight.view(1, -1, 1, 1) + self.gn.bias.view(1, -1, 1, 1))


class ChannelLN(nn.Module):
    """ConvNeXt-style LayerNorm over channels only: x -> permute -> LayerNorm(C) -> permute -> conv1x1."""
    def __init__(self, c=64):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.out(self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class Dilated(nn.Module):
    """#3  x -> Conv2d(C, C, 3, padding=d, dilation=d).  pad_to > W: zero-pad W, conv, crop (same function)."""
    def __init__(self, c=24, w=50, pad_to=0, d=3):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=d, dilation=d)
        self.w, self.pad_to = w, pad_to

    def forward(self, x):
        if self.pad_to > self.w:
            return self.conv(F.pad(x, (0, self.pad_to - self.w)))[..., :self.w]
        return self.conv(x)


class Attention(nn.Module):
    """#4 #6 #7  single-head attention over N = h*h tokens:
    q, k, v = conv1x1(x); y = conv1x1( softmax(q^T k * t) v^T ).   kb / vb: constant offset added to K / V biases."""
    def __init__(self, c=32, t=0.2, kb=0.0, vb=0.0):
        super().__init__()
        self.q, self.k, self.v, self.out = (nn.Conv2d(c, c, 1) for _ in range(4))
        with torch.no_grad():
            self.k.bias += kb
            self.v.bias += vb
        self.t = t

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).reshape(b, c, h * w).transpose(1, 2)  # (1, N, C)
        k = self.k(x).reshape(b, c, h * w)                  # (1, C, N)
        v = self.v(x).reshape(b, c, h * w).transpose(1, 2)  # (1, N, C)
        p = torch.softmax((q @ k) * self.t, dim=-1)         # (1, N, N)
        return self.out((p @ v).transpose(1, 2).reshape(b, c, h, w))


class AttentionBiasRewrite(Attention):
    """#7 workaround: K bias dropped (constant along the softmax axis), V bias added after P.V (rows of P sum to 1)."""
    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).reshape(b, c, h * w).transpose(1, 2)
        k = F.conv2d(x, self.k.weight).reshape(b, c, h * w)
        v = F.conv2d(x, self.v.weight).reshape(b, c, h * w).transpose(1, 2)
        p = torch.softmax((q @ k) * self.t, dim=-1)
        y = p @ v + self.v.bias.view(1, 1, -1)
        return self.out(y.transpose(1, 2).reshape(b, c, h, w))


class ConvT(nn.Module):
    """#5  x -> ConvTranspose2d(C, C, k, stride=s, padding=p) -> conv1x1."""
    def __init__(self, c=16, k=2, p=0, s=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(c, c, k, s, p)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.out(self.up(x))


class ConvTRewrite(ConvT):
    """#5 workaround (k = s = 2 only): conv1x1 to 4C channels + DepthToSpace(2, CRD) - the identical function."""
    def __init__(self, c=16):
        super().__init__(c, 2, 0)
        w = self.up.weight  # (Cin, Cout, 2, 2)
        self.c1 = nn.Conv2d(c, 4 * c, 1)
        with torch.no_grad():  # CRD channel order: c * 4 + dy * 2 + dx
            self.c1.weight.copy_(w.permute(1, 2, 3, 0).reshape(4 * c, c, 1, 1))
            self.c1.bias.copy_(self.up.bias.repeat_interleave(4))

    def forward(self, x):
        return self.out(F.pixel_shuffle(self.c1(x), 2))


class Sum3(nn.Module):
    """#11  conv1x1 -> y + y + y -> conv1x1, every layer (input included) a16_w16: a UINT16 network input."""
    def __init__(self, c=16):
        super().__init__()
        self.a, self.b = nn.Conv2d(c, c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        y = self.a(x)
        return self.b(y + y + y)


def case(name):
    """name -> (module, input shape C, H, W)."""
    a = dict(kv.split("=") for kv in name.split(":")[1].split(",")) if ":" in name else {}
    f = lambda k, d: type(d)(a.get(k, d))  # noqa: E731
    fam = name.split(":")[0]
    if fam == "dynmm":
        return DynMatmul(f("c", 32), f("h", 10)), (f("c", 32), f("h", 10), f("h", 10))
    if fam in ("gn", "gn1rw"):
        c, h = f("c", 64), f("h", 20)
        return (GN(c, f("g", 1)) if fam == "gn" else GN1Rewrite(c, 1)), (c, h, h)
    if fam == "cln":
        return ChannelLN(f("c", 64)), (f("c", 64), f("h", 20), f("h", 20))
    if fam == "dil":
        return Dilated(f("c", 24), f("w", 50), f("pad_to", 0), f("d", 3)), (f("c", 24), f("h", 30), f("w", 50))
    if fam in ("attn", "attnrw"):
        cls = Attention if fam == "attn" else AttentionBiasRewrite
        n = f("h", 28)
        return cls(f("c", 32), f("t", 0.2), f("kb", 0.0), f("vb", 0.0)), (f("c", 32), n, n)
    if fam == "convt":
        return ConvT(f("c", 16), f("k", 2), f("p", 0), f("s", 2)), (f("c", 16), f("h", 24), f("h", 24))
    if fam == "convtrw":
        return ConvTRewrite(f("c", 16)), (f("c", 16), f("h", 24), f("h", 24))
    if fam == "sum3":
        return Sum3(f("c", 16)), (f("c", 16), f("h", 24), f("h", 24))
    raise SystemExit(f"unknown case {name}")


ALL = ["dynmm", "gn:g=1", "gn:g=4", "gn1rw", "dil:w=50", "dil:w=50,pad_to=56", "attn:h=8,t=0.05",
       "attn:t=0.2", "attn:t=1.0", "attn:h=16,kb=3.0", "attnrw:h=16,kb=3.0", "convt:k=2", "convt:k=3,p=1",
       "convt:k=4,p=1", "convtrw", "sum3"]


def export(out, name):
    torch.manual_seed(0)
    m, (c, h, w) = case(name)
    m = m.eval()
    d = os.path.join(out, name.replace(":", "_").replace(",", "_").replace("=", ""))
    os.makedirs(d, exist_ok=True)
    x = torch.randn(1, c, h, w)
    torch.onnx.export(m, x, f"{d}/m.onnx", input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    fold_pads(f"{d}/m.onnx", x.numpy())
    return d


def fold_pads(path, x):
    """torch exports F.pad with a computed `pads` input (ConstantOfShape/Concat/...); the Hailo parser wants a constant.
    Evaluate it once, store it as an initializer and drop the nodes that only fed it."""
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper
    m = onnx.load(path)
    g = m.graph
    init = {i.name for i in g.initializer}
    pads = [n.input[1] for n in g.node if n.op_type == "Pad" and n.input[1] not in init]
    if not pads:
        return
    t = onnx.load(path)
    t.graph.output.extend(helper.make_tensor_value_info(p, onnx.TensorProto.INT64, None) for p in pads)
    vals = ort.InferenceSession(t.SerializeToString(), providers=["CPUExecutionProvider"]).run(pads, {"x": x})
    for p, v in zip(pads, vals):
        g.initializer.append(numpy_helper.from_array(v.astype(np.int64), p))
    need = {o.name for o in g.output}
    keep = []
    for n in reversed(list(g.node)):  # keep only nodes that still feed an output
        if any(o in need for o in n.output) and not all(o in pads for o in n.output):
            keep.append(n)
            need.update(n.input)
    del g.node[:]
    g.node.extend(reversed(keep))
    onnx.save(m, path)


if __name__ == "__main__":
    for nm in (ALL if sys.argv[2:] == ["all"] else sys.argv[2:]):
        print(export(sys.argv[1], nm))

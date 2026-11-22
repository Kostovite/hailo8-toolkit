"""Export YOLO26 with every PSABlock replaced by its token-tail form (no DDR portal on Hailo-8; identical math).

Hailo keeps attention tokens as a 1 x N image. Ultralytics turns the attention output back into H x W before the 3x3
pe conv and the residual add, and the DFC routes that token->spatial reshape (feeding a 3x3 window) through host
RAM. Token-tail form: pe(v) is computed in spatial layout from a V-only 1x1 conv and reshaped INTO tokens; proj, the
residual and the ffn (all pointwise) run on (B, C, 1, N); one token->spatial reshape at the block output feeds only
pointwise consumers, which the DFC keeps on chip.

Usage: python psa_tt.py PT OUT.onnx IMGSZ
"""
import shutil
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import PSABlock


def fuse(conv):
    w = conv.conv.weight
    bn = conv.bn
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return (w * s[:, None, None, None]).detach(), (bn.bias - bn.running_mean * s).detach()


class PSABlockTT(nn.Module):
    def __init__(self, blk):
        super().__init__()
        assert blk.add, "token tail assumes the residual form"
        self.attn, self.ffn = blk.attn, blk.ffn
        at = blk.attn
        per = at.key_dim * 2 + at.head_dim
        rows = torch.cat([torch.arange(h * per + 2 * at.key_dim, h * per + 2 * at.key_dim + at.head_dim)
                          for h in range(at.num_heads)])
        w, b = fuse(at.qkv)
        self.pe_v = nn.Conv2d(w.shape[1], len(rows), 1)
        self.pe_v.weight.data, self.pe_v.bias.data = w[rows].clone(), b[rows].clone()
        for a in ("f", "i", "type", "np"):  # Ultralytics graph bookkeeping
            if hasattr(blk, a):
                setattr(self, a, getattr(blk, a))

    def forward(self, t):
        at = self.attn
        B, C, H, W = t.shape
        N = H * W
        q, k, v = at.qkv(t).view(B, at.num_heads, at.key_dim * 2 + at.head_dim, N).split(
            [at.key_dim, at.key_dim, at.head_dim], dim=2)
        attn = ((q * at.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        o = (v @ attn.transpose(-2, -1)).reshape(B, C, 1, N)
        pe = at.pe(self.pe_v(t)).reshape(B, C, 1, N)
        y = t.reshape(B, C, 1, N) + at.proj(o + pe)
        y = y + self.ffn(y)
        return y.reshape(B, C, H, W)


def swap(mod):
    n = 0
    for name, ch in mod.named_children():
        if isinstance(ch, PSABlock):
            setattr(mod, name, PSABlockTT(ch))
            n += 1
        else:
            n += swap(ch)
    return n


def main():
    pt, out, imgsz = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ref_y = YOLO(pt)
    ref_path = ref_y.export(format="onnx", imgsz=imgsz, opset=13, simplify=True, dynamic=False)
    shutil.move(ref_path, out.replace(".onnx", "_ref.onnx"))
    y = YOLO(pt)
    y.model.float().eval()
    with torch.no_grad():
        print("PSABlocks replaced:", swap(y.model))
    path = y.export(format="onnx", imgsz=imgsz, opset=13, simplify=True, dynamic=False)
    shutil.move(path, out)
    x = np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32)
    # compare the six one2one head convs (what the Hailo HEF outputs): the end-to-end top-k decode reorders
    # near-tied detections on float-rounding differences, so its output is not a usable check
    a, b = (heads(f, x) for f in (out.replace(".onnx", "_ref.onnx"), out))
    err = max(float(np.abs(a[h] - b[h]).max() / (np.abs(a[h]).max() + 1e-9)) for h in a)
    print(f"saved {out}; max relative head-conv difference vs original {err:.2e}")
    assert err < 1e-4


def heads(path, x):
    import onnx
    m = onnx.load(path)
    names = [n.output[0] for n in m.graph.node if n.op_type == "Conv" and "one2one_cv" in n.name and n.name.endswith(".2/Conv")]
    assert len(names) == 6, names
    del m.graph.output[:]
    m.graph.output.extend([onnx.helper.make_tensor_value_info(h, onnx.TensorProto.FLOAT, None) for h in names])
    return dict(zip(names, ort.InferenceSession(m.SerializeToString()).run(None, {"images": x})))

if __name__ == "__main__":
    main()

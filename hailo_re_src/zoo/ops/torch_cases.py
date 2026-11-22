"""Torch-exported probe cases: the ONNX a user actually gets from torch.onnx.export (opset 17, legacy exporter).

Each case is  x -> module -> identity 1x1 conv, exported to OUT_DIR/T_<name>/m.onnx with its input domain in dom.txt;
probe.py run T_<name> OUT_DIR then parses / quantizes / emulates / compiles it like the hand-built cases.

Usage (general env): python torch_cases.py OUT_DIR   -> prints the case names
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

C, H, W = (int(os.environ.get(k, d)) for k, d in (("TC_C", 16), ("TC_H", 24), ("TC_W", 24)))  # shape sweep: env
torch.manual_seed(0)


class L(nn.Module):
    def __init__(self, f, cout=C):
        super().__init__()
        self.f = f
        self.out = nn.Conv2d(cout, cout, 1, bias=False)
        nn.init.eye_(self.out.weight.data.view(cout, cout))

    def forward(self, x):
        return self.out(self.f(x))


class Fn(nn.Module):
    def __init__(self, fn, *mods):
        super().__init__()
        self.fn, self.mods = fn, nn.ModuleList(mods)

    def forward(self, x):
        return self.fn(x, *self.mods)


def conv(cin=C, cout=C, k=3, **kw):
    return nn.Conv2d(cin, cout, k, padding=k // 2, **kw)


class MHA(nn.Module):
    def __init__(self, sdpa=False):
        super().__init__()
        self.qkv, self.proj, self.sdpa = nn.Conv2d(C, 3 * C, 1), nn.Conv2d(C, C, 1), sdpa

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).reshape(b, 3, 2, c // 2, h * w).unbind(1)  # 2 heads
        if self.sdpa:
            o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)).transpose(-1, -2)
        else:
            a = (q.transpose(-1, -2) @ k * (c // 2) ** -0.5).softmax(-1)
            o = v @ a.transpose(-1, -2)
        return self.proj(o.reshape(b, c, h, w))


class Seq(nn.Module):
    """recurrent layer over W, features C*H -> back to (1, C, H, W)"""

    def __init__(self, typ):
        super().__init__()
        self.rnn = typ(C * H, C * H, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y, _ = self.rnn(x.reshape(b, c * h, w).transpose(1, 2))
        return y.transpose(1, 2).reshape(b, c, h, w)


class ViTBlock(nn.Module):
    """pre-norm transformer block on tokens (1, N, C): LayerNorm -> MHA -> LayerNorm -> GELU MLP"""

    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.att = nn.MultiheadAttention(C, 2, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        u = self.n1(t)
        t = t + self.att(u, u, u, need_weights=False)[0]
        t = t + self.mlp(self.n2(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


CASES = {
    # activations
    "GELU": (nn.GELU(), "normal"), "GELU_tanh": (nn.GELU("tanh"), "normal"), "SiLU": (nn.SiLU(), "normal"),
    "Mish": (nn.Mish(), "normal"), "Hardswish": (nn.Hardswish(), "normal"), "Hardsigmoid": (nn.Hardsigmoid(), "normal"),
    "SELU": (nn.SELU(), "normal"), "CELU": (nn.CELU(), "normal"), "ELU": (nn.ELU(), "normal"),
    "Softplus": (nn.Softplus(), "normal"), "Tanhshrink": (nn.Tanhshrink(), "normal"),
    "Hardtanh": (nn.Hardtanh(), "normal"), "ReLU6": (nn.ReLU6(), "normal"), "LogSigmoid": (nn.LogSigmoid(), "normal"),
    "Softshrink": (nn.Softshrink(), "normal"), "Hardshrink": (nn.Hardshrink(), "normal"),
    "Threshold": (nn.Threshold(0.3, 0.0), "normal"), "QuickGELU": (Fn(lambda x: x * torch.sigmoid(1.702 * x)), "normal"),
    "Sin": (Fn(torch.sin), "normal"), "Cos": (Fn(torch.cos), "normal"), "Erf": (Fn(torch.erf), "normal"),
    "Atan": (Fn(torch.atan), "normal"), "Exp": (Fn(torch.exp), "small"), "Log": (Fn(torch.log), "pos"),
    "Sqrt": (Fn(torch.sqrt), "pos"), "Rsqrt": (Fn(torch.rsqrt), "pos"), "Reciprocal": (Fn(torch.reciprocal), "pos"),
    "Square": (Fn(torch.square), "normal"), "Pow2": (Fn(lambda x: x ** 2), "normal"), "Pow3": (Fn(lambda x: x ** 3), "small"),
    "Pow05": (Fn(lambda x: x ** 0.5), "pos"), "Abs": (Fn(torch.abs), "normal"), "Sign": (Fn(torch.sign), "normal"),
    "Round": (Fn(torch.round), "normal"), "Floor": (Fn(torch.floor), "normal"), "Clamp": (Fn(lambda x: x.clamp(-1, 2)), "normal"),
    # binary / reductions / statistics
    "Div_tensor": (Fn(lambda x, c: x / (c(x).abs() + 0.5), conv()), "normal"),
    "Max_tensor": (Fn(lambda x, c: torch.maximum(x, c(x)), conv()), "normal"),
    "Amax_C": (Fn(lambda x: x.amax(1, keepdim=True)), "normal", 1), "Amax_W": (Fn(lambda x: x.amax(3, keepdim=True)), "normal"),
    "Amax_HW": (Fn(lambda x: x.amax((2, 3), keepdim=True)), "normal"), "Amin_W": (Fn(lambda x: x.amin(3, keepdim=True)), "normal"),
    "Mean_C": (Fn(lambda x: x.mean(1, keepdim=True)), "normal", 1), "Var_C": (Fn(lambda x: x.var(1, keepdim=True)), "normal", 1),
    "Std_HW": (Fn(lambda x: x.std((2, 3), keepdim=True)), "normal"), "Norm_C": (Fn(lambda x: x.norm(dim=1, keepdim=True)), "normal", 1),
    "Logsumexp_C": (Fn(lambda x: x.logsumexp(1, keepdim=True)), "normal", 1),
    "Cumsum_W": (Fn(lambda x: x.cumsum(3)), "normal"), "Argmax_C": (Fn(lambda x: x.argmax(1, keepdim=True).float()), "normal", 1),
    "Softmax_C": (nn.Softmax(1), "normal"), "Softmax_W": (nn.Softmax(-1), "normal"), "LogSoftmax_C": (nn.LogSoftmax(1), "normal"),
    "Sigmoid_gate": (Fn(lambda x, c: x * torch.sigmoid(c(x)), conv()), "normal"),
    # normalization
    "BatchNorm2d": (nn.BatchNorm2d(C).eval(), "normal"), "InstanceNorm2d": (nn.InstanceNorm2d(C, affine=True), "normal"),
    "GroupNorm": (nn.GroupNorm(4, C), "normal"), "GroupNorm1": (nn.GroupNorm(1, C), "normal"),
    "LayerNorm_C": (Fn(lambda x, n: n(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), nn.LayerNorm(C)), "normal"),
    "LayerNorm_CHW": (nn.LayerNorm([C, H, W]), "normal"), "LocalRespNorm": (nn.LocalResponseNorm(3), "normal"),
    "F_normalize_C": (Fn(lambda x: F.normalize(x, dim=1)), "normal"),
    "RMSNorm_C": (Fn(lambda x: x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-6)), "normal"),
    # pooling / resampling / layout
    "AvgPool3s1": (nn.AvgPool2d(3, 1, 1), "normal"), "AvgPool3s1_nopadcount": (nn.AvgPool2d(3, 1, 1, count_include_pad=False), "normal"),
    "AdaptiveAvg4": (nn.AdaptiveAvgPool2d(4), "normal"), "AdaptiveMax4": (nn.AdaptiveMaxPool2d(4), "normal"),
    "MaxPool3s2": (nn.MaxPool2d(3, 2, 1), "normal"), "MaxPool_ceil": (nn.MaxPool2d(3, 2, ceil_mode=True), "normal"),
    "Upsample_bilinear": (nn.Upsample(scale_factor=2, mode="bilinear"), "normal"),
    "Upsample_bilinear_ac": (nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), "normal"),
    "Upsample_bicubic": (nn.Upsample(scale_factor=2, mode="bicubic"), "normal"),
    "Interp_down_bilinear": (Fn(lambda x: F.interpolate(x, scale_factor=0.5, mode="bilinear")), "normal"),
    "PixelShuffle": (nn.PixelShuffle(2), "normal", C // 4), "PixelUnshuffle": (nn.PixelUnshuffle(2), "normal", 4 * C),
    "Flip_W": (Fn(lambda x: x.flip(3)), "normal"), "Roll_W": (Fn(lambda x: x.roll(3, 3)), "normal"),
    "Chunk_add": (Fn(lambda x: sum(x.chunk(2, 1))), "normal", C // 2),
    "ZeroPad": (nn.ZeroPad2d(1), "normal"), "ReflectPad": (nn.ReflectionPad2d(1), "normal"),
    "ReplicatePad": (nn.ReplicationPad2d(1), "normal"),
    "GridSample": (Fn(lambda x: F.grid_sample(x, torch.rand(1, H, W, 2) * 2 - 1, align_corners=False)), "normal"),
    "Unfold_fold": (Fn(lambda x: F.fold(F.unfold(x, 2, stride=2), (H, W), 2, stride=2)), "normal"),
    # convolutions
    "ConvTranspose_k3s2": (nn.ConvTranspose2d(C, C, 3, 2, 1, output_padding=1), "normal"),
    "Conv_dil3": (nn.Conv2d(C, C, 3, padding=3, dilation=3), "normal"),
    "Conv_5x1": (nn.Conv2d(C, C, (5, 1), padding=(2, 0)), "normal"),
    "Conv1d_seq": (Fn(lambda x, c: c(x.flatten(2)).reshape(x.shape), nn.Conv1d(C, C, 5, padding=2)), "normal"),
    # attention / transformer / recurrent
    "Attention_manual": (MHA(), "normal"), "Attention_sdpa": (MHA(sdpa=True), "normal"), "ViT_block": (ViTBlock(), "normal"),
    "Linear_tokens": (Fn(lambda x, l: l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), nn.Linear(C, C)), "normal"),
    "LSTM": (Seq(nn.LSTM), "normal"), "GRU": (Seq(nn.GRU), "normal"),
}


def main(out):
    for name, spec in CASES.items():
        mod, dom = spec[0], spec[1]
        cout = spec[2] if len(spec) > 2 else C
        d = os.path.join(out, "T_" + name)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "dom.txt"), "w").write(dom)
        m = L(mod, cout).eval()
        try:
            torch.onnx.export(m, torch.randn(1, C, H, W), os.path.join(d, "m.onnx"), opset_version=17,
                              input_names=["x"], output_names=["y"], dynamo=False)
            print("T_" + name)
        except Exception as e:  # noqa: BLE001
            open(os.path.join(d, "export_error.txt"), "w").write(f"{type(e).__name__}: {e}")
            print(f"# T_{name} export failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])

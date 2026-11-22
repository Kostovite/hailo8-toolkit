"""Hailo-8 / DFC operator probe: one micro ONNX per op -> parse -> level-0 quantize -> bit-exact HSim vs float -> compile.

Each case is  X -> <op graph> -> identity 1x1 Conv -> Y  (the conv gives every op a real quantized consumer). The
error of the op is the SQNR (dB) of the bit-exact HSim output against onnxruntime float on held-out inputs drawn
from the op's domain; case "baseline" (no op) is the 8-bit floor every other number should be read against.

Usage (DFC env): python probe.py list
                 python probe.py run CASE OUT_DIR [--no-compile]      -> OUT_DIR/CASE/result.json
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

C, H, W = 16, 24, 24
RNG = np.random.default_rng(0)


def domain(kind, n, rng):
    shape = (n, C, H, W)
    if kind == "normal":
        return rng.normal(0, 1.5, shape)
    if kind == "pos":
        return rng.uniform(0.05, 4, shape)
    if kind == "unit":
        return rng.uniform(-0.95, 0.95, shape)
    if kind == "ge1":
        return rng.uniform(1.05, 4, shape)
    if kind == "small":
        return rng.uniform(-2.5, 2.5, shape)
    if kind == "near1":  # products of many values (CumProd / ReduceProd)
        return rng.uniform(0.85, 1.15, shape)
    raise ValueError(kind)


class G:
    """Tiny graph builder: g.op("Relu", x) -> output name."""

    def __init__(self):
        self.nodes, self.inits, self.k = [], [], 0

    def c(self, arr, dtype=np.float32):
        self.k += 1
        name = f"c{self.k}"
        self.inits.append(numpy_helper.from_array(np.asarray(arr, dtype), name))
        return name

    def op(self, typ, *ins, n_out=1, **attrs):
        self.k += 1
        outs = [f"t{self.k}_{i}" for i in range(n_out)]
        self.nodes.append(helper.make_node(typ, list(ins), outs, name=f"{typ}_{self.k}", **attrs))
        return outs[0] if n_out == 1 else outs

    def conv(self, x, cin, cout, k=1, rand=False, **attrs):
        if rand:
            w = RNG.normal(0, 1 / math.sqrt(cin * k * k), (cout, cin // attrs.get("group", 1), k, k))
        else:
            w = np.zeros((cout, cin, 1, 1))
            w[np.arange(min(cin, cout)), np.arange(min(cin, cout))] = 1
        return self.op("Conv", x, self.c(w), self.c(np.zeros(cout)), kernel_shape=[k, k],
                       pads=[k // 2] * 4 if k > 1 else [0] * 4, **attrs)


def s(v):  # scalar constant: rank 0, as torch.onnx.export writes it (the parser's Pow reads float(init.tolist()))
    return np.array(v, np.float32)


# name -> (opset, domain, builder(g, x) -> y, out_channels)
CASES = {}


def case(name, opset=17, dom="normal", cout=C):
    def deco(f):
        CASES[name] = (opset, dom, f, cout)
        return f
    return deco


def unary(name, typ, dom="normal", opset=17, **attrs):
    CASES[name] = (opset, dom, lambda g, x: g.op(typ, x, **attrs), C)


case("baseline")(lambda g, x: x)
# --- unary activations (ONNX ops) ---
for nm, typ, dom in [("Relu", "Relu", "normal"), ("Sigmoid", "Sigmoid", "normal"), ("Tanh", "Tanh", "normal"),
                     ("Exp", "Exp", "small"), ("Log", "Log", "pos"), ("Sqrt", "Sqrt", "pos"),
                     ("Reciprocal", "Reciprocal", "pos"), ("Abs", "Abs", "normal"), ("Neg", "Neg", "normal"),
                     ("Softplus", "Softplus", "normal"), ("Softsign", "Softsign", "normal"), ("Sign", "Sign", "normal"),
                     ("Erf", "Erf", "normal"), ("Floor", "Floor", "normal"), ("Ceil", "Ceil", "normal"),
                     ("Round", "Round", "normal"), ("Sin", "Sin", "normal"), ("Cos", "Cos", "normal"),
                     ("Tan", "Tan", "unit"), ("Asin", "Asin", "unit"), ("Acos", "Acos", "unit"), ("Atan", "Atan", "normal"),
                     ("Sinh", "Sinh", "small"), ("Cosh", "Cosh", "small"), ("Asinh", "Asinh", "normal"),
                     ("Acosh", "Acosh", "ge1"), ("Atanh", "Atanh", "unit"), ("HardSwish", "HardSwish", "normal"),
                     ("HardSigmoid", "HardSigmoid", "normal"), ("Selu", "Selu", "normal"), ("Elu", "Elu", "normal"),
                     ("Celu", "Celu", "normal")]:
    unary(nm, typ, dom)
unary("LeakyRelu", "LeakyRelu", alpha=0.1)
unary("ThresholdedRelu", "ThresholdedRelu", alpha=0.5)
unary("Shrink", "Shrink", lambd=0.5, bias=0.5)
unary("Mish_op18", "Mish", opset=18)
unary("Gelu_op20", "Gelu", opset=20)
unary("Gelu_tanh_op20", "Gelu", opset=20, approximate="tanh")
case("Clip_relu6")(lambda g, x: g.op("Clip", x, g.c(0.0), g.c(6.0)))
case("Clip_pm1")(lambda g, x: g.op("Clip", x, g.c(-1.0), g.c(1.0)))
case("PRelu")(lambda g, x: g.op("PRelu", x, g.c(RNG.uniform(0, 0.3, (C, 1, 1)))))
# --- activation patterns the parser fuses ---
case("SiLU_pattern")(lambda g, x: g.op("Mul", x, g.op("Sigmoid", x)))
case("Swish_beta")(lambda g, x: g.op("Mul", x, g.op("Sigmoid", g.op("Mul", x, g.c(s(1.7))))))
case("Mish_pattern")(lambda g, x: g.op("Mul", x, g.op("Tanh", g.op("Softplus", x))))
case("Gelu_erf_pattern")(lambda g, x: g.op("Mul", g.op("Mul", x, g.c(s(0.5))),
                                           g.op("Add", g.op("Erf", g.op("Div", x, g.c(s(math.sqrt(2))))), g.c(s(1.0)))))
case("Gelu_tanh_pattern")(lambda g, x: g.op("Mul", g.op("Mul", x, g.c(s(0.5))), g.op("Add", g.c(s(1.0)), g.op(
    "Tanh", g.op("Mul", g.c(s(math.sqrt(2 / math.pi))), g.op("Add", x, g.op("Mul", g.c(s(0.044715)),
                                                                             g.op("Pow", x, g.c(s(3.0))))))))))
case("Gelu_sigmoid_pattern")(lambda g, x: g.op("Mul", x, g.op("Sigmoid", g.op("Mul", x, g.c(s(1.702))))))
case("HardSwish_pattern")(lambda g, x: g.op("Div", g.op("Mul", x, g.op("Clip", g.op("Add", x, g.c(s(3.0))), g.c(0.0),
                                                                       g.c(6.0))), g.c(s(6.0))))
case("Square_mul")(lambda g, x: g.op("Mul", x, x))
case("Pow2", dom="normal")(lambda g, x: g.op("Pow", x, g.c(s(2.0))))
case("Pow3", dom="small")(lambda g, x: g.op("Pow", x, g.c(s(3.0))))
case("Pow05", dom="pos")(lambda g, x: g.op("Pow", x, g.c(s(0.5))))
case("Pow_m05", dom="pos")(lambda g, x: g.op("Pow", x, g.c(s(-0.5))))
case("Pow15", dom="pos")(lambda g, x: g.op("Pow", x, g.c(s(1.5))))
case("Rsqrt_pattern", dom="pos")(lambda g, x: g.op("Reciprocal", g.op("Sqrt", x)))
case("Div_by_const")(lambda g, x: g.op("Div", x, g.c(s(3.0))))
case("Const_div_x", dom="pos")(lambda g, x: g.op("Div", g.c(s(1.0)), x))
# --- binary element-wise (two tensors) ---
two = lambda g, x: (x, g.conv(x, C, C, 3, rand=True))  # noqa: E731
for nm in ["Add", "Sub", "Mul", "Max", "Min"]:
    case(f"{nm}_tensors")(lambda g, x, nm=nm: g.op(nm, *two(g, x)))
case("Div_tensors", dom="pos")(lambda g, x: g.op("Div", x, g.op("Add", g.op("Abs", g.conv(x, C, C, 3, rand=True)),
                                                                g.c(s(0.5)))))
case("Pow_tensors", dom="pos")(lambda g, x: g.op("Pow", x, g.op("Sigmoid", g.conv(x, C, C, 3, rand=True))))
case("Greater_where")(lambda g, x: g.op("Where", g.op("Greater", x, g.c(s(0.3))), x, g.c(s(0.0))))
case("Equal")(lambda g, x: g.op("Cast", g.op("Equal", g.op("Round", x), g.c(s(1.0))), to=TensorProto.FLOAT))
case("Mod", dom="pos")(lambda g, x: g.op("Mod", x, g.c(s(1.0)), fmod=1))
# --- reductions ---
for nm in ["ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "ReduceL2", "ReduceL1", "ReduceSumSquare",
           "ReduceLogSumExp", "ReduceProd"]:
    for ax, tag in (([1], "C"), ([3], "W"), ([2, 3], "HW")):
        cout = 1 if tag == "C" else C
        case(f"{nm}_{tag}", cout=cout)(lambda g, x, nm=nm, ax=ax: g.op(nm, x, g.c(np.array(ax, np.int64), np.int64),
                                                                       keepdims=1) if nm in ("ReduceSum",) else
                                       g.op(nm, x, axes=ax, keepdims=1))
case("ArgMax_C", cout=1)(lambda g, x: g.op("Cast", g.op("ArgMax", x, axis=1, keepdims=1), to=TensorProto.FLOAT))
case("ArgMin_C", cout=1)(lambda g, x: g.op("Cast", g.op("ArgMin", x, axis=1, keepdims=1), to=TensorProto.FLOAT))
case("TopK_C", cout=4)(lambda g, x: g.op("TopK", x, g.c(np.array([4], np.int64), np.int64), axis=1, n_out=2)[0])
case("CumSum_W")(lambda g, x: g.op("CumSum", x, g.c(np.array(3, np.int64), np.int64)))
# --- softmax family ---
for ax, tag in ((1, "C"), (3, "W")):
    case(f"Softmax_{tag}")(lambda g, x, ax=ax: g.op("Softmax", x, axis=ax))
    case(f"LogSoftmax_{tag}")(lambda g, x, ax=ax: g.op("LogSoftmax", x, axis=ax))
    case(f"Hardmax_{tag}")(lambda g, x, ax=ax: g.op("Hardmax", x, axis=ax))
# --- normalization ---
case("BatchNorm")(lambda g, x: g.op("BatchNormalization", x, g.c(RNG.uniform(0.5, 2, C)), g.c(RNG.normal(0, 1, C)),
                                    g.c(RNG.normal(0, 1, C)), g.c(RNG.uniform(0.5, 2, C))))
case("InstanceNorm")(lambda g, x: g.op("InstanceNormalization", x, g.c(RNG.uniform(0.5, 2, C)), g.c(RNG.normal(0, .3, C))))
case("LayerNorm_C")(lambda g, x: g.op("Transpose", g.op("LayerNormalization", g.op("Transpose", x, perm=[0, 2, 3, 1]),
                                                       g.c(RNG.uniform(0.5, 2, C)), g.c(RNG.normal(0, .3, C)), axis=-1),
                                      perm=[0, 3, 1, 2]))
case("LayerNorm_W")(lambda g, x: g.op("LayerNormalization", x, g.c(RNG.uniform(0.5, 2, W)), g.c(RNG.normal(0, .3, W)),
                                      axis=-1))
case("GroupNorm_op18")(lambda g, x: g.op("GroupNormalization", x, g.c(RNG.uniform(0.5, 2, 4)), g.c(RNG.normal(0, .3, 4)),
                                         num_groups=4))
CASES["GroupNorm_op18"] = (18,) + CASES["GroupNorm_op18"][1:]
case("LpNorm_C")(lambda g, x: g.op("LpNormalization", x, axis=1, p=2))
case("LRN")(lambda g, x: g.op("LRN", x, size=3))
case("MeanVarNorm")(lambda g, x: g.op("MeanVarianceNormalization", x))
case("L2Norm_pattern")(lambda g, x: g.op("Div", x, g.op("Add", g.op("ReduceL2", x, axes=[1], keepdims=1), g.c(s(1e-6)))))
case("RMSNorm_pattern")(lambda g, x: g.op("Mul", x, g.op("Reciprocal", g.op("Sqrt", g.op("Add", g.op(
    "ReduceMean", g.op("Mul", x, x), axes=[1], keepdims=1), g.c(s(1e-6)))))))
# --- pooling ---
case("MaxPool3s1")(lambda g, x: g.op("MaxPool", x, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
case("MaxPool2s2")(lambda g, x: g.op("MaxPool", x, kernel_shape=[2, 2], strides=[2, 2]))
case("MaxPool5s1")(lambda g, x: g.op("MaxPool", x, kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1]))
case("AvgPool2s2")(lambda g, x: g.op("AveragePool", x, kernel_shape=[2, 2], strides=[2, 2]))
case("AvgPool3s1_pad")(lambda g, x: g.op("AveragePool", x, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
case("AvgPool3s1_pad_incl")(lambda g, x: g.op("AveragePool", x, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
                                              count_include_pad=1))
case("AvgPool7s1")(lambda g, x: g.op("AveragePool", x, kernel_shape=[7, 7], pads=[3, 3, 3, 3], strides=[1, 1]))
case("AvgPool3s2")(lambda g, x: g.op("AveragePool", x, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]))
case("GlobalAvgPool")(lambda g, x: g.op("GlobalAveragePool", x))
case("GlobalMaxPool")(lambda g, x: g.op("GlobalMaxPool", x))
case("LpPool")(lambda g, x: g.op("LpPool", x, kernel_shape=[2, 2], strides=[2, 2], p=2))
# --- conv family ---
case("Conv3x3")(lambda g, x: g.conv(x, C, C, 3, rand=True))
case("Conv7x7")(lambda g, x: g.conv(x, C, C, 7, rand=True))
case("Conv3x3_dil2")(lambda g, x: g.op("Conv", x, g.c(RNG.normal(0, .1, (C, C, 3, 3))), kernel_shape=[3, 3],
                                       pads=[2, 2, 2, 2], dilations=[2, 2]))
case("Conv3x3_s2")(lambda g, x: g.op("Conv", x, g.c(RNG.normal(0, .1, (C, C, 3, 3))), kernel_shape=[3, 3],
                                     pads=[1, 1, 1, 1], strides=[2, 2]))
case("DWConv3x3")(lambda g, x: g.conv(x, C, C, 3, rand=True, group=C))
case("GroupConv_g4")(lambda g, x: g.conv(x, C, C, 3, rand=True, group=4))
case("ConvTranspose_s2")(lambda g, x: g.op("ConvTranspose", x, g.c(RNG.normal(0, .2, (C, C, 2, 2))), kernel_shape=[2, 2],
                                           strides=[2, 2]))
case("ConvTranspose_k4s2")(lambda g, x: g.op("ConvTranspose", x, g.c(RNG.normal(0, .1, (C, C, 4, 4))),
                                             kernel_shape=[4, 4], strides=[2, 2], pads=[1, 1, 1, 1]))
case("Conv1d_pattern")(lambda g, x: g.op("Reshape", g.op("Conv", g.op("Reshape", x, g.c(np.array([1, C, H * W], np.int64),
                                                                                           np.int64)),
                                                         g.c(RNG.normal(0, .2, (C, C, 3))), kernel_shape=[3], pads=[1, 1]),
                                         g.c(np.array([1, C, H, W], np.int64), np.int64)))
# --- matmul / attention ---
case("MatMul_const_W")(lambda g, x: g.op("MatMul", x, g.c(RNG.normal(0, .2, (W, W)))))
case("Gemm_flat", cout=C)(lambda g, x: g.op("Reshape", g.op("Gemm", g.op("Flatten", g.op("GlobalAveragePool", x)),
                                                            g.c(RNG.normal(0, .3, (C, C))), transB=1),
                                            g.c(np.array([1, C, 1, 1], np.int64), np.int64)))
case("MatMul_tensors")(lambda g, x: g.op("MatMul", x, g.op("Transpose", g.conv(x, C, C, 1, rand=True), perm=[0, 1, 3, 2])))
case("Attention_pattern", cout=C)(lambda g, x: _attention(g, x))
case("Einsum_bchw")(lambda g, x: g.op("Einsum", x, g.conv(x, C, C, 1, rand=True), equation="bchw,bchv->bchv"))
# --- shape / layout ---
case("Transpose_HW")(lambda g, x: g.op("Transpose", x, perm=[0, 1, 3, 2]))
case("Reshape_flatHW")(lambda g, x: g.op("Reshape", g.op("Reshape", x, g.c(np.array([1, C, H * W, 1], np.int64), np.int64)),
                                         g.c(np.array([1, C, H, W], np.int64), np.int64)))
case("Concat_C", cout=2 * C)(lambda g, x: g.op("Concat", x, g.conv(x, C, C, 3, rand=True), axis=1))
case("Concat_W")(lambda g, x: g.op("Slice", g.op("Concat", x, g.conv(x, C, C, 3, rand=True), axis=3),
                                   g.c(np.array([0], np.int64), np.int64), g.c(np.array([W], np.int64), np.int64),
                                   g.c(np.array([3], np.int64), np.int64)))
case("Split_C", cout=C // 2)(lambda g, x: g.op("Add", *g.op("Split", x, g.c(np.array([C // 2, C // 2], np.int64), np.int64),
                                                             axis=1, n_out=2)))
case("Slice_C", cout=8)(lambda g, x: g.op("Slice", x, g.c(np.array([4], np.int64), np.int64),
                                          g.c(np.array([12], np.int64), np.int64), g.c(np.array([1], np.int64), np.int64)))
case("Slice_HW_stride2")(lambda g, x: g.op("Slice", x, g.c(np.array([0, 0], np.int64), np.int64),
                                           g.c(np.array([H, W], np.int64), np.int64), g.c(np.array([2, 3], np.int64), np.int64),
                                           g.c(np.array([2, 2], np.int64), np.int64)))
case("Gather_C", cout=4)(lambda g, x: g.op("Gather", x, g.c(np.array([0, 3, 5, 9], np.int64), np.int64), axis=1))
case("Tile_W")(lambda g, x: g.op("Tile", g.op("ReduceMean", x, axes=[3], keepdims=1),
                                 g.c(np.array([1, 1, 1, W], np.int64), np.int64)))
case("Expand_W")(lambda g, x: g.op("Expand", g.op("ReduceMean", x, axes=[3], keepdims=1),
                                   g.c(np.array([1, C, H, W], np.int64), np.int64)))
case("Pad_const")(lambda g, x: g.op("Pad", x, g.c(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), np.int64)))
case("Pad_reflect")(lambda g, x: g.op("Pad", x, g.c(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), np.int64), mode="reflect"))
case("Pad_edge")(lambda g, x: g.op("Pad", x, g.c(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), np.int64), mode="edge"))
case("DepthToSpace", cout=C // 4)(lambda g, x: g.op("DepthToSpace", x, blocksize=2))
case("SpaceToDepth", cout=4 * C)(lambda g, x: g.op("SpaceToDepth", x, blocksize=2))
case("PixelShuffle_pattern", cout=C // 4)(lambda g, x: g.op("Reshape", g.op("Transpose", g.op(
    "Reshape", x, g.c(np.array([1, C // 4, 2, 2, H, W], np.int64), np.int64)), perm=[0, 1, 4, 2, 5, 3]),
    g.c(np.array([1, C // 4, 2 * H, 2 * W], np.int64), np.int64)))
case("ChannelShuffle_pattern")(lambda g, x: g.op("Reshape", g.op("Transpose", g.op(
    "Reshape", x, g.c(np.array([1, 4, C // 4, H, W], np.int64), np.int64)), perm=[0, 2, 1, 3, 4]),
    g.c(np.array([1, C, H, W], np.int64), np.int64)))
for mode, extra, tag in (("nearest", {}, "nearest"), ("linear", {}, "bilinear"),
                         ("linear", {"coordinate_transformation_mode": "align_corners"}, "bilinear_ac"),
                         ("cubic", {}, "bicubic")):
    case(f"Resize2x_{tag}")(lambda g, x, mode=mode, extra=extra: g.op("Resize", x, "", g.c(np.array([1, 1, 2, 2], np.float32)),
                                                                      mode=mode, **extra))
case("Resize_down_bilinear")(lambda g, x: g.op("Resize", x, "", g.c(np.array([1, 1, .5, .5], np.float32)), mode="linear"))
case("GridSample", opset=17)(lambda g, x: g.op("GridSample", x, g.c(RNG.uniform(-1, 1, (1, H, W, 2))),
                                               align_corners=0))
case("RoiAlign", cout=C)(lambda g, x: g.op("Resize", g.op("RoiAlign", x, g.c(np.array([[2, 2, 18, 18]], np.float32)),
                                                          g.c(np.array([0], np.int64), np.int64),
                                                          output_height=12, output_width=12),
                                           "", g.c(np.array([1, 1, 2, 2], np.float32)), mode="nearest"))
# --- recurrent / sequence / spectral ---
case("LSTM_W")(lambda g, x: _rnn(g, x, "LSTM", 4))
case("GRU_W")(lambda g, x: _rnn(g, x, "GRU", 3))
case("RNN_W")(lambda g, x: _rnn(g, x, "RNN", 1))
case("DFT_W", opset=17)(lambda g, x: g.op("ReduceSum", g.op("DFT", g.op("Unsqueeze", x, g.c(np.array([4], np.int64), np.int64)),
                                                           axis=3), g.c(np.array([4], np.int64), np.int64), keepdims=0))
case("STFT_pattern_conv", cout=C)(lambda g, x: _stft_conv(g, x))


def _attention(g, x):
    """single-head self-attention over the H*W tokens: softmax(Q K^T / sqrt(d)) V with 1x1-conv projections."""
    q, k, v = (g.conv(x, C, C, 1, rand=True) for _ in range(3))
    tok = lambda t: g.op("Reshape", t, g.c(np.array([1, C, H * W], np.int64), np.int64))  # noqa: E731
    qt = g.op("Transpose", tok(q), perm=[0, 2, 1])  # (1, N, C)
    a = g.op("Softmax", g.op("Mul", g.op("MatMul", qt, tok(k)), g.c(np.array(1 / math.sqrt(C), np.float32))), axis=-1)
    o = g.op("MatMul", tok(v), g.op("Transpose", a, perm=[0, 2, 1]))  # (1, C, N)
    return g.op("Reshape", o, g.c(np.array([1, C, H, W], np.int64), np.int64))


def _rnn(g, x, typ, gates):
    """sequence over W of C*H features, hidden C*H; output back to (1, C, H, W)."""
    F = C * H
    seq = g.op("Transpose", g.op("Reshape", x, g.c(np.array([1, F, W], np.int64), np.int64)), perm=[2, 0, 1])  # (W,1,F)
    Wt = g.c(RNG.normal(0, 1 / math.sqrt(F), (1, gates * F, F)))
    R = g.c(RNG.normal(0, 1 / math.sqrt(F), (1, gates * F, F)))
    y = g.op(typ, seq, Wt, R, hidden_size=F, n_out=1)  # (W, 1, 1, F)
    return g.op("Reshape", g.op("Transpose", g.op("Reshape", y, g.c(np.array([W, F], np.int64), np.int64)), perm=[1, 0]),
                g.c(np.array([1, C, H, W], np.int64), np.int64))


def _stft_conv(g, x):
    """STFT magnitude along W as a strided conv with fixed cos/sin kernels (the usual on-chip workaround)."""
    n_fft = 8
    k = np.arange(n_fft)
    win = np.hanning(n_fft)
    cos = np.cos(2 * np.pi * np.outer(np.arange(C // 2), k) / n_fft) * win
    sin = -np.sin(2 * np.pi * np.outer(np.arange(C // 2), k) / n_fft) * win
    m = g.op("ReduceMean", x, axes=[1], keepdims=1)  # one signal per row
    kr = np.concatenate([cos, sin])[:, None, None, :]  # (C, 1, 1, n_fft)
    re_im = g.op("Conv", m, g.c(kr), kernel_shape=[1, n_fft], pads=[0, n_fft // 2, 0, n_fft // 2 - 1])
    sq = g.op("Mul", re_im, re_im)
    a, b = g.op("Split", sq, g.c(np.array([C // 2, C // 2], np.int64), np.int64), axis=1, n_out=2)
    mag = g.op("Sqrt", g.op("Add", g.op("Add", a, b), g.c(s(1e-3))))
    return g.op("Concat", mag, mag, axis=1)


def fill_defaults(m):
    """Write every schema-default attribute explicitly: the DFC parser reads attributes without ONNX defaults
    (BatchNormalization without `epsilon` -> IndexError). torch.onnx.export already writes them all."""
    opset = next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    for n in m.graph.node:
        try:
            sch = onnx.defs.get_schema(n.op_type, opset, n.domain)
        except onnx.defs.SchemaError:
            continue
        have = {a.name for a in n.attribute}
        for an, a in sch.attributes.items():
            if an not in have and a.default_value is not None and a.default_value.type != 0:
                n.attribute.append(a.default_value)
                n.attribute[-1].name = an
    return m


def build(name):
    opset, dom, f, cout = CASES[name]
    g = G()
    y = f(g, "x")
    y = g.conv(y, cout, cout)  # identity consumer
    out_shape = None
    graph = helper.make_graph(g.nodes, name, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, H, W])],
                              [helper.make_tensor_value_info(y, TensorProto.FLOAT, out_shape)], g.inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 8
    m = onnx.shape_inference.infer_shapes(fill_defaults(m))
    return m, dom


def sqnr(ref, x):
    ref, x = np.asarray(ref, np.float64).ravel(), np.asarray(x, np.float64).ravel()
    return float(10 * np.log10((ref ** 2).sum() / max(((ref - x) ** 2).sum(), 1e-30)))


def supported_layers(path, name, prec, nhwc_calib):
    """Layers of the model AFTER the full-precision modifications (the stage quantization_param is checked against),
    whose Hailo layer class supports `prec` (a8_w8_a16: only the layers feeding an output). Input layer included
    unless PREC_NO_INPUT=1 (a 16-bit input lifts the 8-bit input floor)."""
    from hailo_sdk_client import ClientRunner, InferenceContext
    from hailo_model_optimization.acceleras.utils.acceleras_definitions import PrecisionMode
    r = ClientRunner(hw_arch="hailo8")
    r.translate_onnx_model(path, name.replace("-", "_"))
    r.load_model_script("model_optimization_flavor(optimization_level=0, compression_level=0)\n")
    r.optimize_full_precision(nhwc_calib[:4])
    model = r._sdk_backend.build_acceleras_model(InferenceContext.SDK_FP_OPTIMIZED)
    ladder = {"max16": ["a16_w16", "a16_w16_a16", "a8_w8_a16", "a8_w8"]}.get(prec)  # best supported per layer
    want = None if ladder else PrecisionMode(prec)
    hn = r.get_hn_dict()
    outs = {i.split("/")[-1] for v in hn["layers"].values() if v["type"] == "output_layer" for i in v["input"]}
    names, skipped = [], []
    for full, layer in model.layers.items():
        short = full.split("/")[-1]
        if prec == "a8_w8_a16" and short not in outs:
            continue
        if "input_layer" in short and os.environ.get("PREC_NO_INPUT") == "1":
            continue
        sup = getattr(layer, "SUPPORTED_PRECISION_MODE", set())
        if ladder:
            continue  # solved jointly below
        ok = want in sup
        (names if ok else skipped).append(short if ok else f"{short}:{type(layer).__name__}")
    if ladder:
        return max_precision_assignment(model, hn), []
    return names, skipped


def _bits(mode):
    """'a16_w16' -> (16, 16, 16); 'a8_w8_a16' -> (8, 8, 16): (input bits, weight bits, output bits)."""
    p = mode.split("_")
    a, w = int(p[0][1:]), int(p[1][1:])
    return a, w, int(p[2][1:]) if len(p) > 2 else a


def max_precision_assignment(model, hn):
    """Exact max-precision mode per layer (scipy MILP): every layer gets one supported mode, a producer's output
    bits equal each consumer's input bits; maximize #16-bit tensors (+ small bonus for 16-bit weights)."""
    from scipy.optimize import LinearConstraint, milp
    layers = {k.split("/")[-1]: v for k, v in model.layers.items()}
    edges = [(i.split("/")[-1], n.split("/")[-1]) for n, v in hn["layers"].items() for i in v["input"]]
    names = [n for n in layers if any(n in e for e in edges)]
    modes = {}
    for n in names:
        sup = sorted(m.value for m in getattr(layers[n], "SUPPORTED_PRECISION_MODE", set()))
        # w4 is a compression mode (and the validator rejects it on weight-less layers): never for max precision
        # a16_w8_* passes the quantizer but the allocator cannot place it ("Assignment of the node needs
        # super-defuse", ViT qkv convs): excluded too.
        modes[n] = [m for m in sup if not m.endswith("_exp") and "non_zero" not in m and "native" not in m
                    and "_w4" not in m and not m.startswith("a16_w8")] or ["a8_w8"]
        if "Concat" in type(layers[n]).__name__:  # lists a8_w8_a16, the quantizer rejects it (CumProd concat)
            modes[n] = [m for m in modes[n] if m != "a8_w8_a16"] or ["a8_w8"]
        if type(layers[n]).__name__ == "HailoArgMax":  # a16_w16 accepted, compile fails ("16x4 is not supported")
            modes[n] = ["a8_w8"]  # and a8_w8_a16 is listed but rejected
        if (type(layers[n]).__name__ in os.environ.get("PREC_EXCLUDE", "").split(",")
                or n in os.environ.get("PREC_EXCLUDE_LAYERS", "").split(",")):  # bisecting compile limits
            modes[n] = [m for m in modes[n] if m in ("a8_w8", "a8_w8_a8")] or ["a8_w8"]
    var = [(n, m) for n in names for m in modes[n]]
    nt = len(names)  # t_n: output of layer n is 16-bit
    idx = {n: i for i, n in enumerate(names)}
    N = len(var) + nt
    c = np.zeros(N)
    for j, (n, m) in enumerate(var):
        c[j] = -0.01 * (_bits(m)[1] / 8.0)  # tie-break: wider weights
    c[len(var):] = -1.0
    A, lo, hi = [], [], []
    def row():
        return np.zeros(N)
    for n in names:
        r_ = row()
        for j, (nn, m) in enumerate(var):
            if nn == n:
                r_[j] = 1
        A.append(r_); lo.append(1); hi.append(1)
        r_ = row()  # t_n = sum y * out16
        for j, (nn, m) in enumerate(var):
            if nn == n:
                r_[j] = float(_bits(m)[2] == 16)
        r_[len(var) + idx[n]] = -1
        A.append(r_); lo.append(0); hi.append(0)
    for u, v in edges:
        if u not in idx or v not in idx:
            continue
        r_ = row()  # t_u = sum_m y_{v,m} * in16(m)
        for j, (nn, m) in enumerate(var):
            if nn == v:
                r_[j] = float(_bits(m)[0] == 16)
        r_[len(var) + idx[u]] = -1
        A.append(r_); lo.append(0); hi.append(0)
    sol = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(N), bounds=(0, 1))
    if sol.x is None:
        return []
    pick = [(n, m) for j, (n, m) in enumerate(var) if sol.x[j] > 0.5]
    return [(n, m) for n, m in pick if m != "a8_w8"]


def run(name, out_dir, compile_hef=True):
    import onnxruntime as ort
    from hailo_sdk_client import ClientRunner, InferenceContext

    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    res = {"case": name, "stage": "build"}
    t0 = time.time()
    try:
        path = os.path.join(d, "m.onnx")
        if name in CASES:
            m, dom = build(name)
            onnx.save(m, path)
        else:  # torch suite: torch_cases.py exported OUT_DIR/CASE/m.onnx and wrote its domain to dom.txt
            dom = open(os.path.join(d, "dom.txt")).read().strip()
        rng = np.random.default_rng(1)
        calib = domain(dom, 64, rng).astype(np.float32)
        test = domain(dom, 8, np.random.default_rng(2)).astype(np.float32)
        ref_path = os.path.join(d, "ref.onnx")  # float reference when m.onnx holds PWL placeholders
        sess = ort.InferenceSession(ref_path if os.path.exists(ref_path) else path)
        ref = np.concatenate([sess.run(None, {"x": t[None]})[0] for t in test])
        res["stage"] = "parse"
        if os.environ.get("LSTM_LIMIT"):  # the fuser refuses LSTMs unrolled past 16 steps (8 bidirectional)
            from hailo_sdk_client.model_translator.fuser.fuser import HailoNNFuser as Fuser
            Fuser.LSTM_SEQ_LEN_WARNING_LIMIT = int(os.environ["LSTM_LIMIT"])
        runner = ClientRunner(hw_arch="hailo8")
        runner.translate_onnx_model(path, name.replace("-", "_"))
        if os.path.exists(os.path.join(d, "fit.json")):  # pwl_inject placeholders -> fitted PWL activations
            import pwl_inject
            res["pwl_layers"] = pwl_inject.apply_pwl(runner, os.path.join(d, "fit.json"))
        hn = runner.get_hn_dict()
        res["hn_types"] = sorted({v["type"] for v in hn["layers"].values()} - {"input_layer", "output_layer"})
        res["activations"] = sorted({v.get("params", {}).get("activation") for v in hn["layers"].values()
                                     if v.get("params", {}).get("activation") not in (None, "linear")})
        res["stage"] = "quantize"
        script = "model_optimization_flavor(optimization_level=0, compression_level=0)\n"
        prec = os.environ.get("PREC", "a8_w8")
        res["prec"] = prec
        if prec != "a8_w8":
            names, skipped = supported_layers(path, name, prec, nhwc_calib=np.transpose(calib, (0, 2, 3, 1)))
            res["prec_layers"], res["prec_skipped"] = names, skipped
            if names and isinstance(names[0], tuple):  # per-layer modes (max16)
                by = {}
                for n_, m_ in names:
                    by.setdefault(m_, []).append(n_)
                for m_, ns in by.items():
                    script += f"quantization_param([{', '.join(ns)}], precision_mode={m_})\n"
            elif names:
                script += f"quantization_param([{', '.join(names)}], precision_mode={prec})\n"
        script += os.environ.get("EXTRA_ALLS", "").replace("\\n", "\n")
        runner.load_model_script(script)
        nhwc = lambda a: np.transpose(a, (0, 2, 3, 1))  # noqa: E731
        runner.optimize(nhwc(calib))
        res["stage"] = "emulate"
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            nat = runner.infer(ctx, nhwc(test))
        with runner.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
            be = runner.infer(ctx, nhwc(test))
        nat, be = (np.asarray(a[0] if isinstance(a, list) else a) for a in (nat, be))
        refn = nhwc(ref) if ref.ndim == 4 else ref
        if nat.shape != refn.shape:
            refn = refn.reshape(nat.shape)
        res["sqnr_native_db"] = sqnr(refn, nat)  # parser/translation fidelity (float)
        res["sqnr_hw_db"] = sqnr(refn, be)  # what the chip computes
        runner.save_har(os.path.join(d, "q.har"))
        if compile_hef:
            res["stage"] = "compile"
            hef = runner.compile()
            open(os.path.join(d, "m.hef"), "wb").write(hef)
            res["hef_MB"] = round(len(hef) / 1e6, 2)
        res["stage"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        res["trace"] = traceback.format_exc()[-1500:]
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "trace"}))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(CASES))
    else:
        run(sys.argv[2], sys.argv[3], "--no-compile" not in sys.argv)

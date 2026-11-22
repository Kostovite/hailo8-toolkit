"""Batch-3 probe cases: every remaining ONNX default-domain op, built like probe.py (x -> op graph -> identity conv).

Built in the general env (onnx 1.22 knows opsets up to 27), then compat (onnx_hailo_compat + compat3, verified
against onnxruntime / onnx's reference evaluator), PWL steps where needed, then probe.py in the DFC env.

Usage (general env): python cases3.py list
                     python cases3.py build CASE OUT_DIR   -> OUT_DIR/CASE/{orig,ref,m}.onnx, dom.txt, [fit.json]
"""
import json
import math
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe  # noqa: E402
from probe import C, H, W, G, s  # noqa: E402

probe.CASES["ReduceProd_C"] = (17, "near1") + probe.CASES["ReduceProd_C"][2:]  # product of 16 channels near 1

RNG = np.random.default_rng(3)
N = H * W
CASES3 = {}


def case(name, opset=17, dom="normal", cout=C):
    def deco(f):
        CASES3[name] = (opset, dom, f, cout)
        return f
    return deco


i64 = lambda g, v: g.c(np.array(v, np.int64), np.int64)  # noqa: E731
two = lambda g, x: g.conv(x, C, C, 3, rand=True)  # noqa: E731

# --- element-wise / logic
case("Sum3")(lambda g, x: g.op("Sum", x, two(g, x), x))
case("Mean2")(lambda g, x: g.op("Mean", x, two(g, x)))
case("Swish_op24", opset=24)(lambda g, x: g.op("Swish", x, alpha=1.7))
case("GreaterOrEqual_where")(lambda g, x: g.op("Where", g.op("GreaterOrEqual", x, g.c(s(0.3))), x, g.c(s(0.0))))
case("LessOrEqual_where")(lambda g, x: g.op("Where", g.op("LessOrEqual", x, g.c(s(0.3))), x, g.c(s(0.0))))
case("Not_cast")(lambda g, x: g.op("Mul", x, g.op("Cast", g.op("Not", g.op("Greater", x, g.c(s(0.3)))),
                                                  to=TensorProto.FLOAT)))
case("And_cast")(lambda g, x: g.op("Mul", x, g.op("Cast", g.op("And", g.op("Greater", x, g.c(s(-1.0))),
                                                                 g.op("Less", x, g.c(s(1.0)))), to=TensorProto.FLOAT)))
case("Xor_cast")(lambda g, x: g.op("Mul", x, g.op("Cast", g.op("Xor", g.op("Greater", x, g.c(s(-1.0))),
                                                                 g.op("Greater", x, g.c(s(1.0)))), to=TensorProto.FLOAT)))
for _t in (-1.0, -0.3, 0.3, 1.0, 2.0):  # a compare mask alone, by threshold (And/Xor diagnosis)
    _tn = f"{'m' if _t < 0 else 'p'}{abs(_t):.1f}".replace(".", "p")
    case(f"Greater_mask_{_tn}")(lambda g, x, t=_t: g.op("Mul", x, g.op("Cast", g.op("Greater", x, g.c(s(t))),
                                                                          to=TensorProto.FLOAT)))
    case(f"Less_mask_{_tn}")(lambda g, x, t=_t: g.op("Mul", x, g.op("Cast", g.op("Less", x, g.c(s(t))),
                                                                       to=TensorProto.FLOAT)))
case("IsNaN_where")(lambda g, x: g.op("Where", g.op("IsNaN", x), g.c(s(0.0)), x))
case("IsInf_where")(lambda g, x: g.op("Where", g.op("IsInf", x), g.c(s(0.0)), x))
# --- reductions / pooling
case("GlobalLpPool_p2", dom="normal")(lambda g, x: g.op("GlobalLpPool", x, p=2))
case("GlobalLpPool_p1", dom="normal")(lambda g, x: g.op("GlobalLpPool", x, p=1))
case("ReduceLogSum_C", dom="pos", cout=1)(lambda g, x: g.op("ReduceLogSum", x, axes=[1], keepdims=1))
case("CumProd_W", opset=26, dom="near1")(lambda g, x: g.op("CumProd", x, i64(g, 3)))
case("Trilu_HW")(lambda g, x: g.op("Trilu", x, upper=1))
# --- normalization / attention
case("RMSNorm_op23_C", opset=23)(lambda g, x: g.op("Transpose", g.op(
    "RMSNormalization", g.op("Transpose", x, perm=[0, 2, 3, 1]), g.c(RNG.uniform(0.5, 2, C)), axis=-1, epsilon=1e-6),
    perm=[0, 3, 1, 2]))


def _qkv(g, x, layout):
    q, k, v = (g.conv(x, C, C, 1, rand=True) for _ in range(3))
    if layout == "4d":  # (1, 2, N, 8) like torch bhnd
        f = lambda t: g.op("Transpose", g.op("Reshape", t, i64(g, [1, 2, C // 2, N])), perm=[0, 1, 3, 2])  # noqa: E731
    else:  # (1, N, C) tokens
        f = lambda t: g.op("Transpose", g.op("Reshape", t, i64(g, [1, C, N])), perm=[0, 2, 1])  # noqa: E731
    return f(q), f(k), f(v)


def _attn(g, x, layout="4d", **kw):
    q, k, v = _qkv(g, x, layout)
    if layout == "4d":
        y = g.op("Attention", q, k, v, **kw)
        return g.op("Reshape", g.op("Transpose", y, perm=[0, 1, 3, 2]), i64(g, [1, C, H, W]))
    y = g.op("Attention", q, k, v, q_num_heads=2, kv_num_heads=2, **kw)
    return g.op("Reshape", g.op("Transpose", y, perm=[0, 2, 1]), i64(g, [1, C, H, W]))


case("Attention_op23_4d", opset=23)(lambda g, x: _attn(g, x))
case("Attention_op23_3d", opset=23)(lambda g, x: _attn(g, x, "3d"))
case("Attention_op23_causal", opset=23)(lambda g, x: _attn(g, x, is_causal=1))


def _rope(g, x, inter):
    pos = np.arange(H)[None]
    inv = 1.0 / (10000 ** (np.arange(0, W, 2) / W))
    ang = np.arange(32)[:, None] * inv[None]
    return g.op("RotaryEmbedding", x, g.c(np.cos(ang)), g.c(np.sin(ang)), i64(g, pos), interleaved=inter)


case("RotaryEmbedding_op23", opset=23)(lambda g, x: _rope(g, x, 0))
case("RotaryEmbedding_op23_inter", opset=23)(lambda g, x: _rope(g, x, 1))


def _lin_attn(g, x):
    q, k, v = _qkv(g, x, "3d")
    y = g.op("LinearAttention", q, k, v, q_num_heads=2, kv_num_heads=2, update_rule="linear", n_out=2)[0]
    return g.op("Reshape", g.op("Transpose", y, perm=[0, 2, 1]), i64(g, [1, C, H, W]))


case("LinearAttention_op27", opset=27)(_lin_attn)
# --- indexing
_perm4 = np.array([3, 0, 9, 5])
case("GatherElements_C", cout=4)(lambda g, x: g.op("GatherElements", x, i64(
    g, np.broadcast_to(_perm4.reshape(1, 4, 1, 1), (1, 4, H, W))), axis=1))
case("GatherND_C", cout=4)(lambda g, x: g.op("GatherND", x, i64(g, [[[0, c] for c in _perm4]])))
case("ScatterND_C")(lambda g, x: g.op("ScatterND", x, i64(g, [[[0, c] for c in (4, 5, 6, 7)]]),
                                      g.op("Slice", two(g, x), i64(g, [0]), i64(g, [4]), i64(g, [1]))))
case("ScatterElements_C")(lambda g, x: g.op("ScatterElements", x, i64(g, np.broadcast_to(
    np.arange(4, 8).reshape(1, 4, 1, 1), (1, 4, H, W))), g.op("Slice", two(g, x), i64(g, [0]), i64(g, [4]), i64(g, [1])),
    axis=1))
case("TensorScatter_op24", opset=24)(lambda g, x: g.op("TensorScatter", x, g.op(
    "Slice", two(g, x), i64(g, [0]), i64(g, [4]), i64(g, [2])), i64(g, [8]), axis=2))
case("OneHot_argmax")(lambda g, x: g.op("OneHot", g.op("ArgMax", x, axis=1, keepdims=0), i64(g, C),
                                        g.c(np.array([0.0, 1.0], np.float32)), axis=1))
_keep = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1], bool)
case("Compress_C", cout=int(_keep.sum()))(lambda g, x: g.op("Compress", x, g.c(_keep, bool), axis=1))
case("CenterCropPad_crop", opset=18)(lambda g, x: g.op("CenterCropPad", x, i64(g, [16, 16]), axes=[2, 3]))
case("CenterCropPad_pad", opset=18)(lambda g, x: g.op("CenterCropPad", x, i64(g, [28, 28]), axes=[2, 3]))
case("ReverseSequence_C")(lambda g, x: g.op("ReverseSequence", x, i64(g, [12]), batch_axis=0, time_axis=1))
# --- conv / pooling family
case("Col2Im_k2s2", opset=18, cout=4)(lambda g, x: g.op("Col2Im", g.op("Reshape", x, i64(g, [1, C, N])),
                                                        i64(g, [2 * H, 2 * W]), i64(g, [2, 2]), strides=[2, 2]))
case("MaxUnpool_2")(lambda g, x: g.op("MaxUnpool", *g.op("MaxPool", x, kernel_shape=[2, 2], strides=[2, 2], n_out=2),
                                      kernel_shape=[2, 2], strides=[2, 2]))
case("DeformConv_zero", opset=19)(lambda g, x: g.op("DeformConv", x, g.c(RNG.normal(0, .1, (C, C, 3, 3))),
                                                    g.c(np.zeros((1, 18, H, W))), kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
case("CausalConvWithState_op27", opset=27)(lambda g, x: g.op("Reshape", g.op(
    "CausalConvWithState", g.op("Reshape", x, i64(g, [1, C, N])), g.c(RNG.normal(0, .3, (C, 1, 4))),
    activation="silu", n_out=2)[0], i64(g, [1, C, H, W])))
# --- spectral
case("STFT_power", cout=71)(lambda g, x: g.op("Unsqueeze", g.op("ReduceSumSquare", g.op(
    "STFT", g.op("Reshape", g.op("ReduceMean", x, axes=[1], keepdims=1), i64(g, [1, N, 1])), i64(g, 8),
    g.c(np.hanning(16).astype(np.float32))), axes=[-1], keepdims=0), i64(g, [2])))  # (1, frames, 1, bins)
# --- einsum
case("Einsum_conv")(lambda g, x: g.op("Einsum", x, g.c(RNG.normal(0, .3, (C, C))), equation="bchw,oc->bohw"))
case("Einsum_matmul")(lambda g, x: g.op("Einsum", x, two(g, x), equation="bchw,bchv->bcwv"))


# --- quantized models
def _q(g, t, sc, zp):
    return g.op("QuantizeLinear", t, g.c(np.array(sc, np.float32)), g.c(np.array(zp, np.uint8), np.uint8))


def _dq(g, t, sc, zp, dt=np.uint8):
    return g.op("DequantizeLinear", t, g.c(np.array(sc, np.float32)), g.c(np.array(zp, dt), dt))


_wf = RNG.normal(0, .2, (C, C, 3, 3)).astype(np.float32)
_ws = np.float32(np.abs(_wf).max() / 127)
_wq = np.round(_wf / _ws).astype(np.int8)
case("QDQ_conv")(lambda g, x: _dq(g, _q(g, g.op("Conv", _dq(g, _q(g, x, 0.06, 128), 0.06, 128),
                                                  _dq(g, g.c(_wq, np.int8), _ws, 0, np.int8), kernel_shape=[3, 3],
                                                  pads=[1, 1, 1, 1]), 0.2, 128), 0.2, 128))
case("QLinearConv")(lambda g, x: _dq(g, g.op(
    "QLinearConv", _q(g, x, 0.06, 128), g.c(np.array(0.06, np.float32)), g.c(np.array(128, np.uint8), np.uint8),
    g.c(_wq, np.int8), g.c(np.array(_ws, np.float32)), g.c(np.array(0, np.int8), np.int8),
    g.c(np.array(0.2, np.float32)), g.c(np.array(128, np.uint8), np.uint8), kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    0.2, 128))
_mf = RNG.normal(0, .2, (W, W)).astype(np.float32)
_ms = np.float32(np.abs(_mf).max() / 127)
_mq = np.round(_mf / _ms).astype(np.int8)
case("QLinearMatMul_W")(lambda g, x: _dq(g, g.op(
    "QLinearMatMul", _q(g, x, 0.06, 128), g.c(np.array(0.06, np.float32)), g.c(np.array(128, np.uint8), np.uint8),
    g.c(_mq, np.int8), g.c(np.array(_ms, np.float32)), g.c(np.array(0, np.int8), np.int8),
    g.c(np.array(0.25, np.float32)), g.c(np.array(128, np.uint8), np.uint8)), 0.25, 128))


def _dyn(g, x, conv):
    xq, xs, xz = g.op("DynamicQuantizeLinear", x, n_out=3)
    if conv:
        mi = g.op("ConvInteger", xq, g.c(_wq.astype(np.uint8) if False else (_wq.astype(np.int16) + 128).astype(np.uint8),
                                          np.uint8), xz, g.c(np.array(128, np.uint8), np.uint8),
                  kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        ws = _ws
    else:
        mi = g.op("MatMulInteger", xq, g.c((_mq.astype(np.int16) + 128).astype(np.uint8), np.uint8), xz,
                  g.c(np.array(128, np.uint8), np.uint8))
        ws = _ms
    return g.op("Mul", g.op("Cast", mi, to=TensorProto.FLOAT), g.op("Mul", xs, g.c(np.array(ws, np.float32))))


case("DynQuant_MatMulInteger_W")(lambda g, x: _dyn(g, x, False))
case("DynQuant_ConvInteger")(lambda g, x: _dyn(g, x, True))
case("Hardmax_C_onehot")(lambda g, x: g.op("OneHot", g.op("ArgMax", x, axis=1, keepdims=0), i64(g, C),
                                             g.c(np.array([0.0, 1.0], np.float32)), axis=1))  # = Hardmax(x, 1), exact
# --- ops that fold to constants
case("Folded_misc", opset=18)(lambda g, x: g.op("Add", g.op("Mul", g.op("Dropout", x), g.op(
    "Reshape", g.op("HannWindow", i64(g, W)), i64(g, [1, 1, 1, W]))), g.op("Mul", g.op("CastLike", x, x), g.op(
        "Reshape", g.op("Div", g.op("Cast", g.op("Range", i64(g, 0), i64(g, W), i64(g, 1)), to=TensorProto.FLOAT),
                        g.op("Cast", g.op("Size", g.op("HammingWindow", i64(g, W))), to=TensorProto.FLOAT)),
        i64(g, [1, 1, 1, W])))))
case("EyeLike_mask")(lambda g, x: g.op("Mul", x, g.op("EyeLike", g.c(np.zeros((H, W))))))


def build_orig(name):
    opset, dom, f, cout = CASES3[name] if name in CASES3 else probe.CASES[name]
    g = G()
    y = f(g, "x")
    y = g.conv(y, cout, cout)
    graph = helper.make_graph(g.nodes, name, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, H, W])],
                              [helper.make_tensor_value_info(y, TensorProto.FLOAT, None)], g.inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 10 if opset > 21 else 8
    m = onnx.shape_inference.infer_shapes(probe.fill_defaults(m))
    return m, dom


STEP_OPS = {"Floor", "Ceil", "Round", "Sign"}


def build(name, out):
    import onnx_hailo_compat as CMP
    import pwl_inject as PW
    if name.startswith("R_"):  # a probe.py case, rewritten: probe.py run would rebuild the raw case by its own name
        name = name[2:]
    d = os.path.join(out, "R_" + name if name in probe.CASES else name)
    os.makedirs(d, exist_ok=True)
    for stale in ("fit.json", "calib.npy", "m.onnx", "ref.onnx", "result.json"):
        if os.path.exists(f"{d}/{stale}"):
            os.remove(f"{d}/{stale}")
    m, dom = build_orig(name)
    onnx.save(m, f"{d}/orig.onnx")
    open(f"{d}/dom.txt", "w").write(dom)
    CMP.run(f"{d}/orig.onnx", f"{d}/ref.onnx", CMP.ALL, check=False)
    rng = np.random.default_rng(5)
    x = probe.domain(dom, 4, rng).astype(np.float32)
    a = np.concatenate([CMP.compat3.reference(f"{d}/orig.onnx", {"x": t[None]})[0] for t in x])
    b = np.concatenate([CMP.compat3.reference(f"{d}/ref.onnx", {"x": t[None]})[0] for t in x])
    err = float(probe.sqnr(a, b))
    left = sorted({n.op_type for n in onnx.load(f"{d}/ref.onnx").graph.node})
    info = {"case": name, "rewrite_vs_orig_db": err, "ops_after": left,
            "opset_after": next(o.version for o in onnx.load(f"{d}/ref.onnx").opset_import if o.domain in ("", "ai.onnx"))}
    steps = set(left) & STEP_OPS
    if {"Sign", "Floor", "Abs", "Mul"} <= set(left):  # fmod's Sign * Floor(|t|): one Trunc staircase
        steps.add("Trunc")
    if steps:  # step functions: pwl_inject staircases on the calibrated range (ref.onnx stays the float reference)
        np.save(f"{d}/calib.npy", probe.domain(dom, 16, np.random.default_rng(1)).astype(np.float32))
        PW.fit(f"{d}/ref.onnx", f"{d}/calib.npy", f"{d}/fit.json", steps, 9)
        PW.mark(f"{d}/ref.onnx", f"{d}/fit.json", f"{d}/m.onnx")
        info["pwl"] = sorted(steps)
    else:
        import shutil
        shutil.copy(f"{d}/ref.onnx", f"{d}/m.onnx")
    json.dump(info, open(f"{d}/build.json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(CASES3))
    else:
        build(sys.argv[2], sys.argv[3])

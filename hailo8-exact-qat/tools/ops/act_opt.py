"""Native Hailo activation vs our fitted PWL, on  x -> conv3x3 -> ACT -> identity 1x1 conv  (same weights).

For each activation: (a) native: the ONNX pattern the parser maps to its built-in APU fit; (b) pwl: the activation
replaced by Relu placeholders right after the conv (fused into the conv's APU when monotone), pieces fitted to the
conv output distribution on the calibration set. Bit-exact HSim SQNR vs onnxruntime float, HN layer count, compile.

Usage (DFC env): python act_opt.py ACT OUT_DIR [--pieces 9]
"""
import json
import math
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import pwl_inject as P

C, H, W = 16, 24, 24
ACTS = {  # name: (onnx pattern builder on tensor t, numpy f)
    "Sigmoid": (lambda g, t: g.op("Sigmoid", t), lambda x: 1 / (1 + np.exp(-x))),
    "Tanh": (lambda g, t: g.op("Tanh", t), np.tanh),
    "SiLU": (lambda g, t: g.op("Mul", t, g.op("Sigmoid", t)), lambda x: x / (1 + np.exp(-x))),
    "GELU": (lambda g, t: g.op("Mul", g.op("Mul", t, g.op("Add", g.op("Erf", g.op("Div", t, g.c(np.float32(math.sqrt(2))))),
                                                       g.c(np.float32(1.0)))), g.c(np.float32(0.5))),
             lambda x: 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))),
    "Mish": (lambda g, t: g.op("Mul", t, g.op("Tanh", g.op("Softplus", t))), lambda x: x * np.tanh(np.log1p(np.exp(x)))),
    "HardSwish": (lambda g, t: g.op("HardSwish", t), lambda x: x * np.clip(x + 3, 0, 6) / 6),
    "Softplus": (lambda g, t: g.op("Softplus", t), lambda x: np.log1p(np.exp(x))),
    "ELU": (lambda g, t: g.op("Elu", t, alpha=1.0), lambda x: np.where(x > 0, x, np.exp(x) - 1)),
    "Exp": (lambda g, t: g.op("Exp", t), np.exp),
    "Softsign": (lambda g, t: g.op("Softsign", t), lambda x: x / (1 + np.abs(x))),
    "HardSigmoid": (lambda g, t: g.op("HardSigmoid", t, alpha=0.2, beta=0.5), lambda x: np.clip(0.2 * x + 0.5, 0, 1)),
    "LeakyRelu": (lambda g, t: g.op("LeakyRelu", t, alpha=0.1), lambda x: np.where(x > 0, x, 0.1 * x)),
}


class G:
    def __init__(self):
        self.nodes, self.inits, self.k = [], [], 0

    def c(self, arr):
        self.k += 1
        n = f"c{self.k}"
        self.inits.append(numpy_helper.from_array(np.asarray(arr), n))
        return n

    def op(self, typ, *ins, **attrs):
        self.k += 1
        o = f"t{self.k}"
        self.nodes.append(helper.make_node(typ, list(ins), [o], name=f"{typ}_{self.k}", **attrs))
        return o


def model(act_builder, scale):
    rng = np.random.default_rng(0)
    g = G()
    w = (rng.normal(0, 1 / math.sqrt(C * 9), (C, C, 3, 3)) * scale).astype(np.float32)
    t = g.op("Conv", "x", g.c(w), g.c(rng.normal(0, 0.3, C).astype(np.float32)), kernel_shape=[3, 3], pads=[1] * 4)
    y = act_builder(g, t)
    eye = np.eye(C, dtype=np.float32).reshape(C, C, 1, 1)
    out = g.op("Conv", y, g.c(eye), g.c(np.zeros(C, np.float32)), kernel_shape=[1, 1])
    gr = helper.make_graph(g.nodes, "m", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, H, W])],
                           [helper.make_tensor_value_info(out, TensorProto.FLOAT, None)], g.inits)
    m = helper.make_model(gr, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return onnx.shape_inference.infer_shapes(m), t


def sq(a, b):
    return float(10 * np.log10((a ** 2).sum() / max(((a - b) ** 2).sum(), 1e-30)))


def dfc(path, calib, test, ref, fit_json=None, script=""):
    from hailo_sdk_client import ClientRunner, InferenceContext
    r = ClientRunner(hw_arch="hailo8")
    r.translate_onnx_model(path, "m")
    if fit_json:
        P.apply_pwl(r, fit_json)
    hn = r.get_hn_dict()
    types = [v["type"] for v in hn["layers"].values() if v["type"] not in ("input_layer", "output_layer")]
    acts = [v["params"].get("activation") for v in hn["layers"].values() if "activation" in v.get("params", {})]
    r.load_model_script("model_optimization_flavor(optimization_level=0, compression_level=0)\n" + script)
    nhwc = lambda a: np.transpose(a, (0, 2, 3, 1))  # noqa: E731
    r.optimize(nhwc(calib))
    with r.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
        y = np.asarray(r.infer(ctx, nhwc(test)))
    hef = r.compile()
    return {"sqnr_hw_db": sq(nhwc(ref), y.reshape(nhwc(ref).shape)), "layers": len(types), "types": types,
            "acts": acts, "hef_MB": round(len(hef) / 1e6, 2)}


def main():
    act, out = sys.argv[1], sys.argv[2]
    pieces = int(sys.argv[sys.argv.index("--pieces") + 1]) if "--pieces" in sys.argv else 9
    scale = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 1.0
    d = os.path.join(out, f"{act}_p{pieces}_s{scale:g}" + ("_mono" if "--monotone" in sys.argv else "") + ("_free" if "--free" in sys.argv else ""))
    os.makedirs(d, exist_ok=True)
    build, f = ACTS[act]
    ref_m, tap = model(build, scale)
    onnx.save(ref_m, f"{d}/ref.onnx")
    rng = np.random.default_rng(1)
    calib = rng.normal(0, 1.5, (64, C, H, W)).astype(np.float32)
    test = np.random.default_rng(2).normal(0, 1.5, (8, C, H, W)).astype(np.float32)
    sess = ort.InferenceSession(f"{d}/ref.onnx")
    ref = np.concatenate([sess.run(None, {"x": t[None]})[0] for t in test])
    # conv output samples (the activation's input) on the calibration set
    tm = onnx.load(f"{d}/ref.onnx")
    tm.graph.output.append(helper.make_empty_tensor_value_info(tap))
    ts = ort.InferenceSession(tm.SerializeToString())
    xs = np.concatenate([ts.run(None, {"x": c[None]})[1].ravel() for c in calib])
    if "--free" in sys.argv:  # one fused APU layer, free offsets
        P.UNIFORM = 0.02
        b, sl, of, snr, rng_ = P.fit_free(f, xs, pieces)
        layers = {"G0": (list(map(float, b)), list(map(float, sl)), list(map(float, of)))}
    else:
        fitter = P.fit_monotone if "--monotone" in sys.argv else P.fit_hinges
        b, a, c0, c, snr, rng_ = fitter(f, xs, pieces)
        layers = P.to_layers(b, a, c0, c, rng_[0])
    fit = {"pwl0": {"node": "act", "func": act, "range": rng_, "pieces": pieces, "fit_sqnr_db": snr, "layers": layers}}
    json.dump(fit, open(f"{d}/fit.json", "w"), indent=1)
    # marked model: conv -> placeholders (+ add/sub) -> identity conv
    mm, _ = model(lambda g, t: _placeholders(g, t, layers), scale)
    onnx.save(mm, f"{d}/pwl.onnx")
    res = {"act": act, "pieces": pieces, "scale": scale, "monotone": "--monotone" in sys.argv, "range": rng_, "fit_sqnr_db": snr, "pwl_layers": len(layers)}
    for tag, path, fj in (("native", f"{d}/ref.onnx", None), ("pwl", f"{d}/pwl.onnx", f"{d}/fit.json")):
        try:
            res[tag] = dfc(path, calib, test, ref, fj)
        except Exception as e:  # noqa: BLE001
            res[tag] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    json.dump(res, open(f"{d}/result.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "types"})
                      for k, v in res.items()}))


def _placeholders(g, t, layers):
    parts = {"G": [], "H": []}
    for ln in layers:
        g.k += 1
        o = f"pwl0__{ln}"
        g.nodes.append(helper.make_node("Relu", [t], [o], name=o))
        parts[ln[0]].append(o)

    def tree(ts):
        while len(ts) > 1:
            ts = [g.op("Add", ts[i], ts[i + 1]) for i in range(0, len(ts) - 1, 2)] + (ts[-1:] if len(ts) % 2 else [])
        return ts[0] if ts else None
    gg, hh = tree(parts["G"]), tree(parts["H"])
    return gg if hh is None else g.op("Sub", gg, hh)


if __name__ == "__main__":
    main()

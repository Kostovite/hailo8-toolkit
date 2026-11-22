"""Black-box probing of the Hailo-8 compiler: tiny parametric ONNX nets -> parse, level-0 quantize, single-context
compile -> the compiler's own view of each node (HEF `mapping`): resources, cycles, latency, defuse decisions and the
hardware rule strings (se_formula). One case per process; run.py fans cases out over cores.

Usage (DFC env): python sweep.py list FAMILY            -> case names
                 python sweep.py run CASE OUT_DIR [ALLS] -> OUT_DIR/CASE/{m.onnx,m.hef,result.json}
CASE = family:k=v,k=v (e.g. conv:c=64,o=64,h=80,w=80,k=3,s=1)
"""
import json
import os
import re
import sys
import time
import traceback

import numpy as np
import onnx
from onnx import TensorProto, helper as oh, numpy_helper as nh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEF = dict(c=64, o=64, h=80, w=80, k=3, s=1, g=1, d=1, n=1, q=8, pre=0, post=0, act="relu")
ACTS = {"relu": "Relu", "sigmoid": "Sigmoid", "silu": None, "none": None, "lrelu": "LeakyRelu", "tanh": "Tanh"}


def parse_case(case):
    if case.startswith("@"):  # a prebuilt pseudo-model: @DIR (DIR/m.onnx), params from its input shape
        m = onnx.load(case[1:] + "/m.onnx")
        dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
        return "file", dict(DEF, c=dims[1], h=dims[2], w=dims[3], path=case[1:])
    fam, _, rest = case.partition(":")
    p = dict(DEF)
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        p[k] = v if k == "act" else int(v)
    return fam, p


class G:
    def __init__(self):
        self.nodes, self.inits, self.n = [], [], 0
        self.rng = np.random.default_rng(0)

    def name(self, base):
        self.n += 1
        return f"{base}{self.n}"

    def const(self, a):
        nm = self.name("w")
        self.inits.append(nh.from_array(np.asarray(a, np.float32), nm))
        return nm

    def op(self, typ, ins, **attrs):
        out = self.name(typ.lower())
        self.nodes.append(oh.make_node(typ, ins, [out], **attrs))
        return out

    def conv(self, x, cin, cout, k, s=1, g=1, d=1, act="relu"):
        w = self.rng.normal(0, 1 / np.sqrt(cin // g * k * k), (cout, cin // g, k, k))
        y = self.op("Conv", [x, self.const(w), self.const(self.rng.normal(0, .1, cout))], kernel_shape=[k, k],
                    strides=[s, s], pads=[(k - 1) * d // 2] * 4, group=g, dilations=[d, d])
        return self.activation(y, act)

    def activation(self, y, act):
        if act == "silu":
            return self.op("Mul", [y, self.op("Sigmoid", [y])])
        if act == "lrelu":
            return self.op("LeakyRelu", [y], alpha=0.1)
        return self.op(ACTS[act], [y]) if ACTS.get(act) else y


def build(fam, p):
    """Family -> (onnx model, input shape). Each family isolates one op; `n` repeats it as a chain."""
    g = G()
    c, o, h, w, k, s = p["c"], p["o"], p["h"], p["w"], p["k"], p["s"]
    x = "x"
    for _ in range(p["pre"]):  # boundary probes: convs in front of the op ...
        x = g.conv(x, c, c, 3)
    if fam == "conv":  # n chained convs (c -> o -> o ...)
        cin = c
        for _ in range(p["n"]):
            x = g.conv(x, cin, o, k, s, 1, p["d"], p["act"])
            cin = o
    elif fam == "dw":  # depthwise kxk
        for _ in range(p["n"]):
            x = g.conv(x, c, c, k, s, c, p["d"], p["act"])
    elif fam == "gconv":
        x = g.conv(x, c, o, k, s, p["g"], 1, p["act"])
    elif fam == "maxpool":
        x = g.op("MaxPool", [g.conv(x, c, c, 1)], kernel_shape=[k, k], strides=[s, s], pads=[(k - 1) // 2] * 4)
    elif fam == "avgpool":
        x = g.op("AveragePool", [g.conv(x, c, c, 1)], kernel_shape=[k, k], strides=[s, s], pads=[(k - 1) // 2] * 4)
    elif fam == "add":  # two branches + elementwise add
        x = g.op("Add", [g.conv(x, c, o, k), g.conv(x, c, o, 1)])
    elif fam == "concat":
        x = g.op("Concat", [g.conv(x, c, o, k), g.conv(x, c, o, 1)], axis=1)
    elif fam == "resize":  # nearest upsample xs
        y = g.conv(x, c, o, 1)
        x = g.op("Resize", [y, "", g.const([1, 1, s, s])], mode="nearest")
    elif fam == "act":  # standalone activation after a 1x1
        x = g.activation(g.conv(x, c, c, 1, act="none"), p["act"])
    elif fam == "matmul":  # attention-style: (hw x o) @ (o x hw) -> reshaped back to an image, then a 1x1
        y1 = g.conv(x, c, o, 1)
        y2 = g.conv(x, c, o, 1)
        a = g.op("Reshape", [y1, g.const_i([1, o, h * w])])
        b = g.op("Reshape", [y2, g.const_i([1, o, h * w])])
        mm = g.op("MatMul", [g.op("Transpose", [a], perm=[0, 2, 1]), b])  # (1, hw, hw)
        x = g.conv(g.op("Reshape", [g.op("Transpose", [mm], perm=[0, 2, 1]), g.const_i([1, h * w, h, w])]),
                   h * w, o, 1)
    elif fam == "bilinear":
        x = g.op("Resize", [g.conv(x, c, o, 1), "", g.const([1, 1, s, s])], mode="linear")
    elif fam in ("mul", "sub", "max", "min"):  # elementwise binary of two branches
        x = g.op(fam.capitalize(), [g.conv(x, c, o, k), g.conv(x, c, o, 1)])
    elif fam in ("reducemean", "reducemax", "reducesum"):  # over channels (axis 1) or spatial (axes 2,3: s=0)
        axes = [1] if p["s"] == 1 else [2, 3]
        y = g.conv(x, c, o, 1)
        x = (g.op("ReduceSum", [y, g.const_i(axes)], keepdims=1) if fam == "reducesum" else  # opset 17: axes attr
             g.op({"reducemean": "ReduceMean", "reducemax": "ReduceMax"}[fam], [y], axes=axes, keepdims=1))
    elif fam == "gap":  # global average pool + dense classifier
        x = g.op("Gemm", [g.op("Flatten", [g.op("GlobalAveragePool", [g.conv(x, c, c, 1)])]),
                          g.const(g.rng.normal(0, .1, (o, c))), g.const(np.zeros(o))], transB=1)
    elif fam == "softmax":  # over channels
        x = g.op("Softmax", [g.conv(x, c, o, 1)], axis=1)
    elif fam == "slice":  # take the first half of the channels
        x = g.conv(g.op("Slice", [g.conv(x, c, o, 1), g.const_i([0]), g.const_i([o // 2]), g.const_i([1])]),
                   o // 2, o, 1)
    elif fam == "split":  # split channels in 2, conv each, concat
        y = g.conv(x, c, o, 1)
        a = g.op("Slice", [y, g.const_i([0]), g.const_i([o // 2]), g.const_i([1])])
        b = g.op("Slice", [y, g.const_i([o // 2]), g.const_i([o]), g.const_i([1])])
        x = g.op("Concat", [g.conv(a, o // 2, o // 2, k), g.conv(b, o // 2, o // 2, k)], axis=1)
    elif fam == "d2s":
        x = g.op("DepthToSpace", [g.conv(x, c, o * 4, 1)], blocksize=2, mode="DCR")
    elif fam == "s2d":
        x = g.conv(g.op("SpaceToDepth", [g.conv(x, c, o, 1)], blocksize=2), o * 4, o, 1)
    elif fam == "shuffle":  # channel shuffle (ShuffleNet): reshape-transpose-reshape
        y = g.conv(x, c, o, 1)
        y = g.op("Reshape", [y, g.const_i([1, 2, o // 2, h, w])])
        y = g.op("Transpose", [y], perm=[0, 2, 1, 3, 4])
        x = g.conv(g.op("Reshape", [y, g.const_i([1, o, h, w])]), o, o, 1)
    elif fam == "convt":  # transposed conv upsample x s
        w_ = g.rng.normal(0, 1 / np.sqrt(c * k * k), (c, o, k, k))
        x = g.op("ConvTranspose", [g.conv(x, c, c, 1), g.const(w_)], kernel_shape=[k, k], strides=[s, s])
    elif fam == "pad":
        x = g.conv(g.op("Pad", [g.conv(x, c, o, 1), g.const_i([0, 0, 1, 1, 0, 0, 1, 1])]), o, o, 3)
    elif fam == "block":  # C2f-like: 1x1 -> split -> n bottlenecks (3x3,3x3,+) -> concat -> 1x1
        y = g.conv(x, c, o, 1)
        a = g.op("Slice", [y, g.const_i([0]), g.const_i([o // 2]), g.const_i([1])])
        b = g.op("Slice", [y, g.const_i([o // 2]), g.const_i([o]), g.const_i([1])])
        outs = [a, b]
        for _ in range(p["n"]):
            b = g.op("Add", [b, g.conv(g.conv(b, o // 2, o // 2, 3), o // 2, o // 2, 3)])
            outs.append(b)
        x = g.conv(g.op("Concat", outs, axis=1), o // 2 * len(outs), o, 1)
    elif fam == "skip":  # residual held across n kxk convs: the skip tensor must be buffered for n layers' latency
        a = g.conv(x, c, c, 1)
        b = a
        for _ in range(p["n"]):
            b = g.conv(b, c, c, k, 1, 1, p["d"], p["act"])
        x = g.op("Add", [a, b])
    elif fam == "fan":  # one tensor -> n parallel 1x1 branches -> concat (fan-out / concat-input limits)
        a = g.conv(x, c, c, 1)
        x = g.op("Concat", [g.conv(a, c, o, 1) for _ in range(p["n"])], axis=1)
    elif fam in ("layernorm", "instnorm", "groupnorm"):  # normalisation over C / HW per channel / channel groups
        y = g.conv(x, c, o, 1)
        if fam == "layernorm":
            t = g.op("Transpose", [y], perm=[0, 2, 3, 1])
            t = g.op("LayerNormalization", [t, g.const(g.rng.uniform(.5, 1.5, o)), g.const(g.rng.normal(0, .1, o))],
                     axis=-1, epsilon=1e-5)
            x = g.op("Transpose", [t], perm=[0, 3, 1, 2])
        else:
            gr = o if fam == "instnorm" else max(1, p["g"])
            t = g.op("Reshape", [y, g.const_i([1, gr, -1])])
            m_ = g.op("ReduceMean", [t], axes=[2], keepdims=1)
            dv = g.op("Sub", [t, m_])
            var = g.op("ReduceMean", [g.op("Mul", [dv, dv])], axes=[2], keepdims=1)
            t = g.op("Div", [dv, g.op("Sqrt", [g.op("Add", [var, g.const([1e-5])])])])
            x = g.op("Add", [g.op("Mul", [g.op("Reshape", [t, g.const_i([1, o, h, w])]),
                                          g.const(g.rng.uniform(.5, 1.5, (1, o, 1, 1)))]),
                             g.const(g.rng.normal(0, .1, (1, o, 1, 1)))])
    elif fam == "bn":  # standalone BatchNorm behind an add (not foldable into a conv)
        y = g.op("Add", [g.conv(x, c, o, 1), g.conv(x, c, o, 3)])
        x = g.op("BatchNormalization", [y, g.const(g.rng.uniform(.5, 1.5, o)), g.const(g.rng.normal(0, .1, o)),
                                        g.const(g.rng.normal(0, .1, o)), g.const(g.rng.uniform(.5, 1.5, o))],
                 epsilon=1e-5)
    elif fam == "div":  # tensor / tensor (positive denominator)
        x = g.op("Div", [g.conv(x, c, o, k), g.op("Add", [g.conv(x, c, o, 1, act="sigmoid"), g.const([.5])])])
    elif fam == "l2norm":  # x / ||x||_C
        y = g.conv(x, c, o, 1)
        nrm = g.op("Sqrt", [g.op("Add", [g.op("ReduceSum", [g.op("Mul", [y, y]), g.const_i([1])], keepdims=1),
                                         g.const([1e-6])])])
        x = g.op("Div", [y, nrm])
    elif fam == "pow2":
        x = g.op("Mul", [g.conv(x, c, o, 1, act="none"), g.conv(x, c, o, 1, act="none")]) if p["n"] == 2 else \
            g.op("Pow", [g.conv(x, c, o, 1, act="none"), g.const([2.0])])
    elif fam == "argmax":
        x = g.op("Cast", [g.op("ArgMax", [g.conv(x, c, o, 1)], axis=1, keepdims=1)], to=TensorProto.FLOAT)
    elif fam == "tile":  # repeat along W
        x = g.op("Tile", [g.conv(x, c, o, 1), g.const_i([1, 1, 1, 2])])
    elif fam == "avgpool_nc":  # padded avg pool excluding the pad (count_include_pad=0 -> const divisor map)
        x = g.op("AveragePool", [g.conv(x, c, c, 1)], kernel_shape=[k, k], strides=[s, s], pads=[(k - 1) // 2] * 4,
                 count_include_pad=0)
    elif fam == "softmax_w":
        x = g.op("Softmax", [g.conv(x, c, o, 1)], axis=3)
    elif fam == "vit":  # tokens: LN -> qkv -> attention -> proj -> + residual -> LN -> MLP -> + residual
        y = g.conv(x, c, o, 1)
        t = g.op("Transpose", [g.op("Reshape", [y, g.const_i([1, o, h * w])])], perm=[0, 2, 1])  # (1, N, o)

        def ln(z):
            return g.op("LayerNormalization", [z, g.const(np.ones(o)), g.const(np.zeros(o))], axis=-1)

        def lin(z, a, b_):
            return g.op("Add", [g.op("MatMul", [z, g.const(g.rng.normal(0, 1 / np.sqrt(a), (a, b_)))]),
                                g.const(np.zeros(b_))])
        qkv = lin(ln(t), o, 3 * o)
        q, k_, v = [g.op("Slice", [qkv, g.const_i([i * o]), g.const_i([(i + 1) * o]), g.const_i([2])])
                    for i in range(3)]
        a = g.op("Softmax", [g.op("Mul", [g.op("MatMul", [q, g.op("Transpose", [k_], perm=[0, 2, 1])]),
                                          g.const([o ** -.5])])], axis=-1)
        t = g.op("Add", [t, lin(g.op("MatMul", [a, v]), o, o)])
        u = lin(ln(t), o, 2 * o)
        t = g.op("Add", [t, lin(g.op("Relu", [u]), 2 * o, o)])
        x = g.op("Reshape", [g.op("Transpose", [t], perm=[0, 2, 1]), g.const_i([1, o, h, w])])
    else:
        raise ValueError(fam)
    inp = oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, c, h, w])
    if p["post"]:  # ... and behind it (channel count of the op output from one run)
        import onnxruntime as ort
        tmp = oh.make_model(oh.make_graph(g.nodes, "probe", [inp], [oh.make_tensor_value_info(x, TensorProto.FLOAT,
                                                                                            None)], g.inits),
                            opset_imports=[oh.make_opsetid("", 17)])
        tmp.ir_version = 8
        co = ort.InferenceSession(tmp.SerializeToString()).run(None, {"x": np.zeros((1, c, h, w), np.float32)})[0]
        if co.ndim == 4:
            for _ in range(p["post"]):
                x = g.conv(x, co.shape[1], co.shape[1], 3)

    out = oh.make_tensor_value_info(x, TensorProto.FLOAT, None)
    m = oh.make_model(oh.make_graph(g.nodes, "probe", [inp], [out], g.inits), opset_imports=[oh.make_opsetid("", 17)])
    m.ir_version = 8
    import onnxruntime as ort  # output shape: run it once (inference leaves graph outputs untyped)
    y = ort.InferenceSession(m.SerializeToString()).run(None, {"x": np.zeros((1, c, h, w), np.float32)})[0]
    m.graph.output[0].CopyFrom(oh.make_tensor_value_info(x, TensorProto.FLOAT, list(y.shape)))
    return m, (c, h, w)


def _const_i(self, a):
    nm = self.name("s")
    self.inits.append(nh.from_array(np.asarray(a, np.int64), nm))
    return nm


G.const_i = _const_i

NUM = re.compile(r"\d+")


def harvest(hef_path, alls_text):
    """The compiler's per-node view from the HEF + its defuse decisions from the auto script."""
    from google.protobuf.json_format import MessageToDict
    from ighw import load
    gph = load(hef_path)
    nodes = {n.index: n for n in gph.network_graph.nodes}
    out = {"bottleneck_fps": gph.bottleneck_fps, "nodes": []}
    for cn in gph.contexts_graph.nodes:
        ng = cn.network_graph
        byidx = {n.index: n for n in ng.nodes}
        for r in ng.resources_list:
            n = byidx.get(r.layer_index)
            if n is None:
                continue
            rd = MessageToDict(r, preserving_proto_field_name=True)
            for k in ("buffers", "min_buffers", "max_buffers"):
                rd.pop(k, None)
            ap = MessageToDict(n.allocation_params, preserving_proto_field_name=True)
            out["nodes"].append({"name": n.name, "type": n.DESCRIPTOR.fields_by_name["type"].enum_type
                                 .values_by_number[n.type].name, "in": [[s.height, s.width, s.features] for s in n.input_shapes],
                                 "out": [[s.height, s.width, s.features] for s in n.output_shapes],
                                 "defuse": MessageToDict(n.defuse_params, preserving_proto_field_name=True),
                                 "alloc": ap, "res": rd, "context": cn.context.context_name})
    prof = {}
    for c in gph.profiler_data.context:
        for L in c.layers:
            prof[L.layer_name] = [[t.start_time, t.end_time] for t in L.time]
    out["profile"] = prof
    rules = set()

    def walk(f):
        if f.explanation:
            rules.add(NUM.sub("#", f.explanation.replace(f.layer_name, "<L>") if f.layer_name else f.explanation))
        for s_ in f.formulas:
            walk(s_)
    for f in gph.se_formula:
        walk(f)
    out["rules"] = sorted({re.sub(r"\b(in|at node|for layer|set at node) [\w-]+$", r"\1 <L>", r) for r in rules})
    out["defuse_lines"] = [ln for ln in alls_text.splitlines() if " = defuse(" in ln]
    return out


def run(case, out_dir, extra=""):
    from hailo_sdk_client import ClientRunner
    if os.environ.get("SMX_HEADROOM"):  # softmax-denominator headroom (RULES.md section 26)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "turbo"))
        import smx_headroom
        smx_headroom.install(os.environ["SMX_HEADROOM"])
    fam, p = parse_case(case)
    d = os.path.abspath(os.path.join(out_dir, os.path.basename(case[1:]) if case.startswith("@") else
                                     re.sub(r"[^\w=,.-]", "_", case)))
    os.makedirs(d, exist_ok=True)
    if fam == "file":
        p["path"] = os.path.abspath(p["path"])
    os.chdir(d)  # private cwd: parallel compiles in one cwd clobber errors.pb / rotating logs
    res = {"case": case, "family": fam, "params": p, "stage": "build"}
    t0 = time.time()
    try:
        if fam == "file":
            m, (c, h, w) = onnx.load(p["path"] + "/m.onnx"), (p["c"], p["h"], p["w"])
        else:
            m, (c, h, w) = build(fam, p)
        onnx.save(m, os.path.join(d, "m.onnx"))
        res["stage"] = "parse"
        r = ClientRunner(hw_arch="hailo8")
        r.translate_onnx_model(os.path.join(d, "m.onnx"), "probe")
        res["hn_types"] = sorted({v["type"] for v in r.get_hn_dict()["layers"].values()})
        res["stage"] = "quantize"
        script = "model_optimization_flavor(optimization_level=0, compression_level=0)\n"
        if p["q"] == 16:  # every computing layer 16-bit
            ls = [k_.split("/")[-1] for k_, v in r.get_hn_dict()["layers"].items()
                  if v["type"] not in ("input_layer", "output_layer")]
            script += f"quantization_param([{', '.join(ls)}], precision_mode=a16_w16)\n"
        calib = np.random.default_rng(1).normal(0, 1, (8, h, w, c)).astype(np.float32)
        if fam == "file" and os.path.exists(os.path.join(p["path"], "calib.npy")):  # the model's own inputs (NCHW)
            calib = np.load(os.path.join(p["path"], "calib.npy")).transpose(0, 2, 3, 1).astype(np.float32)
        for _ in range(4):  # the quantizer names its own fix (e.g. matmul shift delta -> force_range_in): apply, retry
            try:
                r.load_model_script(script + extra)
                r.optimize(calib)
                break
            except Exception as e:  # noqa: BLE001
                fix = re.findall(r"(quantization_param\(\[[^\]]+\], force_range_in=\[[^\]]+\], "
                                 r"force_range_index=\[[^\]]+\]\))", str(e))
                if not fix:
                    raise
                cmd = fix[-1]  # the suggestion can violate the validator's own "0 must be in range" (DFC bug):
                rng = [float(x) for x in re.search(r"force_range_in=\[([^\]]+)\]", cmd).group(1).split(",")]
                rng = [v for lo, hi in zip(rng[::2], rng[1::2]) for v in (min(lo, 0.0), max(hi, 0.0))]
                cmd = re.sub(r"force_range_in=\[[^\]]+\]", "force_range_in=[" + ", ".join(f"{v:.6g}" for v in rng)
                             + "]", cmd)
                script += cmd + "\n"
                res.setdefault("quant_fixes", []).append(cmd)
                r = ClientRunner(hw_arch="hailo8")
                r.translate_onnx_model(os.path.join(d, "m.onnx"), "probe")
        r.save_har(os.path.join(d, "q.har"))  # quantized, uncompiled: input of the external pipeline (bnd.py)
        res["stage"] = "compile"
        if os.environ.get("SWEEP_NO_STOCK"):  # front end only: q.har for the own compiler, no stock compile
            res["stage"] = "quantized"
            raise StopIteration
        t = time.time()
        hef = r.compile()
        res["compile_s"] = round(time.time() - t, 1)
        open(os.path.join(d, "m.hef"), "wb").write(hef)
        r.save_har(os.path.join(d, "c.har"))
        import tarfile
        with tarfile.open(os.path.join(d, "c.har")) as tf:
            nm = next((x for x in tf.getnames() if x.endswith(".auto.alls")), None)
            alls = tf.extractfile(nm).read().decode() if nm else ""
        open(os.path.join(d, "auto.alls"), "w").write(alls)
        res.update(harvest(os.path.join(d, "m.hef"), alls))
        res["stage"] = "ok"
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        res["trace"] = traceback.format_exc()[-1500:]
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(d, "result.json"), "w"))
    print(json.dumps({k: res.get(k) for k in ("case", "stage", "bottleneck_fps", "compile_s", "error")}))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], open(sys.argv[4]).read() if len(sys.argv) > 4 else "")

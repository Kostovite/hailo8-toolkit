"""Exact Hailo-8 integer executor for YOLO26-style .hn graphs (conv/dw/conv-add, ew_add/sub, C2PSA attention:
matmul, reduce_max, reduce_sum, ew_mult, standalone activation, spatial_reshape), on GPU.

Parameters come from a capture.py ALL_OPS=1 run: per-layer conv records (<layer>.npz) and per-op records
(_all_ops.pkl). Every recorded layer output is checked as the graph runs (fed only by this executor upstream).

Usage: python graph26.py HN CAPTURE_DIR IMG [device]
"""
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

import intops as I
from graph import conv_params, maxpool, to_t


def _t(a, dev):
    return torch.as_tensor(np.asarray(a, np.float64), device=dev)


def nchw(a, dev):
    return _t(a, dev).permute(0, 3, 1, 2)


class Params:
    def __init__(self, cap_dir, dev):
        self.dev = dev
        self.conv = {os.path.basename(f)[:-4].replace("__", "/"): np.load(f) for f in glob.glob(os.path.join(cap_dir, "*.npz"))}
        with open(os.path.join(cap_dir, "_all_ops_np.pkl"), "rb") as fh:
            self.ops = pickle.load(fh)

    def op(self, layer, name):
        return self.ops[f"{layer}/{name}"]

    def op_by_type(self, layer, typ):
        hits = [v for k, v in self.ops.items() if k.startswith(layer + "/") and v["type"] == typ]
        assert len(hits) == 1, (layer, typ, len(hits))
        return hits[0]

    def ops_by_type(self, layer, typ):
        return [v for k, v in sorted(self.ops.items()) if k.startswith(layer + "/") and v["type"] == typ]


def _out_bits(P, layers, name):
    """Output bit width of layer `name`, walking up through layers without an APU (reshapes, format conversions)."""
    for _ in range(16):
        try:
            return int(P.op_by_type(name, "ActivationOp")["out_bits"])
        except Exception:  # noqa: BLE001
            ins = layers.get(name, {}).get("input", [])
            if not ins:
                return None
            name = ins[0]
    return None


def lossy_of(rec, prefix="outq"):
    return (int(rec[f"{prefix}_bits"]), bool(rec[f"{prefix}_signed"]), bool(rec[f"{prefix}_wrap"]))


def apu_of(rec, x, dev):
    return I.apu(x, _t(rec["thr"], dev), _t(rec["mant"], dev), _t(rec["shift"], dev), _t(rec["offs"], dev),
                 int(rec["final_shift"]), bool(rec["mant_signed"]), lossy_of(rec, "clip1"), lossy_of(rec, "clip2"),
                 lossy_of(rec, "out"))


def bias_of(rec, x, dev):
    b = _t(rec["bias"], dev).reshape(1, -1, 1, 1) if np.asarray(rec["bias"]).size > 1 else _t(rec["bias"], dev)
    return I.hw_lossy(x + b, *lossy_of(rec))


def mock_of(rec, x, dev):
    kq = _t(rec["kernel_q"], dev)
    kq = kq.reshape(1, -1, 1, 1) if kq.numel() > 1 else kq
    v = I.apu_round_shift(x.long() * kq.long(), torch.full(x.shape, int(rec["pre_acc_shift"]), dtype=torch.long,
                                                          device=dev), 3, True)
    return I.hw_lossy(v, *lossy_of(rec)).to(torch.float64)


def layer_ref(P, n, dev):
    """HSim output of layer n: its last recorded atomic op (records are in execution order)."""
    recs = [v for k, v in P.ops.items() if k.startswith(n + "/") and "out0" in v]
    if not recs:
        return None
    o = np.asarray(recs[-1]["out0"])
    return nchw(o, dev) if o.ndim == 4 else _t(o, dev)


class _Rec(dict):
    """A writable stand-in for an NpzFile conv record."""
    @property
    def files(self):
        return list(self.keys())


def silicon_weights_operands(P, layers):
    """Silicon rule for dynamic-matmul weights operands (chip-verified bit-exact, probe/runs/a1_exact): the quantizer
    forces an 8-bit producer feeding a matmul's 2nd input to SIGNED (get_signed_output on LAYER_OUT_WEIGHTS), but when
    the producer's function is non-negative (ReLU-like) the chip runs its APU output stage UNSIGNED (0..255), and the
    matmul then reads the bytes as int8 (intops x1_bits). Rewrites those producers' records; returns their names."""
    if I.SDK_MATMUL:
        return []
    hit = []
    for m, L in layers.items():
        if L["type"] != "matmul" or len(L["input"]) < 2:
            continue
        n = L["input"][1]
        for _ in range(16):  # walk up through reshapes / format conversions to the layer with the APU
            if any(k.startswith(n + "/") and v.get("type") == "ActivationOp" for k, v in P.ops.items()):
                break
            n = layers[n]["input"][0] if layers.get(n, {}).get("input") else None
            if n is None:
                break
        if n is None:
            continue
        ra = P.op_by_type(n, "ActivationOp")
        if int(ra.get("out_bits", 0)) != 8 or not bool(ra.get("out_signed", False)):
            continue
        thr = np.asarray(ra["thr"], np.float64)
        lo, hi = max(float(thr.min()), -32768.0), min(float(thr.max()), 32767.0)
        a = torch.linspace(lo, hi, 4097, dtype=torch.float64).reshape(1, 1, 1, -1).repeat(1, thr.shape[1] if thr.ndim == 2 else 1, 1, 1)
        y = I.apu(a, _t(ra["thr"], "cpu"), _t(ra["mant"], "cpu"), _t(ra["shift"], "cpu"), _t(ra["offs"], "cpu"),
                  int(ra["final_shift"]), bool(ra["mant_signed"]), lossy_of(ra, "clip1"), lossy_of(ra, "clip2"),
                  lossy_of(ra, "out"))
        if float(y.min()) < 0:
            continue
        for k, v in P.ops.items():
            if k.startswith(n + "/") and v.get("type") == "ActivationOp":
                v["out_signed"] = False; v["clip2_signed"] = False
        if n in P.conv:
            d = _Rec({k: P.conv[n][k] for k in P.conv[n].files})
            for k in ("out_signed", "clip2_signed"):
                if k in d:
                    d[k] = np.array(False)
            P.conv[n] = d
        hit.append(n)
    return hit


def run(hn, P, img, dev, check=True, ints=None, cont=False):
    """cont=True (coverage mode): an unsupported or inexact layer is recorded and replaced by its HSim output,
    so every downstream layer is still checked."""
    layers = hn["layers"]
    silicon_weights_operands(P, layers)
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for i in layers[n]["input"]:
            visit(i)
        order.append(n)

    for n in layers:
        visit(n)
    out, report = {}, []
    free = not check and os.environ.get("G26_KEEP_ALL") != "1"  # inference: drop a tensor after its last consumer
    if free:
        keep = {i for v in layers.values() if v["type"] == "output_layer" for i in v["input"]}
        # a host postprocess (NMS) is skipped: the chip tensors that feed it are the real outputs
        keep |= {i for v in layers.values() if v["type"] == "postprocess" for i in v["input"]}
        last = {}
        for k, m in enumerate(order):
            for i in layers[m]["input"]:
                last[i] = k
    for k_step, n in enumerate(order):
        if free and k_step:
            for i in layers[order[k_step - 1]]["input"]:
                if last.get(i) == k_step - 1 and i not in keep:
                    out.pop(i, None)
        L = layers[n]
        t, prm = L["type"], L.get("params", {})
        if t in ("output_layer", "postprocess"):  # postprocess (NMS) runs on the host, not the chip
            continue
        try:
            ins = I.layer_inputs(out, layers, n)
        except KeyError:
            if not cont:
                raise
            report.append((n, t, -3, 0))  # blocked: an input could not be produced
            continue
        try:
            try:
                y = _dispatch(n, L, t, prm, ins, P, layers, img, dev, ints)
            except Exception as e0:  # noqa: BLE001  a dedicated handler that does not fit this layer variant
                try:
                    y = _chain(n, L, ins, P, layers, dev)  # generic: the layer's captured atomic-op chain
                except NotImplementedError as e1:
                    raise NotImplementedError(f"{type(e0).__name__}: {str(e0)[:80]} | chain: {e1}") from None
        except Exception as e:  # noqa: BLE001
            if not cont:
                raise
            print(f"UNSUPPORTED {n} ({t}): {type(e).__name__}: {str(e)[:120]}", flush=True)
            report.append((n, t, -2, 0))
            ref = layer_ref(P, n, dev)
            if ref is not None:
                out[n] = ref
            continue
        if t == "const_input" and torch.is_tensor(y) and img is not None and y.shape[0] == 1 and len(img) > 1:
            y = y.expand(len(img), *y.shape[1:])  # constants (class token, ...) are per frame: broadcast to the batch
        out[n] = y
        if check and isinstance(y, list):  # multi-output: slice k vs its concat_op_k / output passthrough record
            outp = _out_ops([(k_, v_) for k_, v_ in P.ops.items() if k_.rsplit("/", 1)[0] == n], L)
            for k, yk in enumerate(y):
                rk = outp[k][1] if outp else P.ops.get(f"{n}/concat_op_{k}")
                if rk is not None and "out0" in rk:
                    rk = nchw(rk["out0"], dev)
                    bad = int((yk != rk).sum()) if rk.shape == yk.shape else -1
                    report.append((n, t, bad, rk.numel()))
                    if bad:
                        print(f"MISMATCH {n} ({t}) slice {k}: {bad}/{rk.numel()}", flush=True)
            continue
        if check:
            ref = None
            if n in P.conv and "act_out" in P.conv[n].files:
                ref = nchw(P.conv[n]["act_out"], dev)
            elif cont:  # the layer's output is its LAST atomic op (a slice layer starts with a passthru of its input)
                ref = layer_ref(P, n, dev)
            else:
                po = [v for k, v in P.ops.items() if k.startswith(n + "/") and v["type"] == "PassthruOp"]
                if len(po) == 1 and "out0" in po[0]:
                    ref = nchw(po[0]["out0"], dev)
            if ref is not None:
                if ref.shape != y.shape:
                    print(f"SHAPE {n} ({t}) got {tuple(y.shape)} ref {tuple(ref.shape)}", flush=True)
                    report.append((n, t, -1, ref.numel()))
                    if cont:
                        out[n] = ref
                    continue
                bad = int((y != ref).sum())
                report.append((n, t, bad, ref.numel()))
                if bad:
                    print(f"MISMATCH {n} ({t}): {bad}/{ref.numel()} max {float((y - ref).abs().max()):.0f}", flush=True)
                    if cont:
                        out[n] = ref
            elif cont:
                report.append((n, t, -4, 0))  # ran, but no HSim record to compare with
    return out, report


_CHAINS = {}


def _out_ops(recs, layer):
    """The per-output-edge ops of a multi-output layer (output_passthough_op1..N), in output order; else []."""
    if len(layer.get("output", [])) < 2:
        return []
    o = [(k, v) for k, v in recs if re.search(r"output_pass\w*?(\d+)$", k)]
    o.sort(key=lambda kv: int(re.search(r"(\d+)$", kv[0]).group(1)))
    return o if len(o) == len(layer["output"]) else []


def build_chain(n, L, P, layers):
    """atomic.Chain for HN layer n from its captured ops (shared by graph26 and qatfast)."""
    import atomic as AT
    recs = [(k, v) for k, v in P.ops.items() if k.rsplit("/", 1)[0] == n]
    if not recs:
        raise NotImplementedError(f"{n}: no captured ops")
    refs = []
    for i in L["input"]:  # reference layer inputs: the producers' captured outputs (per output edge)
        pr = [(k, v) for k, v in P.ops.items() if k.rsplit("/", 1)[0] == i and "out0" in v]
        outp = _out_ops(pr, layers[i])
        if outp:
            k_ = [o for o in layers[i]["output"]].index(n)
            refs.append(np.asarray(outp[k_][1]["out0"], np.float64))
        else:
            refs.append(np.asarray(pr[-1][1]["out0"], np.float64) if pr else np.zeros(0))
    refs = [np.transpose(r, (0, 3, 1, 2)) if r.ndim == 4 else r for r in refs]
    recs_nchw = []
    for k, v in recs:
        v2 = dict(v)
        if v["type"] == "ConvStrippedOp" and n in P.conv:
            v["convrec"] = P.conv[n]
            v2["convrec"] = P.conv[n]
        for kk in list(v2):
            if kk[:2] in ("in", "ou") and kk[-1].isdigit() and np.asarray(v2[kk]).ndim == 4:
                v2[kk] = np.transpose(np.asarray(v2[kk]), (0, 3, 1, 2))
        recs_nchw.append((k, v2))
    ch = AT.Chain(recs_nchw, refs)
    # the atomic functions expect NHWC records for "out0"-based ops: keep the original records, minus any gather
    # map the chain's validation rejected (a reduce op's summed index tensor)
    for (k, v), (k2, v2) in zip(recs, recs_nchw):
        if "perm" in v and "perm" not in v2:
            v.pop("perm")
    ch.recs = recs
    outp = _out_ops(recs, L)
    if outp:
        names = [k for k, _ in recs]
        ch.out_idx = [names.index(k) for k, _ in outp]
    return ch


def _chain(n, L, ins, P, layers, dev):
    if n not in _CHAINS:
        _CHAINS[n] = build_chain(n, L, P, layers)
    xs = [x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in ins]
    y = _CHAINS[n](xs)
    return [t.to(dev) for t in y] if isinstance(y, list) else y.to(dev)


def use_chain(n, L, P):
    """Layers whose dedicated handler does not cover the captured variant (shared rule for graph26 and qatfast)."""
    t, prm = L["type"], L.get("params", {})
    ops = [v for k, v in P.ops.items() if k.rsplit("/", 1)[0] == n]
    types = {v["type"] for v in ops}
    if t == "format_conversion" and any("perm" in v for v in ops):
        return True
    if t in ("reduce_max", "reduce_sum", "reduce_mean") and L["output_shapes"][0][1:3] != L["input_shapes"][0][1:3]:
        return True
    if t == "maxpool" and any(v["type"] == "MaxPoolOp" and "attr:pool_size" in v for v in ops):
        return True
    if t == "concat" and any(v["type"] == "ConcatOp" and v.get("attr:_group_sizes") is not None for v in ops):
        return True
    if t == "resize" and any(float(r_) != int(r_) for r_ in list(prm.get("resize_h_ratio_list", [1]))
                             + list(prm.get("resize_w_ratio_list", [1]))):
        return True
    if t == "ew_mult" and "ElementwiseMultOnMacOp" in types:
        return True
    if t == "concat" and len({tuple(s[1:3]) for s in L["input_shapes"]}) > 1:  # spatial concat (class token)
        return True
    return t in ("reduce_mean", "feature_multiplier", "slice", "const_input", "space_to_depth", "avgpool",
                 "precision_splitter", "external_pad", "deconv", "depth_to_space")


def _dispatch(n, L, t, prm, ins, P, layers, img, dev, ints):
    if True:
        if t == "dense" and n in P.conv:  # FlattenOp (captured gather map) + the conv chain
            import atomic as AT
            fl = [v for k, v in P.ops.items() if k.rsplit("/", 1)[0] == n and v["type"] == "FlattenOp"]
            x = AT.gather(fl[0], ins[0]) if fl and "perm" in fl[0] else ins[0]
            if x.dim() == 2:
                x = x[:, :, None, None]
            r = P.conv[n]
            p = {**to_t(r, dev), **conv_params(r, dev)}
            y = I.conv_layer(x, p)[2]
        elif t == "input_layer" and img is not None:
            po = [v for k, v in P.ops.items() if k.rsplit("/", 1)[0] == n]
            sc = float(np.asarray(po[-1].get("output_scale", 1.0)).ravel()[0]) if po else 1.0
            zp = float(np.asarray(po[-1].get("output_zero_point", 0.0)).ravel()[0]) if po else 0.0
            x = torch.as_tensor(img, dtype=torch.float64, device=dev).permute(0, 3, 1, 2)
            if abs(sc - 1.0) > 1e-9 or zp != 0.0:  # the host quantizes the input: round(x / scale + zp)
                mode = os.environ.get("INPUT_ROUND", "even")
                q = I.f32_quantize(x, sc, zp)  # HSim/HailoRT quantize in IEEE float32 (f64 misses ties, CUDA f32 div is inexact)
                q = torch.round(q) if mode == "even" else torch.floor(q + 0.5)
                bits = int(po[-1].get("outq_bits", 8)) if po else 8  # UINT16 inputs: the chip uses all of 0..65535
                x = q.clamp(0, 2 ** bits - 1)
            y = x
        elif t in ("conv", "dw", "normalization", "batch_norm"):
            r = P.conv[n]
            p = {**to_t(r, dev), **conv_params(r, dev)}
            if ints is not None and n in ints:  # trained integers override the HAR's
                p["k"], p["bias"] = ints[n][0].to(torch.float64), ints[n][1].to(torch.float64)
            if prm.get("elementwise_add"):
                ew = P.op_by_type(n, "ElementwiseAddOp")
                p.update(ca_factor=_t(ew["elwa_factor"], dev), ca_shift=torch.tensor(float(ew["pre_acc_shift"])),
                         ca_feed=torch.tensor(float(ew["elwa_feed_repeat"])))
                y = I.conv_add_layer(ins[0], ins[1], p)[3]
            else:
                y = I.conv_layer(ins[0], p)[2]
        elif t in ("ew_add", "ew_sub"):
            y = I.ew_add_layer(ins[0], ins[1], to_t(P.conv[n], dev))[2]
        elif t == "argmax":  # channel index of the max (HSim: first index on ties; reverse_order = last)
            x_ = ins[0].flip(1) if prm.get("reverse_order") else ins[0]
            y = torch.argmax(x_, dim=1, keepdim=True)
            y = (x_.shape[1] - 1 - y if prm.get("reverse_order") else y).to(ins[0].dtype)
        elif t == "feature_splitter":
            y = I.feature_split(ins[0], L)
        elif t == "shortcut":
            y = ins[0]
        elif t == "concat" and any(v["type"] == "ConcatOp" and v.get("attr:_group_sizes") is not None
                                   for k, v in P.ops.items() if k.rsplit("/", 1)[0] == n):
            raise NotImplementedError("grouped concat: atomic chain")
        elif t == "concat":
            y = (torch.cat(ins, dim={'spatial_h': 2, 'spatial_w': 3}[prm['concat_axis']])
                 if prm.get('concat_axis') in ('spatial_h', 'spatial_w')  # HN concat_axis; default = features
                 else I.concat_features(ins, I.concat_groups(layers, n)))
        elif t == "maxpool" and any(v["type"] == "MaxPoolOp" and "attr:pool_size" in v
                                    for k, v in P.ops.items() if k.rsplit("/", 1)[0] == n):  # needs AUTO_ATTRS capture
            raise NotImplementedError("maxpool: atomic chain (padding never wins)")
        elif t == "maxpool":
            y = maxpool(ins[0], L)
        elif t == "resize" and any(float(r_) != int(r_) for r_ in list(prm.get("resize_h_ratio_list", [1])) + list(prm.get("resize_w_ratio_list", [1]))):
            raise NotImplementedError("non-integer resize ratio: atomic chain")
        elif t == "resize":
            y = ins[0].repeat_interleave(int(prm["resize_h_ratio_list"][0]), 2)
            y = y.repeat_interleave(int(prm["resize_w_ratio_list"][0]), 3)
        elif t == "format_conversion" and any("perm" in v for k, v in P.ops.items() if k.rsplit("/", 1)[0] == n):
            raise NotImplementedError("format_conversion: use the captured gather map (atomic chain)")
        elif t == "format_conversion":
            assert prm["conversion_type"] == "spatial_reshape", prm
            oh, ow = L["output_shapes"][0][1:3]
            y = ins[0].reshape(ins[0].shape[0], ins[0].shape[1], oh, ow)  # row-major H*W <-> (H, W)
        elif t == "activation":
            mk = P.op_by_type(n, "MockConvOp")
            y = apu_of(P.op_by_type(n, "ActivationOp"), bias_of(P.op_by_type(n, "AddBiasOp"), mock_of(mk, ins[0], dev), dev), dev)
        elif t == "matmul":
            m = P.op_by_type(n, "MatmulOp")
            zp0 = float(np.asarray(m["input_zero_points"][0]).ravel()[0])
            acc = I.matmul_layer_acc(ins[0], ins[1], int(m["_groups"]), bool(m["_transpose_matmul_input"]),
                                     bool(m["_zp_comp_added"]), int(m.get("zp_comp_rank", 1)),
                                     np.asarray(m.get("feed_repeat", 1), np.float64), zp0,
                                     int(m["_multiplier_shift"]), lossy_of(m),
                                     windows=np.asarray(m.get("_input_windows", [1, 1, 1])).ravel()[:2],
                                     x1_bits=_out_bits(P, layers, L["input"][1]))
            y = apu_of(P.op_by_type(n, "ActivationOp"), acc, dev)
        elif t in ("reduce_max", "reduce_sum", "reduce_mean") and L["output_shapes"][0][1:3] != L["input_shapes"][0][1:3]:
            raise NotImplementedError("spatial reduction: atomic chain")
        elif t == "reduce_max":
            y = I.reduce_max_layer(ins[0], int(prm["groups"]))
        elif t == "reduce_sum":
            rs = P.op_by_type(n, "ReduceSumOp")
            acc = I.reduce_sum_acc(ins[0], rs["kernel_q"], int(rs["pre_acc_shift"]), int(prm["groups"]), lossy_of(rs))
            y = apu_of(P.op_by_type(n, "ActivationOp"), bias_of(P.op_by_type(n, "AddBiasOp"), acc, dev), dev)
        elif t == "ew_mult":
            mks, bss = P.ops_by_type(n, "MockConvOp"), P.ops_by_type(n, "AddBiasOp")  # sorted: conv_op_a, conv_op_b
            mu = P.op_by_type(n, "ElementwiseMultOp")
            a = bias_of(bss[0], mock_of(mks[0], ins[0], dev), dev)
            b = bias_of(bss[1], mock_of(mks[1], ins[1], dev), dev)
            rep = prm.get("input_repeats", [[1, 1, 1], [1, 1, 1]])
            a, b = I.apply_repeats(a, rep[0]), I.apply_repeats(b, rep[1])
            y = apu_of(P.op_by_type(n, "ActivationOp"), I.ew_mult_acc(a, b, int(mu["_multiplier_shift"]), lossy_of(mu)), dev)
        else:
            raise NotImplementedError(f"{n}: {t}")
        return y


if __name__ == "__main__":
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    hn_path, cap_dir, img_path = pos[:3]
    dev = pos[3] if len(pos) > 3 else "cuda"
    cont = "--cont" in flags
    js = next((f.split("=", 1)[1] for f in flags if f.startswith("--json=")), None)
    import cv2
    hn = json.load(open(hn_path))
    P = Params(cap_dir, dev)
    if img_path.endswith(".npy"):  # non-image input (audio features, tokens, ...): the exact array capture.py fed
        img = np.load(img_path).astype(np.float64)
    else:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        in_l = [v for v in hn["layers"].values() if v["type"] == "input_layer"][0]
        h, w = in_l["input_shapes"][0][1:3]
        if img.shape[:2] != (h, w):  # same letterbox as capture.py / evaluate.py
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from letterbox import letterbox
            img = letterbox(img, (h, w))[0]
        img = img[None].astype(np.float64)
    with torch.no_grad():
        out, report = run(hn, P, img, dev, cont=cont)
    ok = sum(b == 0 for _, _, b, _ in report)
    print(f"{ok}/{len(report)} checked layers exact when chained from raw pixels")
    by_t = {}
    for _, t, b, _ in report:
        by_t.setdefault(t, [0, 0])
        by_t[t][0] += b == 0
        by_t[t][1] += 1
    print({t: f"{a}/{b}" for t, (a, b) in by_t.items()})
    if js:
        status = {0: "exact", -1: "shape", -2: "unsupported", -3: "blocked", -4: "unchecked"}
        json.dump({"layers": [{"layer": n, "type": t, "status": status.get(b, "mismatch"), "bad": b, "numel": m}
                              for n, t, b, m in report]}, open(js, "w"), indent=0)

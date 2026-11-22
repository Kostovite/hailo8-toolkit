"""Batch-4 rewrites: model-level passes for Hailo-8 SILICON / compiler defects (RULES.md sections 25-26), run by
onnx_hailo_compat after the node rules. Exact in float (each pass is checked by onnx_hailo_compat's float check).

  convt_ks    ConvTranspose kernel = stride (k2s2, k4s4, ...: the compiler's DECONV / FEATURE_INTERLEAVE lowering of
              is_super_deconv's kernel == stride shapes is ~100% wrong on the chip) -> Conv1x1 (C -> k*k*C) +
              DepthToSpace(k, CRD). (old name convt_k2s2)
  convt_same  ConvTranspose whose pads are not TF-SAME: the parser always builds output = in * stride with the odd
              pad at the END and drops the ONNX begin pad (k3s2p1 on 24: 48 x 48 = onnx 47 x 47 with one extra
              leading row / column; RULES 26) -> the SAME-padded ConvTranspose + Slice to the ONNX window
  dil_pad     dilated Conv whose output width is not a multiple of 8 (the compiler cuts a row into 8-aligned pieces
              + one remainder piece; a piece 9..8+d-1 wide (d <= 4) reads stale memory on the chip) -> zero-Pad W
              on the right so the output is a multiple of 8, Conv, Slice back
  attn_bias   K / V biases of softmax attention: K's dropped (softmax-invariant), V's moved after P.V (rows of P sum
              to 1): centred K / V avoid the dynamic matmul's uncompensated zero point (+2..+8 dB on chip)
"""
import os
import sys

import numpy as np
from onnx import helper, numpy_helper as nh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "turbo"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixes"))  # release layout
MODEL_PASSES = ["convt_ks", "convt_same", "dil_pad", "slice_down", "attn_bias"]
ALIAS = {"convt_k2s2": "convt_ks"}


def _shapes(m):
    import onnxruntime as ort
    import onnx
    ort.set_default_logger_severity(3)
    m2 = onnx.ModelProto(); m2.CopyFrom(m)
    have = {o.name for o in m2.graph.output}
    for n in m2.graph.node:
        for o in n.output:
            if o not in have:
                m2.graph.output.append(helper.make_tensor_value_info(o, 1, None))
    s = ort.InferenceSession(m2.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {i.name: np.zeros([d.dim_value or 1 for d in i.type.tensor_type.shape.dim], np.float32)
             for i in m.graph.input if i.name not in {x.name for x in m.graph.initializer}}
    names = [o.name for o in s.get_outputs()]
    shp = {k: v.shape for k, v in zip(names, s.run(names, feeds))}
    shp.update({k: v.shape for k, v in feeds.items()})
    return shp


def convt_ks(m):
    g = m.graph; inits = {i.name: nh.to_array(i) for i in g.initializer}; nodes, hit = [], 0
    for n in g.node:
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        if n.op_type != "ConvTranspose":
            nodes.append(n); continue
        k = list(a.get("kernel_shape", [])) or (list(inits[n.input[1]].shape[2:]) if n.input[1] in inits else [])
        if len(k) != 2 or k[0] != k[1] or k[0] < 2 or list(a.get("strides", [1, 1])) != k \
                or any(a.get("pads", [0])) or a.get("group", 1) != 1 or any(a.get("output_padding", [0])) \
                or any(d != 1 for d in a.get("dilations", [1, 1])) or a.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET") \
                or n.input[1] not in inits:
            nodes.append(n); continue
        W = inits[n.input[1]]; cin, cout, kh, kw = W.shape
        B = inits[n.input[2]] if len(n.input) > 2 and n.input[2] in inits else np.zeros(cout, np.float32)
        wn, bn, cn = n.output[0] + "_w_d2s", n.output[0] + "_b_d2s", n.output[0] + "_c_d2s"
        g.initializer.extend([nh.from_array(W.transpose(1, 2, 3, 0).reshape(cout * kh * kw, cin, 1, 1).astype(np.float32), wn),
                              nh.from_array(np.repeat(B, kh * kw).astype(np.float32), bn)])
        nodes += [helper.make_node("Conv", [n.input[0], wn, bn], [cn], kernel_shape=[1, 1]),
                  helper.make_node("DepthToSpace", [cn], [n.output[0]], blocksize=kh, mode="CRD")]
        hit += 1
    del g.node[:]; g.node.extend(nodes)
    return m, hit


def convt_same(m):
    shp = _shapes(m); g = m.graph; nodes, hit = [], 0
    for n in g.node:
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        if n.op_type != "ConvTranspose" or n.input[0] not in shp or len(shp[n.input[0]]) != 4 \
                or any(a.get("output_padding", [0])) or any(d != 1 for d in a.get("dilations", [1, 1])) \
                or a.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            nodes.append(n); continue
        k = list(a.get("kernel_shape", [])) or list(shp.get(n.input[1], (0, 0, 0, 0))[2:])
        st, pads = list(a.get("strides", [1, 1])), list(a.get("pads", [0, 0, 0, 0]))
        same, crop, ok = [0, 0, 0, 0], [], True
        for ax in range(2):
            full = (shp[n.input[0]][2 + ax] - 1) * st[ax] + k[ax]
            tot = full - shp[n.input[0]][2 + ax] * st[ax]  # the parser's output is in * stride
            sb, se, pb, pe = tot // 2, tot - tot // 2, pads[ax], pads[2 + ax]
            ok &= tot >= 0 and pb >= sb and pe >= se
            same[ax], same[2 + ax] = sb, se
            crop.append((pb - sb, full - sb - se - (pe - se)))
        if not ok or same == pads:  # already what the parser builds (or not expressible as a crop of it)
            nodes.append(n); continue
        y, cy = n.output[0], n.output[0] + "_same_full"
        conv = helper.make_node("ConvTranspose", list(n.input), [cy], name=n.name)
        conv.attribute.extend(x for x in n.attribute if x.name != "pads")
        conv.attribute.append(helper.make_attribute("pads", same))
        g.initializer.extend([nh.from_array(np.array([c[0] for c in crop], np.int64), y + "_cs_s"),
                              nh.from_array(np.array([c[1] for c in crop], np.int64), y + "_cs_e"),
                              nh.from_array(np.array([2, 3], np.int64), y + "_cs_a")])
        nodes += [conv, helper.make_node("Slice", [cy, y + "_cs_s", y + "_cs_e", y + "_cs_a"], [y])]
        hit += 1
    del g.node[:]; g.node.extend(nodes)
    return m, hit


def dil_pad(m):
    shp = _shapes(m); g = m.graph; nodes, hit, padded = [], 0, set()  # one Pad per (input, padded width)
    for n in g.node:
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        d = list(a.get("dilations", [1, 1]))
        if n.op_type != "Conv" or max(d) <= 1 or n.input[0] not in shp or len(shp[n.input[0]]) != 4:
            nodes.append(n); continue
        x, y = n.input[0], n.output[0]; w, wo = shp[x][3], shp[y][3]
        wp = w + (-(-wo // 8) * 8 - wo) * list(a.get("strides", [1, 1]))[1]  # output width -> a multiple of 8
        if wp == w:
            nodes.append(n); continue
        pn, cy, xp = f"{x}_dilpad{wp}_pads", y + "_dilpad_full", f"{x}_dilpad{wp}"
        g.initializer.extend([nh.from_array(np.array([0], np.int64), y + "_dp_s"), nh.from_array(np.array([wo], np.int64), y + "_dp_e"),
                              nh.from_array(np.array([3], np.int64), y + "_dp_a")])
        if xp not in padded:
            padded.add(xp)
            g.initializer.append(nh.from_array(np.array([0, 0, 0, 0, 0, 0, 0, wp - w], np.int64), pn))
            nodes.append(helper.make_node("Pad", [x, pn], [xp], mode="constant"))
        conv = helper.make_node("Conv", [xp] + list(n.input[1:]), [cy], name=n.name)
        conv.attribute.extend(n.attribute)
        nodes += [conv, helper.make_node("Slice", [cy, y + "_dp_s", y + "_dp_e", y + "_dp_a"], [y])]
        hit += 1
    del g.node[:]; g.node.extend(nodes)
    return m, hit


UNARY = ("Relu", "LeakyRelu", "PRelu", "Clip", "Sigmoid", "HardSigmoid", "HardSwish", "Tanh", "Elu", "Selu")


def slice_down(m):
    """Slice -> pointwise activation  ==  activation -> Slice. The dil_pad crop otherwise sits between a conv and its
    activation: the parser cannot fuse them and stores the PRE-activation output (negatives included) in 8 bits."""
    g = m.graph; hit = 0
    while True:
        cons = {}
        for n in g.node:
            for i in n.input:
                cons.setdefault(i, []).append(n)
        outs = {o.name for o in g.output}
        swap = None
        for n in g.node:
            if n.op_type == "Slice" and n.output[0] not in outs and len(cons.get(n.output[0], [])) == 1:
                a = cons[n.output[0]][0]
                if a.op_type in UNARY and a.input[0] == n.output[0]:
                    swap = (n, a); break
        if not swap:
            break
        sl, act = swap
        x, mid, y = sl.input[0], sl.output[0], act.output[0]
        act.input[0], act.output[0] = x, mid  # act now reads the full tensor, writes `mid`
        sl.input[0], sl.output[0] = mid, y     # the slice crops the activated tensor
        nodes = list(g.node); i, j = nodes.index(sl), nodes.index(act)
        nodes[i], nodes[j] = act, sl
        del g.node[:]; g.node.extend(nodes)
        hit += 1
    return m, hit


def attn_bias(m):
    import attn_bias_rewrite
    m2, moved = attn_bias_rewrite.rewrite_model(m)
    return m2, len(moved)


def apply(m, rules):
    count = {}
    rules = [ALIAS.get(r, r) for r in rules]
    for p in MODEL_PASSES:
        if p in rules:
            m, h = globals()[p](m)
            if h:
                count[p] = h
    return m, count

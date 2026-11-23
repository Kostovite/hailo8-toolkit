"""aifi_spatial.py IN.onnx OUT.onnx : run the AIFI encoder layer in the 28x28 (NCHW) layout instead of tokens [1, 784, C].

Why (DFC 3.33, Hailo-8, measured 2026-11-22 on layout_g1): in token layout every 16-bit tensor is a 784-wide row
(784 x 256 x 2 B = 392 KiB, the row limit is 128 KiB). The full compile then fails with "Assignment of the node needs
super-defuse"; cut down, the 16-bit positional-embedding const_input (Agent infeasible) and the 8->16 bit
precision_change after the token reshape ("more than 1 subclusters are needed for 16bit L2 biases") cannot be placed.
Everything in AIFI except the two attention matmuls is per token (projections, positional add, residuals, layer norms,
GELU MLP), so it is computed identically on the 28x28 map with 1x1 convs and channel (axis 1) reductions. Only the
attention core (head reshapes, Q.K, softmax, P.V) stays in token layout, entered by Reshape+Transpose per q/k/v and
left by Transpose+Reshape after P.V -- the same patterns the DFC already parses as format_conversion.
Float-exact: check with onnx_parity.py.
"""
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

PFX = "/encoder/aifi.0/"
CORE = ("self_attn/Reshape", "self_attn/Transpose", "self_attn/MatMul", "self_attn/Mul", "self_attn/Softmax",
        "MatMul_1_output_0_vbias")  # head reshapes/transposes, Q.K, scale, softmax, P.V (+ folded v bias)


def main(src, dst):
    m = onnx.load(src); g = m.graph
    init = {t.name: t for t in g.initializer}
    nodes = list(g.node)
    # AIFI span: from the entry Reshape to the exit Reshape_1 (inclusive)
    i0 = next(i for i, n in enumerate(nodes) if n.name == PFX + "Reshape")
    i1 = next(i for i, n in enumerate(nodes) if n.name == PFX + "Reshape_1")
    span = nodes[i0:i1 + 1]
    entry_r, entry_t = span[0], span[1]; exit_t, exit_r = span[-2], span[-1]
    assert entry_t.name == PFX + "Transpose" and exit_t.name == PFX + "Transpose_1", (entry_t.name, exit_t.name)
    src_nchw = entry_r.input[0]
    sp = {entry_t.output[0]: src_nchw}           # token tensor name -> its NCHW twin
    out_nodes, newinit = [], []
    cnt = [0]

    def nm(base):
        cnt[0] += 1; return f"aifisp_{base}_{cnt[0]}"

    def const(arr, base):
        t = numpy_helper.from_array(np.ascontiguousarray(arr.astype(np.float32)), nm(base)); newinit.append(t); return t.name

    def is_core(n):
        return any(c in n.name or any(c in o for o in n.output) for c in CORE)

    def to_token(x):  # NCHW [1,C,28,28] -> [1,784,C]
        a = nm("r"); b = nm("t")
        sh = nm("shp"); newinit.append(numpy_helper.from_array(np.array([1, 0, -1], np.int64), sh))
        out_nodes.append(helper.make_node("Reshape", [x, sh], [a], name=a))
        out_nodes.append(helper.make_node("Transpose", [a], [b], name=b, perm=[0, 2, 1]))
        return b

    def to_nchw(x, c):  # [1,784,C] -> [1,C,28,28]
        a = nm("t"); b = nm("r")
        out_nodes.append(helper.make_node("Transpose", [x], [a], name=a, perm=[0, 2, 1]))
        sh = nm("shp"); newinit.append(numpy_helper.from_array(np.array([1, c, 28, 28], np.int64), sh))
        out_nodes.append(helper.make_node("Reshape", [a, sh], [b], name=b))
        return b

    tok_of = {}  # NCHW tensor -> token-layout copy made for the core (made once)
    core_out_nchw = {}
    body = span[2:-2]
    # v bias after P.V (compat moved v_proj's bias there): softmax rows sum to 1, so P.(V + b) = P.V + b, which after
    # the head merge is a per-channel bias of the o_proj input -> fold b @ W_o into the o_proj bias (exact). Left as an
    # Add it would become a 784-wide 16-bit normalization layer in token layout.
    vb = None
    for n in list(body):
        if n.op_type == "Add" and any(x.endswith("_vbias") and x in init for x in n.input):
            c = next(x for x in n.input if x.endswith("_vbias") and x in init); d = next(x for x in n.input if x != c)
            a = numpy_helper.to_array(init[c])  # broadcast over [1, heads, tokens, dim]
            vb = np.broadcast_to(a, (1, a.shape[1], 1, a.shape[-1])).reshape(-1).astype(np.float64)  # head-major = merge order
            for k in body:
                for j, x in enumerate(k.input):
                    if x == n.output[0]: k.input[j] = d
            body.remove(n)
    # q/k/v identity requant layers (add_qkv_identity.py) are not needed in this layout: the DFC does the 16 -> 8 bit
    # step as a precision_change on the 28x28 map before the token reshape. Kept, the parser moves the K and V ones
    # behind the reshape and quantization then puts 8 -> 16 bit precision_changes on 784-wide rows (unplaceable).
    if os.environ.get("AIFI_KEEP_ID", "0") != "1":
        for n in [n for n in body if n.name.endswith("/IdMatMul")]:
            ad = next(k for k in body if n.output[0] in k.input and k.name.endswith("/IdAdd"))
            for k in body:
                for j, x in enumerate(k.input):
                    if x == ad.output[0]: k.input[j] = n.input[0]
            body.remove(n); body.remove(ad)
    cons = {}
    for n in body:
        for x in n.input: cons.setdefault(x, []).append(n)
    fold = {}  # MatMul output -> the per-channel bias Add that follows it (folded into the Conv)
    for n in body:
        if n.op_type == "MatMul" and n.input[1] in init and len(cons.get(n.output[0], [])) == 1:
            a = cons[n.output[0]][0]
            cs = [x for x in a.input if x in init]
            if a.op_type == "Add" and len(cs) == 1 and numpy_helper.to_array(init[cs[0]]).size == numpy_helper.to_array(init[n.input[1]]).shape[1]:
                fold[n.output[0]] = (a, numpy_helper.to_array(init[cs[0]]).reshape(-1))
    skip = {id(a) for a, _ in fold.values()}
    for n in body:
        if id(n) in skip:
            continue
        if is_core(n):
            ins = []
            for x in n.input:
                if x in sp:  # an NCHW-world tensor enters the core
                    if x not in tok_of: tok_of[x] = to_token(sp[x])
                    ins.append(tok_of[x])
                else:
                    ins.append(x)
            out_nodes.append(helper.make_node(n.op_type, ins, list(n.output), name=n.name,
                                              **{a.name: helper.get_attribute_value(a) for a in n.attribute}))
            continue
        # NCHW world: every non-constant input must be NCHW (core outputs are converted back here)
        ins = []
        for x in n.input:
            if x in sp: ins.append(sp[x])
            elif x in init or x == "": ins.append(x)
            else:  # a core output ([1,784,C]) used outside the core
                if x not in core_out_nchw: core_out_nchw[x] = to_nchw(x, 256)
                ins.append(core_out_nchw[x])
        y = nm(n.op_type.lower())
        if n.op_type == "MatMul":
            W = numpy_helper.to_array(init[n.input[1]])  # y = x @ W, W [Cin, Cout]
            w = const(W.T[:, :, None, None], "w")
            b = fold[n.output[0]][1].astype(np.float64) if n.output[0] in fold else np.zeros(W.shape[1])
            if vb is not None and n.input[0] in core_out_nchw:  # o_proj: absorb the P.V bias
                b = b + vb @ W.astype(np.float64); vb = None
            out_nodes.append(helper.make_node("Conv", [ins[0], w, const(b, "b")], [y], name=y, kernel_shape=[1, 1]))
            if n.output[0] in fold:
                sp[fold[n.output[0]][0].output[0]] = y
        elif n.op_type == "ReduceMean":
            ax = helper.get_attribute_value(next(a for a in n.attribute if a.name == "axes"))
            assert list(ax) in ([-1], [2]), ax
            out_nodes.append(helper.make_node("ReduceMean", ins, [y], name=y, axes=[1], keepdims=1))
        elif n.op_type in ("Add", "Sub", "Mul", "Div", "Erf", "Sqrt"):
            ins2 = []
            for x in ins:
                if x in init:
                    a = numpy_helper.to_array(init[x])
                    if a.size == 1: ins2.append(x)
                    elif a.ndim >= 1 and a.shape[-1] == a.size:  # per-channel [C] -> [1,C,1,1]
                        ins2.append(const(a.reshape(1, -1, 1, 1), "c"))
                    elif a.shape[-2:] == (784, a.shape[-1]):  # positional embedding [1,784,C] -> [1,C,28,28]
                        ins2.append(const(a.reshape(784, -1).T.reshape(1, -1, 28, 28), "pe"))
                    else:
                        raise SystemExit(f"unhandled const {x} {a.shape} in {n.name}")
                else:
                    ins2.append(x)
            out_nodes.append(helper.make_node(n.op_type, ins2, [y], name=y,
                                              **{a.name: helper.get_attribute_value(a) for a in n.attribute}))
        else:
            raise SystemExit(f"unhandled op {n.op_type} {n.name}")
        sp[n.output[0]] = y
    assert vb is None, "P.V bias found but no o_proj to fold it into"
    last = sp[exit_t.input[0]]
    prod = next(n for n in out_nodes if last in n.output)
    prod.output[list(prod.output).index(last)] = exit_r.output[0]  # the last NCHW node writes the AIFI exit tensor
    new = nodes[:i0] + out_nodes + nodes[i1 + 1:]
    del g.node[:]; g.node.extend(new); g.initializer.extend(newinit)
    keep = {x for n in g.node for x in n.input}
    rm = [t for t in g.initializer if t.name not in keep]
    for t in rm: g.initializer.remove(t)
    del g.value_info[:]
    onnx.checker.check_model(m); onnx.save(m, dst)
    print(f"aifi_spatial: {len(span)} nodes -> {len(out_nodes)}; core inputs converted {len(tok_of)}, outputs {len(core_out_nchw)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

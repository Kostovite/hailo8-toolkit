"""Cut a sub-network out of a quantized HAR: the layers of one context become a standalone HAR the DFC compiles
on its own (single context). Edges entering the set become input layers, edges leaving it output layers; both take
the producer's output quantization (qp_out, limvals_out, zero_point_out, scales), so the sub-network computes
exactly the same integers as the context does inside the full model (inter-context tensors cross via host anyway).

Usage (DFC env): python subhar.py FULL.har LAYERS.json OUT.har     LAYERS.json: ["conv1", "conv2", ...] (short names)
       from python: build(full_har, layer_names, out_har)
"""
import copy
import io
import json
import sys
import tarfile

import numpy as np

IO_KEYS = ("qp_in:0", "qp_out:0", "limvals_in:0", "limvals_out:0", "zero_point_in:0", "zero_point_out:0",
           "layer_params/output_scales:0", "layer_params/input_scales:0", "layer_params/output_zero_points:0",
           "layer_params/input_zero_points:0", "layer_params/negative_slopes_correction_factor:0")


def _members(path):
    with tarfile.open(path) as t:
        return {m.name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}


DISTINCT_OUTPUTS = {"feature_splitter", "precision_splitter", "output_mux"}  # outputs are different tensors


def _out_index(layer, consumer):
    """Which tensor of the producer feeds `consumer`: other layers fan ONE tensor out to all consumers (one
    output_shapes entry per consumer, all the same), so they share index 0."""
    return layer["output"].index(consumer) if layer["type"] in DISTINCT_OUTPUTS else 0


def build(full_har, names, out_har):
    files = _members(full_har)
    stem = next(n[:-3] for n in files if n.endswith(".hn") and not n.endswith((".native.hn", ".fp.hn")))
    hn = json.loads(files[f"{stem}.hn"])
    net = hn["name"]
    L = hn["layers"]
    full = lambda s: s if s.startswith(net + "/") else f"{net}/{s}"  # noqa: E731
    keep = {full(n) for n in names}
    missing = keep - set(L)
    assert not missing, f"unknown layers: {sorted(missing)[:5]}"
    # A format_conversion whose producer is cut away would start the sub-network as "input -> format_conversion":
    # the compiler collapses format conversions into their producer (auto_spatial_reshape) and throws
    # out_of_range when that producer is an input layer. Cut AFTER it instead: the input carries the converted
    # tensor; the full model removes the node anyway (remove_node in its script, see assemble.py).
    dropped = sorted(n for n in keep if L[n]["type"] == "format_conversion" and L[n]["input"][0] not in keep
                     and L[L[n]["input"][0]]["type"] != "input_layer")
    keep -= set(dropped)
    q = dict(np.load(io.BytesIO(files[f"{stem}.q.npz"]), allow_pickle=True))
    new_L, new_q = {}, {}
    io_params = {}  # producer -> its output-side quantization, as an IO layer's in and out

    def io_quant(src):
        if src not in io_params:
            p = {}
            for k in ("qp", "limvals", "zero_point"):
                v = q.get(f"{src}/{k}_out:0")
                if v is not None:
                    p[f"{k}_in:0"] = p[f"{k}_out:0"] = v
            for k in ("output_scales", "output_zero_points"):
                v = q.get(f"{src}/layer_params/{k}:0")
                if v is not None:
                    p[f"layer_params/{k}:0"] = v
                    p[f"layer_params/{k.replace('output', 'input')}:0"] = v
            v = q.get(f"{src}/layer_params/negative_slopes_correction_factor:0")
            if v is not None:
                p["layer_params/negative_slopes_correction_factor:0"] = v
            io_params[src] = p
        return io_params[src]

    def prec_bits(src):
        pm = L[src].get("quantization_params", {}).get("precision_mode", "a8_w8_a8")
        out_bits = pm.split("_")[-1] if pm.count("_") == 2 else pm.split("_")[0]
        return "a16_w16_a16" if out_bits.endswith("16") else "a8_w8_a8"

    n_in = n_out = 0
    in_of = {}  # (external producer, output index) -> new input layer name
    for name in [n for n in L if n in keep]:
        lay = copy.deepcopy(L[name])
        for i, src in enumerate(lay["input"]):
            if src in keep:
                continue
            if L[src]["type"] == "input_layer":  # an original network input: keep it as is
                new_L.setdefault(src, copy.deepcopy(L[src]))
                new_L[src]["output"] = [o for o in L[src]["output"] if o in keep]
                for k in IO_KEYS:
                    if f"{src}/{k}" in q:
                        new_q[f"{src}/{k}"] = q[f"{src}/{k}"]
                continue
            oi = _out_index(L[src], name)
            key = (src, oi)
            if key not in in_of:
                n_in += 1
                iname = f"{net}/input_layer{500 + n_in}"
                in_of[key] = iname
                shape = L[src]["output_shapes"][oi]
                new_L[iname] = {"type": "input_layer", "input": [], "output": [], "input_shapes": [shape],
                                "output_shapes": [shape], "original_names": [f"cut_{src.split('/')[-1]}_{oi}"],
                                "compilation_params": {}, "quantization_params": {
                                    "quantization_groups": 1, "quantization_weight_groups": 1,
                                    "precision_mode": prec_bits(src), "bias_mode": "single_scale_decomposition",
                                    "signed_output": L[src].get("quantization_params", {}).get("signed_output", False)},
                                "transposed": False, "engine": "nn_core", "io_type": "standard"}
                for k, v in io_quant(src).items():
                    new_q[f"{iname}/{k}"] = v
            iname = in_of[key]
            lay["input"][i] = iname
            new_L[iname]["output"].append(name)
        outs, out_of = [], {}  # tensor index -> the output layer already made for it
        for j, dst in enumerate(lay["output"]):
            tj = j if lay["type"] in DISTINCT_OUTPUTS else 0
            if dst in keep:
                outs.append(dst)
            elif L[dst]["type"] != "output_layer" and tj in out_of:
                continue  # the same tensor already leaves through an output layer
            elif L[dst]["type"] == "output_layer":
                new_L[dst] = copy.deepcopy(L[dst])
                for k in IO_KEYS:
                    if f"{dst}/{k}" in q:
                        new_q[f"{dst}/{k}"] = q[f"{dst}/{k}"]
                outs.append(dst)
            else:
                n_out += 1
                oname = f"{net}/output_layer{500 + n_out}"
                shape = lay["output_shapes"][j]
                new_L[oname] = {"type": "output_layer", "input": [name], "output": [], "input_shapes": [shape],
                                "output_shapes": [shape], "original_names": [f"cut_{name.split('/')[-1]}_{j}"],
                                "compilation_params": {}, "quantization_params": {
                                    "quantization_groups": 1, "quantization_weight_groups": 1,
                                    "precision_mode": prec_bits(name), "bias_mode": "single_scale_decomposition",
                                    "signed_output": lay.get("quantization_params", {}).get("signed_output", False)},
                                "transposed": False, "engine": "nn_core", "io_type": "standard"}
                for k, v in io_quant(name).items():
                    new_q[f"{oname}/{k}"] = v
                outs.append(oname)
                out_of[tj] = oname
        lay["output"] = outs
        if lay["type"] not in DISTINCT_OUTPUTS:
            lay["output_shapes"] = [lay["output_shapes"][0]] * len(outs)
        new_L[name] = lay
        for k in q:
            if k.startswith(name + "/"):
                new_q[k] = q[k]
    order = [n for n in new_L if new_L[n]["type"] == "input_layer"] + \
            [n for n in new_L if new_L[n]["type"] not in ("input_layer", "output_layer")] + \
            [n for n in new_L if new_L[n]["type"] == "output_layer"]
    hn["layers"] = {n: new_L[n] for n in order}
    orig_order = [o for o in hn["net_params"].get("output_layers_order", [])]
    feeders = [new_L[n]["input"][0] for n in order if new_L[n]["type"] == "output_layer"]
    hn["net_params"]["output_layers_order"] = [o for o in orig_order if o in feeders] + \
        [f for f in feeders if f not in orig_order]  # the network's own outputs first, then the cut edges
    buf = io.BytesIO()
    np.savez(buf, **new_q)
    hb = json.dumps(hn).encode()  # the loader wants a native hn too; the compiler only reads the quantized one
    out = {f"{stem}.hn": hb, f"{stem}.native.hn": hb, f"{stem}.fp.hn": hb, f"{stem}.q.npz": buf.getvalue()}
    skip = {f"{stem}.{x}" for x in ("npz", "fpo.npz", "ho.npz", "stats.npz", "hef", "auto.alls")}
    for k, v in files.items():  # metadata, model script, pre/post-process models: as in the full HAR
        if k not in out and k not in skip:
            out[k] = v
    with tarfile.open(out_har, "w") as t:
        for k, v in out.items():
            ti = tarfile.TarInfo(k)
            ti.size = len(v)
            t.addfile(ti, io.BytesIO(v))
    return {"layers": len(keep), "inputs": n_in, "outputs": n_out, "dropped": [n.split("/")[-1] for n in dropped]}


if __name__ == "__main__":
    print(build(sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]))

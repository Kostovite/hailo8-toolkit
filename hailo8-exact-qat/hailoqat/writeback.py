"""Write trained integer kernels/biases into a quantized HAR through Hailo's own exporter.

Same path as ClientRunner.set_keras_model: acceleras export_acceleras() + export_hw_params() ->
load_params(HAILO_OPTIMIZED / TRANSLATED). Kernels are set as native = (q - zp) * kernel_scale so that
get_quant_kernel() returns exactly the trained ints q; no shift/scale is recomputed.

Usage: python writeback.py IN_HAR INTS_NPZ OUT_HAR
  INTS_NPZ keys: "<layer>/k"  effective kernel ints in capture layout (HWIO, (q - zp) * 2**wps)
                 "<layer>/b"  bias ints at the accumulator (zero-point compensation included, as in q.npz bias:0)
"""
import sys

import numpy as np
import tensorflow as tf

from hailo_sdk_client import ClientRunner, InferenceContext
from hailo_sdk_common.targets.inference_targets import ParamsKinds

src, ints_path, dst = sys.argv[1:4]
ints = np.load(ints_path)
runner = ClientRunner(har=src)
backend = runner._sdk_backend
model = backend.build_acceleras_model(InferenceContext.SDK_QUANTIZED)
model.build(tuple([1] + list(runner.get_hn_model().get_input_layers()[0].input_shape[1:])))

def set_var(obj, attr, value):
    cur = getattr(obj, attr)
    if isinstance(cur, tf.Variable):
        cur.assign(value)
    else:
        setattr(obj, attr, tf.constant(value))


def write_bias(layer, b_target):
    """native = (b + in_zp - out_zp) * scale, so encode_bias()+bias_decompose == b exactly (zps after the new kernel)."""
    layer.enforce_internal_encoding()
    bop = layer.bias_add_op
    scale = np.asarray(bop.output_scale, np.float64)
    residue = np.asarray(bop.input_zero_points[0], np.float64) - np.asarray(bop.output_zero_point, np.float64)
    set_var(bop, "_bias", ((b_target + residue) * scale).astype(np.float32))
    got = np.rint(np.asarray(bop.bias_q)).astype(np.int64)
    bad = np.flatnonzero(got != b_target)
    if bad.size:  # float32 round trip off by one: nudge those channels by a fraction of an LSB
        nb = np.asarray(bop.bias).astype(np.float64).copy()
        nb[bad] += (b_target[bad] - got[bad]) * scale.reshape(-1)[bad if scale.size > 1 else 0] * 0.999
        set_var(bop, "_bias", nb.astype(np.float32))
        got = np.rint(np.asarray(bop.bias_q)).astype(np.int64)
    if not np.array_equal(got, b_target):
        raise SystemExit(f"{layer.full_name}: bias not representable, {int((got != b_target).sum())} channels "
                         f"(e.g. want {b_target[got != b_target][:4]} got {got[got != b_target][:4]})")


changed = 0
biases = {}
for key in ints.files:
    lname, kind = key.rsplit("/", 1)
    layer = model.layers[lname]
    op = layer.conv_op
    if kind == "b":
        biases[lname] = np.rint(ints[key]).astype(np.int64)
        continue
    if kind != "k":
        raise SystemExit(f"{key}: unknown kind")
    k_eff = ints[key].astype(np.float64)
    wps = float(np.asarray(op.weight_placement_shift))
    q_minus_zp = k_eff / 2 ** wps
    assert np.allclose(q_minus_zp, np.round(q_minus_zp)), f"{lname}: kernel not on the 2**wps grid"
    scale = np.asarray(op.kernel_scale, np.float64)
    native = (q_minus_zp * scale).astype(np.float32)
    if isinstance(op.kernel, tf.Variable):
        op.kernel.assign(native)
    else:
        op.kernel = tf.constant(native)
    if getattr(op, "kernel_scale_forced_to_save", False):
        op.kernel_q_forced = native / scale
    got = (np.asarray(op.get_quant_kernel(training=False)) - np.asarray(op.zp_kernel)) * 2 ** wps
    if not np.array_equal(np.rint(got).astype(np.int64), np.rint(k_eff).astype(np.int64)):
        raise SystemExit(f"{lname}: quant kernel does not reproduce the trained ints")
    changed += 1

# The export re-derives some biases (merge_residue_into_bias with zero points recomputed after the kernel change):
# on bias_factor 2/4 layers the stored bias:0 came out shifted by a per-layer constant (+1 ... +42). Read the
# translated params back, pre-compensate those layers and export again until every layer stores the trained ints.
# Decomposed biases (a16_w16: bias = f32(f32(factor * i) * repeats)) are re-decomposed by the export (factor and
# repeats are recomputed from the new values), so they cannot be pinned to the trained ints: write them as natives,
# let the export choose the lattice, and report what was stored. The round-trip eval then measures what ships.
decomposed = {lname for lname in biases
              if getattr(model.layers[lname].bias_add_op.weight_lossy_elements.bias_decompose, "num_decomposition", 0)}
for lname in decomposed:
    bop = model.layers[lname].bias_add_op
    model.layers[lname].enforce_internal_encoding()
    scale = np.asarray(bop.output_scale, np.float64)
    residue = np.asarray(bop.input_zero_points[0], np.float64) - np.asarray(bop.output_zero_point, np.float64)
    set_var(bop, "_bias", ((biases[lname] + residue) * scale).astype(np.float32))
targets = {lname: biases.pop(lname) for lname in decomposed}
offset = {lname: np.zeros_like(b) for lname, b in biases.items()}
for attempt in range(4):
    for lname, b in biases.items():
        write_bias(model.layers[lname], b - offset[lname])
    backend.load_params(model.export_acceleras(), ParamsKinds.HAILO_OPTIMIZED)
    backend.load_params(model.export_hw_params(), ParamsKinds.TRANSLATED)
    tr = runner.get_params_translated()
    bad = {}
    for lname, b in biases.items():
        got = np.rint(np.asarray(tr[f"{lname}/bias:0"], np.float64)).astype(np.int64).reshape(-1)
        if not np.array_equal(got, b):
            bad[lname] = got - b
    if not bad:
        break
    print(f"pass {attempt}: {len(bad)} layers stored a shifted bias: "
          + ", ".join(f"{k.split('/')[-1]} {np.unique(v)[:3].tolist()}" for k, v in bad.items()), flush=True)
    for lname, d in bad.items():
        offset[lname] = offset[lname] + d
else:
    raise SystemExit(f"biases still not reproduced after compensation: {sorted(bad)}")
for lname, b in targets.items():  # stored: bias:0 = 16-bit i, accumulator bias = f32(f32(factor * i) * feed_repeat)
    i = np.asarray(tr[f"{lname}/bias:0"], np.float64).reshape(-1).astype(np.float32)
    fac = np.float32(np.asarray(tr[f"{lname}/bias_factor:0"]).reshape(-1)[0])
    rep = np.float32(np.asarray(tr[f"{lname}/bias_feed_repeat:0"]).reshape(-1)[0])
    got = ((fac * i) * rep).astype(np.float64)
    print(f"{lname}: decomposed bias (factor {fac:.0f} x {rep:.0f}) stored within {int(np.abs(got - b).max())} of the "
          f"trained value", flush=True)
runner.save_har(dst)
print(f"wrote {changed} kernels, {len(biases)} biases -> {dst} (verified against the translated params)")

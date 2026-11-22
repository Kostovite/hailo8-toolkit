"""Capture Hailo bit-exact integer tensors and op parameters for every conv layer of a quantized HAR.

SDK_BIT_EXACT was verified against the Hailo-8 chip (hailo_re/chip_vs_emulator.py), so the tensors written here
are the ground truth that hailoqat's torch ops must reproduce exactly.

Usage: python capture.py HAR IMG OUT_DIR [LAYER_SUBSTR ...]
Writes OUT_DIR/<layer>.npz with:
  conv_in, conv_out (accumulator), bias_out, act_out           int tensors, NHWC
  k (effective int kernel), pre_acc_shift, acc_bits/signed/wrap, pad_*, strides, dilation, groups, depthwise
  bias, bias_bits/signed/wrap
  thr, mant, shift, offs, clip1_*, clip2_*, final_shift, mant_signed, out_bits/signed/wrap
"""
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from hailo_model_optimization.acceleras.atomic_ops.activation_op import ActivationOp
from hailo_model_optimization.acceleras.atomic_ops.bias_add_op import AddBiasOp
from hailo_model_optimization.acceleras.atomic_ops.conv_stripped_op import ConvStrippedOp
from hailo_model_optimization.acceleras.atomic_ops.element_wise_add_op import ElementwiseAddDirectOp
from hailo_model_optimization.acceleras.utils.acceleras_definitions import PaddingType, StrideAlignType
from hailo_model_optimization.acceleras.atomic_ops._misc_internals import get_tf_same_padding
from hailo_model_optimization.flows.inference_flow import SimulationInferenceModel
from hailo_sdk_client import ClientRunner, InferenceContext

HAR, IMG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FILTERS = sys.argv[4:]
os.makedirs(OUT, exist_ok=True)
tf.config.run_functions_eagerly(True)

records = {}


def layer_of(op):
    # op.full_name looks like "best/conv5/conv_op"
    return op.full_name.rsplit("/", 1)[0]


def wanted(lname):
    return not FILTERS or any(f in lname for f in FILTERS)


def lossy(prefix, el):
    return {f"{prefix}_bits": el.bits, f"{prefix}_signed": bool(el.signed), f"{prefix}_wrap": bool(el.wraparound)}


def np_(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


orig_conv = ConvStrippedOp.call_bit_exact
orig_bias = AddBiasOp.call_bit_exact
orig_act = ActivationOp.call_bit_exact


def conv_hook(self, inputs, training=False, **kwargs):
    out = orig_conv(self, inputs, training=training, **kwargs)
    ln = layer_of(self)
    if wanted(ln):
        qk = self.get_quant_kernel(training=False)
        k = (qk - tf.cast(self.zp_kernel, qk.dtype)) * 2 ** self.weight_placement_shift
        x = inputs[0]
        if self.padding == PaddingType.SAME:
            pb_h, pe_h, pb_w, pe_w = get_tf_same_padding(self.dilation_rate, *x.shape[1:3], *self.kernel_size,
                                                         *self.strides)
            if self.stride_align == StrideAlignType.NW:
                pb_h, pe_h, pb_w, pe_w = pe_h, pb_h, pe_w, pb_w
        else:
            pb_h = pe_h = pb_w = pe_w = 0
        r = records.setdefault(ln, {})
        r.update(conv_in=np_(x), conv_out=np_(out), k=np.rint(np_(k)).astype(np.int64),
                 pre_acc_shift=np_(self.pre_acc_shift), pad=np.array([pb_h, pe_h, pb_w, pe_w]),
                 pad_value=np_(self.padding_const_value_q), strides=np.array(self.strides),
                 dilation=np.array(self.dilation_rate), groups=self.groups, depthwise=bool(self.is_depthwise),
                 group_sizes=np.array(self.group_sizes), **lossy("acc", self.output_lossy_element))
    return out


def bias_hook(self, inputs, training=False, **kwargs):
    out = orig_bias(self, inputs, training=training, **kwargs)
    ln = layer_of(self)
    if wanted(ln) and ln in records:
        b = self.weight_lossy_elements.bias_decompose(self.encode_bias(), training=False)
        records[ln].update(bias_in=np_(inputs[0]), bias_out=np_(out), bias=np.rint(np_(b)).astype(np.int64),
                           precision_split_zp=bool(self._precision_split_zp), **lossy("bias", self.output_lossy_element))
    return out


def act_hook(self, inputs, **kwargs):
    out = orig_act(self, inputs, **kwargs)
    ln = layer_of(self)
    if wanted(ln) and ln in records:
        w = self.weight_lossy_elements
        mant = w.mantissa(self.mantissas_candidate)
        shift = -1 * (tf.cast(self.exponents, tf.int64) + tf.cast(self.shift_data, tf.int64))
        records[ln].update(act_in=np_(inputs[0]), act_out=np_(out), thr=np_(self.acc_thresholds),
                           mant=np.rint(np_(mant)).astype(np.int64), shift=np_(shift).astype(np.int64),
                           offs=np.rint(np_(self.real_offsets)).astype(np.int64),
                           final_shift=int(np_(self.apu_final_shift)), mant_signed=bool(w.mantissa.signed),
                           act_name=str(self.act_name.value),
                           out_scale=np_(self.output_scale), out_zp=np_(self.output_zero_point),
                           **lossy("clip1", w.clip1), **lossy("clip2", w.clip2),
                           **lossy("out", self.output_lossy_element))
    return out


orig_ew = ElementwiseAddDirectOp.call_bit_exact


def ew_hook(self, inputs, training=False, **kwargs):
    ins = [np_(i) for i in inputs]  # before repeat_inputs mutates the list
    out = orig_ew(self, inputs, training=training, **kwargs)
    ln = layer_of(self)
    if wanted(ln):
        records.setdefault(ln, {}).update(
            ew_in0=ins[0], ew_in1=ins[1], ew_out=np_(out),
            ew_k=np.rint(np_(self.get_quant_kernel(training=False))).astype(np.int64),
            ew_shift=np_(self.pre_acc_shift), ew_repeats=np.array(self.input_repeats),
            **lossy("ewacc", self.output_lossy_element))
    return out


ElementwiseAddDirectOp.call_bit_exact = ew_hook

# Generic recorder (ALL_OPS=1): integer in/out of every atomic op + its scalar/array attributes, keyed by op name.
# Ground truth for op types without a dedicated hook (matmul, softmax, mock conv, slices, reshapes, ...).
from hailo_model_optimization.acceleras.atomic_ops.base_atomic_op import BaseAtomicOp  # noqa: E402
from hailo_model_optimization.acceleras.atomic_ops.base_non_arithmetic_op import BaseNonArithmeticAtomicOp  # noqa: E402

op_records = {}
orig_be_run = BaseAtomicOp._bit_exact_run
ATTRS = ("kernel_q", "_multiplier_shift", "elwa_factor", "elwa_feed_repeat", "_reduce_axes", "input_repeats", "pre_acc_shift", "_multiplier_shift", "_groups", "_input_windows", "_transpose_matmul_input",
         "_zp_comp_added", "zp_comp_rank", "feed_repeat", "_input_tiles", "_axis", "cut_off", "kernel_scale",
         "input_zero_points", "input_scales", "output_scale", "output_zero_point", "recipe")


def be_run_hook(self, inputs, training, **kwargs):
    if not (os.environ.get("ALL_OPS") == "1" and wanted(layer_of(self))):
        return orig_be_run(self, inputs, training, **kwargs)
    # line-for-line BaseAtomicOp._bit_exact_run, keeping the integer tensors on both sides of call_bit_exact
    ins = self._to_numeric(inputs, training=training)
    ins = [tf.cast(t, self.INT_TYPE_TF) for t in ins]
    outs = self.call_bit_exact(ins, training=training, **kwargs)
    outs = outs if isinstance(outs, list) else [outs]
    out = self._from_numeric(outs, should_quant=False, training=training)
    if True:
        rec = {"type": type(self).__name__}
        for i, t in enumerate(ins):
            rec[f"in{i}"] = np_(t)
        for i, t in enumerate(outs):
            rec[f"out{i}"] = np_(t)
        for a in ATTRS:
            if hasattr(self, a):
                try:
                    v = getattr(self, a)
                    v = [np_(x) for x in v] if isinstance(v, (list, tuple)) else np_(v)
                    rec[a] = np.asarray(v, dtype=object) if isinstance(v, list) else v
                except Exception:
                    pass
        for el_name in ("output_lossy_element",):
            el = getattr(self, el_name, None)
            if el is not None and getattr(el, "bits", None) is not None:
                rec.update(lossy("outq", el))
        if hasattr(self, "get_quant_kernel"):
            try:
                rec["kernel_q"] = np_(self.get_quant_kernel(training=False))
            except Exception:
                pass
        if isinstance(self, AddBiasOp):
            try:
                rec["bias"] = np.rint(np_(self.weight_lossy_elements.bias_decompose(self.encode_bias(), training=False)))
            except Exception:
                pass
        if isinstance(self, ActivationOp):
            w = self.weight_lossy_elements
            rec.update(thr=np_(self.acc_thresholds), mant=np.rint(np_(w.mantissa(self.mantissas_candidate))),
                       shift=np_(-1 * (tf.cast(self.exponents, tf.int64) + tf.cast(self.shift_data, tf.int64))),
                       offs=np.rint(np_(self.real_offsets)), final_shift=int(np_(self.apu_final_shift)),
                       mant_signed=bool(w.mantissa.signed), **lossy("clip1", w.clip1), **lossy("clip2", w.clip2),
                       **lossy("out", self.output_lossy_element))
        if os.environ.get("AUTO_ATTRS") == "1":  # every small attribute, for unit-testing any atomic op type
            for a, v in vars(self).items():
                if a in rec or a.startswith("__"):
                    continue
                try:
                    if isinstance(v, (bool, int, float, str)):
                        rec["attr:" + a] = v
                    elif isinstance(v, (list, tuple)) and len(v) < 4096 and all(isinstance(x, (int, float, bool, np.integer, np.floating)) for x in v):
                        rec["attr:" + a] = np.asarray(v)
                    elif hasattr(v, "numpy") or isinstance(v, np.ndarray):
                        arr = np_(v)
                        if arr.size < 1 << 16:
                            rec["attr:" + a] = arr
                    elif hasattr(v, "value") and isinstance(getattr(v, "value"), (str, int)):  # enums
                        rec["attr:" + a] = v.value
                except Exception:  # noqa: BLE001
                    pass
        if os.environ.get("AUTO_ATTRS") == "1" and isinstance(self, BaseNonArithmeticAtomicOp):
            # data movement: push an index tensor (1-based; inputs concatenated in order) through the op's own float
            # path -> exact gather map for any layout op (0 = a constant the op inserts, e.g. padding)
            try:
                sizes = [int(np.prod(t.shape)) for t in ins]
                if sum(sizes) < (1 << 24):  # float32 holds integers exactly below 2^24
                    idx, off = [], 1
                    for t, sz in zip(ins, sizes):
                        idx.append(tf.reshape(tf.range(off, off + sz, dtype=tf.float32), t.shape))
                        off += sz
                    pm = self.call_native(idx)
                    pm = pm[0] if isinstance(pm, (list, tuple)) else pm
                    rec["perm"] = np.rint(np_(pm)).astype(np.int64)
            except Exception as e:  # noqa: BLE001
                rec["perm_error"] = f"{type(e).__name__}: {str(e)[:100]}"
        for lut in ("exp_lut", "inv_lut"):
            if hasattr(type(self), lut):
                try:
                    rec[lut] = np_(getattr(self, lut))
                except Exception:
                    pass
        op_records[self.full_name] = rec
    return out


BaseAtomicOp._bit_exact_run = be_run_hook
ConvStrippedOp.call_bit_exact = conv_hook
AddBiasOp.call_bit_exact = bias_hook
ActivationOp.call_bit_exact = act_hook

runner = ClientRunner(har=HAR)
model = runner._sdk_backend.build_acceleras_model(InferenceContext.SDK_BIT_EXACT)
flow = SimulationInferenceModel(model)
flow.set_quantized()
flow.set_native(False)
flow.set_bit_exact(True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from letterbox import letterbox  # noqa: E402

in_shape = runner.get_hn_model().get_input_layers()[0].input_shape  # [-1, H, W, C]
if IMG.endswith(".npy"):  # non-image input: NCHW float array, fed as is (NHWC)
    img = np.transpose(np.load(IMG).astype(np.float32).reshape([-1] + list(np.load(IMG).shape[-3:])), (0, 2, 3, 1))[:1]
else:
    img = letterbox(cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB), tuple(in_shape[1:3]))[0].astype(np.float32)[None]
flow.run(tf.data.Dataset.from_tensor_slices((img, np.zeros(1, np.float32))), 1, 1)

TENSORS = ("conv_in", "conv_out", "bias_in", "bias_out", "act_in", "act_out", "ew_in0", "ew_in1", "ew_out")
for ln, r in records.items():
    if os.environ.get("PARAMS_ONLY") == "1":
        r = {k: v for k, v in r.items() if k not in TENSORS}
    np.savez(os.path.join(OUT, ln.replace("/", "__") + ".npz"), **r)
if op_records:
    import pickle
    with open(os.path.join(OUT, "_all_ops.pkl"), "wb") as fh:
        pickle.dump(op_records, fh, protocol=4)
print(f"captured {len(records)} conv layers, {len(op_records)} atomic ops -> {OUT}")

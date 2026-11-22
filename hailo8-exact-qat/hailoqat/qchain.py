"""Trainable atomic-op chains for qatfast: any HN layer that has no dedicated QAT module runs as its captured chain
of acceleras atomic ops (atomic.py), with the exact integer value in the forward and a straight-through surrogate
gradient. Exact values are computed in float64 (LayerNorm sums of squares exceed float32's 2^24); the layer output
(an APU / data-movement result of <= 16 bits) is returned in float32.

Trainable: the APUs of the chain (qatfast.APU, registered in the net's `apus` dict). Data movement (gather, slice,
concat, reorder, pad, resize, maxpool) is exact and differentiable as is.
"""
import numpy as np
import torch

import atomic as A
import intops as I


def ste(exact, surrogate):
    return exact.detach() + (surrogate - surrogate.detach())


def _per_c(v, like):
    v = torch.as_tensor(np.asarray(v, np.float64), device=like.device, dtype=like.dtype)
    return v.reshape(1, -1, 1, 1) if v.numel() > 1 else v


def _kq_s(rec, x):
    return _per_c(rec["kernel_q"], x), 2.0 ** -int(rec["pre_acc_shift"])


def _sum_like(rec, v, x_shape):
    """Sum v over the axes the op reduces (channel groups or H / W), from the recorded output shape."""
    o = np.asarray(rec["out0"]).shape
    n, c, h, w = x_shape
    if o[3] != c:
        g = o[3]
        return v.reshape(n, g, c // g, h, w).sum(2)
    if o[1] == 1 and h != 1:
        v = v.sum(2, keepdim=True)
    if o[2] == 1 and w != 1:
        v = v.sum(3, keepdim=True)
    return v


def q_reduce_sum(rec, x):
    kq, s = _kq_s(rec, x)
    return ste(A.reduce_sum_axes(rec, x), _sum_like(rec, x * kq * s, x.shape))


def q_reduce_sum_wo_kernel(rec, x):
    groups = int(A._a(rec, "_groups", 1) or 1)
    n, c, h, w = x.shape
    return ste(A.reduce_sum_wo_kernel(rec, x), x.reshape(n, groups, c // groups, h, w).sum(2))


def q_ew_mult_on_mac(rec, a, b):
    y = A.ew_mult_on_mac(rec, a, b)
    if a.shape != b.shape:
        rep = np.asarray(A._a(rec, "input_repeats")).reshape(2, -1)
        a, b = I.apply_repeats(a, rep[0].tolist()), I.apply_repeats(b, rep[1].tolist())
    zp = np.asarray(rec["input_zero_points"], np.float64).ravel()
    zb = float(zp[1]) if zp.size > 1 else float(zp[0])
    s = 2.0 ** -int(np.asarray(rec["_multiplier_shift"]).ravel()[0])
    return ste(y, (a * b - a * zb - b * float(zp[0])) * s)


def q_ew_mult(rec, a, b):
    y = A.ew_mult(rec, a, b)
    if a.shape != b.shape:
        rep = np.asarray(A._a(rec, "input_repeats", [[1, 1, 1], [1, 1, 1]])).reshape(2, -1)
        a, b = I.apply_repeats(a, rep[0].tolist()), I.apply_repeats(b, rep[1].tolist())
    return ste(y, a * b * 2.0 ** -int(np.asarray(rec["_multiplier_shift"]).ravel()[0]))


def q_mock_conv(rec, x):
    kq, s = _kq_s(rec, x)
    return ste(A.mock_conv(rec, x), x * kq * s)


def q_avgpool(rec, x):
    kq, s = _kq_s(rec, x)
    y = A.avgpool(rec, x)
    kh, kw = (int(v) for v in np.asarray(A._a(rec, "kernel_size")).ravel()[:2])
    sh, sw = (int(v) for v in np.asarray(A._a(rec, "strides")).ravel()[:2])
    sur = torch.nn.functional.avg_pool2d(x * kq * s, (kh, kw), (sh, sw)) * (kh * kw)
    return ste(y, sur)


def q_split_low(rec, x):
    return ste(A.split_low(rec, x), x)


def q_split_high(rec, x, low=None):
    b = int(np.asarray(A._a(rec, "low_bits", 8)).ravel()[0])
    return ste(A.split_high(rec, x), x * 2.0 ** -b)


def q_add_bias(rec, x):
    b = _per_c(np.asarray(rec["bias"], np.float64), x)
    bits, signed, wrap = A.lossy(rec)
    y = I.hw_lossy((x + b).detach(), bits, signed, wrap)
    lo, hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)
    if wrap:  # modular element (the bias absorbs a zero-point offset and wraps by design): derivative 1
        return ste(y, x + b)
    inside = ((x.detach() + b >= lo) & (x.detach() + b <= hi)).to(x.dtype)
    return ste(y, (x + b) * inside)


def q_conv_stripped(rec, x):
    """Frozen MAC accumulator (deconv chains): exact value, gradient of the float conv with the integer kernel."""
    from graph import conv_params
    y = A.conv_stripped(rec, x)
    p = conv_params(rec["convrec"], x.device)
    sur = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (int(p["pad"][2]), int(p["pad"][3]), int(p["pad"][0]),
                                                                  int(p["pad"][1]))),
                                     p["k"].to(x.dtype), stride=tuple(p["strides"].tolist()),
                                     dilation=tuple(p["dilation"].tolist()), groups=int(p["groups"]))
    return ste(y, sur * 2.0 ** -float(p["pre_acc_shift"]))


QOPS = {
    "ReduceSumOp": q_reduce_sum, "ReduceMeanOp": q_reduce_sum, "ReduceSumWOKernelOp": q_reduce_sum_wo_kernel,
    "ElementwiseMultOnMacOp": q_ew_mult_on_mac, "ElementwiseMultOp": q_ew_mult, "MockConvOp": q_mock_conv,
    "AvgPoolOp": q_avgpool, "SplitPrecisionLow": q_split_low, "SplitPrecisionHigh": q_split_high,
    "AddBiasOp": q_add_bias, "AddBiasDeconvOp": lambda r, x: q_add_bias(dict(r, bias=np.tile(
        np.asarray(r["bias"]).ravel(), max(1, x.shape[1] // np.asarray(r["bias"]).size))), x),
    "ConvStrippedOp": q_conv_stripped,
}


class QChain(torch.nn.Module):
    """One HN layer as a trainable atomic-op chain. `apu_factory(rec)` returns the trainable APU module for each
    ActivationOp (the net registers it in its `apus` dict)."""

    def __init__(self, chain, apu_factory, dev="cpu"):
        super().__init__()
        self.chain, self.dev = chain, dev  # atomic.Chain: records (NHWC originals) + wiring + out_idx
        self.apu_keys = {}
        for i, (name, r) in enumerate(chain.recs):
            if r["type"] == "ActivationOp":
                self.apu_keys[i] = apu_factory(name, r)

    def forward(self, ins, apus):
        outs = []
        for i, ((name, r), srcs) in enumerate(zip(self.chain.recs, self.chain.wiring)):
            args = [ins[j] if kind == "in" else outs[j] for kind, j in srcs]
            t = r["type"]
            if i in self.apu_keys:
                y = apus[self.apu_keys[i]](args[0])  # float64 in: exact (chain accumulators exceed 2^24)
            elif "perm" in r:
                y = A.gather(r, *args)
            elif t in QOPS:
                y = QOPS[t](r, *args)
            elif t == "ConstOp":
                y = A.const(r).to(self.dev)
            elif t in A.OPS:
                y = A.OPS[t](r, *args)  # exact and differentiable as is (slice, concat, reorder, pad, maxpool, ...)
            else:
                raise NotImplementedError(f"atomic op {t} ({name})")
            outs.append(y)
        if getattr(self.chain, "out_idx", None):
            return [outs[j].to(torch.float32) for j in self.chain.out_idx]
        return outs[-1].to(torch.float32)

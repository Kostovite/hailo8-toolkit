"""Hailo-8 atomic ops (acceleras call_bit_exact semantics) on NCHW float64 integer-valued tensors.

Each function takes the capture record of the op (capture.py ALL_OPS=1 AUTO_ATTRS=1) and its inputs, and returns
the op's integer output. atomic_test.py checks every captured instance against the recorded output.
"""
import numpy as np
import torch

import intops as I


def _a(rec, k, default=None):
    v = rec.get(k, rec.get("attr:" + k, default))
    return v


def lossy(rec, prefix="outq"):
    return int(rec[f"{prefix}_bits"]), bool(rec[f"{prefix}_signed"]), bool(rec[f"{prefix}_wrap"])


def _t(x, like):
    return torch.as_tensor(np.asarray(x, np.float64), device=like.device)


def _per_c(v, like):
    v = _t(v, like)
    return v.reshape(1, -1, 1, 1) if v.numel() > 1 else v


def reduce_sum(rec, x):
    """ReduceSumOp / ReduceMeanOp: round(x * kq / 2^s) per element, sum per channel group (accumulator element)."""
    groups = int(_a(rec, "_groups", 1) or 1)
    return I.reduce_sum_acc(x, rec["kernel_q"], int(rec["pre_acc_shift"]), groups, lossy(rec))


def reduce_sum_wo_kernel(rec, x):
    """ReduceSumWOKernelOp: plain integer sum of channel groups (no kernel, no rounding), int32 wraparound."""
    groups = int(_a(rec, "_groups", 1) or 1)
    n, c, h, w = x.shape
    return I.wrap(x.reshape(n, groups, c // groups, h, w).sum(2), 32, True)


def ew_mult_on_mac(rec, a, b):
    """ElementwiseMultOnMacOp: a*b - a*zp_b - b*zp_a, banker's-rounded >> multiplier_shift."""
    rep = _a(rec, "input_repeats")
    if rep is not None and a.shape != b.shape:  # captured inputs are already repeated (in-place in the op)
        rep = np.asarray(rep).reshape(2, -1)
        a, b = I.apply_repeats(a, rep[0].tolist()), I.apply_repeats(b, rep[1].tolist())
    zp = np.asarray(rec["input_zero_points"], np.float64).ravel()
    zp_a, zp_b = float(zp[0]), float(zp[1]) if zp.size > 1 else float(zp[0])
    post = a * b - a * zp_b - b * zp_a
    s = int(np.asarray(rec["_multiplier_shift"]).ravel()[0])
    if s > 0:
        post = I.bankers_int_shift(post.round().to(torch.int64), s).to(torch.float64)
    return post


def reorder(rec, x):
    """ReorderOp: gather input channels by the recipe; index == C means the quantized constant 1."""
    recipe = np.array([int(np.asarray(r)) for r in np.asarray(rec["recipe"], dtype=object).ravel()])
    sc = float(np.mean(np.asarray(rec["input_scales"], np.float64)))
    zp = float(np.asarray(rec["input_zero_points"], np.float64).ravel()[0])
    one = I.hw_lossy(torch.tensor(1.0 / sc + zp, dtype=torch.float64, device=x.device), *lossy(rec))
    pad = torch.cat([x, one.expand(x.shape[0], 1, x.shape[2], x.shape[3])], 1)
    return pad[:, torch.as_tensor(recipe, device=x.device)]


def slice_(rec, x):
    """SliceOp: [start, stop, step] per NHWC axis (height, width, features)."""
    h0, h1, hs = (int(v) for v in np.asarray(_a(rec, "_height_slice")).ravel())
    w0, w1, ws = (int(v) for v in np.asarray(_a(rec, "_width_slice")).ravel())
    f0, f1, fs = (int(v) for v in np.asarray(_a(rec, "_features_slice")).ravel())
    return x[:, f0:f1:fs, h0:h1:hs, w0:w1:ws]


def concat(rec, *xs):
    """ConcatOp: no requantization. Axis = the one where the inputs differ (features / width / height); with
    _group_sizes, features are interleaved group by group (zero-point compensation channels of a grouped matmul)."""
    gs = _a(rec, "_group_sizes")
    diff = [d for d in (1, 2, 3) if len({x.shape[d] for x in xs}) > 1]
    axis = diff[0] if diff else 1
    if gs is None or axis != 1:
        return torch.cat(xs, axis)
    gs = [int(g) for g in np.asarray(gs).ravel()]
    tot, parts = sum(gs), []
    for i, g in enumerate(gs):
        before = sum(gs[:i])
        for x in xs:
            unit = x.shape[1] // tot
            parts.append(x[:, unit * before: unit * (before + g)])
    return torch.cat(parts, 1)


def const(rec):
    return torch.as_tensor(np.asarray(rec["out0"], np.float64)).permute(0, 3, 1, 2)


def passthru(rec, x):
    return x


def add_bias(rec, x):
    b = np.asarray(rec["bias"], np.float64)
    return I.hw_lossy(x + _per_c(b, x), *lossy(rec))


def activation(rec, x):
    return I.apu(x, _t(rec["thr"], x), _t(rec["mant"], x), _t(rec["shift"], x), _t(rec["offs"], x),
                 int(rec["final_shift"]), bool(rec["mant_signed"]), lossy(rec, "clip1"), lossy(rec, "clip2"),
                 lossy(rec, "out"))


OPS = {
    "ReduceSumOp": reduce_sum, "ReduceMeanOp": reduce_sum, "ReduceSumWOKernelOp": reduce_sum_wo_kernel,
    "ElementwiseMultOnMacOp": ew_mult_on_mac, "ReorderOp": reorder, "SliceOp": slice_, "ConcatOp": concat,
    "PassthruOp": passthru, "AddBiasOp": add_bias, "ActivationOp": activation,
}
OPS["ConstOp"] = const


def space_to_depth(rec, x):
    """SpaceToDepthOp: blocks (hb, wb), channel order CRD (c, dy, dx); the output layout (normal (h', w') grid or
    ViT-style flattened tokens (1, h'*w')) is the recorded output shape."""
    hb, wb = (int(v) for v in np.asarray(_a(rec, "block_sizes", [2, 2])).ravel())
    n, c, h, w = x.shape
    y = x.reshape(n, c, h // hb, hb, w // wb, wb).permute(0, 1, 3, 5, 2, 4)  # n, c, dy, dx, h', w'
    y = y.reshape(n, c * hb * wb, h // hb, w // wb)
    oshape = np.asarray(rec["out0"]).shape  # NHWC
    if oshape[1] != h // hb:
        y = y.reshape(n, c * hb * wb, oshape[1], oshape[2])
    return y


OPS["SpaceToDepthOp"] = space_to_depth


def _nchw(a):
    a = np.asarray(a, np.float64)
    return torch.as_tensor(a).permute(0, 3, 1, 2) if a.ndim == 4 else torch.as_tensor(a)


class Chain:
    """Run one HN layer as its captured atomic-op chain. Wiring (which op input is a layer input or an earlier
    op's output) is found once by exact matching of the captured integer tensors, then reused for new inputs."""

    def __init__(self, recs, layer_ins_ref):
        self.recs = recs  # [(op name, record)] in execution order
        self.wiring = []
        outs = []
        for name, r in recs:
            srcs = []
            for i in range(8 if r["type"] != "ConstOp" else 0):  # ConstOp's input is a dummy (the network input)
                if f"in{i}" not in r:
                    break
                v = np.asarray(r[f"in{i}"], np.float64)
                src = None
                for j in range(len(outs) - 1, -1, -1):  # latest earlier op with an identical output
                    if outs[j].shape == v.shape and np.array_equal(outs[j], v):
                        src = ("op", j)
                        break
                if src is None:
                    rep = r.get("input_repeats")
                    for j, li in enumerate(layer_ins_ref):
                        if li.shape == v.shape and np.array_equal(li, v):
                            src = ("in", j)
                            break
                        if rep is not None and li.ndim == 4:  # captured input is post-repeat: compare repeated
                            rr = np.asarray(rep).reshape(-1, 3)[i]
                            lr = li
                            for ax, k in zip((2, 3, 1), rr):  # (h, w, c) -> NCHW axes
                                lr = np.repeat(lr, int(k), axis=ax)
                            if lr.shape == v.shape and np.array_equal(lr, v):
                                src = ("in", j)
                                break
                if src is None and len(layer_ins_ref) == 1 and not outs:
                    src = ("in", 0)  # first op of a single-input layer
                if src is None:
                    raise NotImplementedError(f"cannot wire input {i} of {name}")
                srcs.append(src)
            self.wiring.append(srcs)
            outs.append(np.asarray(r.get("out0"), np.float64))
            if "perm" in r:  # trust a gather map only if it reproduces the recorded output (reduce ops are
                try:          # "non-arithmetic" classes too, and summing an index tensor is not a gather)
                    args = [torch.as_tensor(layer_ins_ref[j]) if kind == "in" else torch.as_tensor(outs[j])
                            for kind, j in srcs]
                    y_ = gather(r, *args).numpy()
                    ref_ = np.asarray(r["out0"], np.float64)  # NCHW (graph26 converts records) or NHWC (raw)
                    alt_ = _nchw(ref_).numpy() if ref_.ndim == 4 else ref_
                    ok = any(y_.shape == t_.shape and bool((y_ == t_).all()) for t_ in (ref_, alt_))
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok:
                    r.pop("perm")

    def __call__(self, ins):
        outs = []
        for (name, r), srcs in zip(self.recs, self.wiring):
            f = gather if "perm" in r else OPS.get(r["type"])  # a captured gather map is exact for any layout op
            if f is gather and "perm" not in r:
                f = None
            if f is None:
                raise NotImplementedError(f"atomic op {r['type']} ({name})")
            args = [ins[j] if kind == "in" else outs[j] for kind, j in srcs]
            outs.append(f(r) if r["type"] == "ConstOp" else f(r, *args))
        if getattr(self, "out_idx", None):  # multi-output layer (precision splitter): one tensor per output edge
            return [outs[j] for j in self.out_idx]
        return outs[-1]


def gather(rec, *xs):
    """Any data-movement op: out = concat(inputs, NHWC-flattened)[perm - 1]; perm 0 = the op's inserted constant
    (taken from the recorded output). perm comes from capture.py pushing an index tensor through the op."""
    perm = np.asarray(rec["perm"])
    B = int(xs[0].shape[0])
    if perm.shape[0] == 1 and B > 1:  # map recorded on ONE sample (capture batch 1): apply it per sample
        flat = torch.cat([x.permute(0, 2, 3, 1).reshape(B, -1) if x.dim() == 4 else x.reshape(B, -1) for x in xs], 1)
        p = torch.as_tensor(perm.reshape(-1), device=flat.device)
        out = flat[:, (p - 1).clamp(min=0)]
        if (p == 0).any():
            const = torch.as_tensor(np.asarray(rec["out0"], np.float64).reshape(-1), device=flat.device).to(out.dtype)
            out = torch.where((p == 0)[None], const[None], out)
        out = out.reshape((B,) + perm.shape[1:])
        return out.permute(0, 3, 1, 2) if out.dim() == 4 else out
    flat = torch.cat([x.permute(0, 2, 3, 1).reshape(-1) if x.dim() == 4 else x.reshape(-1) for x in xs])
    p = torch.as_tensor(perm.reshape(-1), device=flat.device)
    out = flat[(p - 1).clamp(min=0)]
    if (p == 0).any():
        const = torch.as_tensor(np.asarray(rec["out0"], np.float64).reshape(-1), device=flat.device)
        out = torch.where(p == 0, const, out)
    out = out.reshape(perm.shape)
    return out.permute(0, 3, 1, 2) if out.dim() == 4 else out


for _t_ in ("SpatialReshapeOp", "TransposeWidthFeaturesOp", "TransposeHeightWidthOp", "FeaturesToWidthFeaturesOp",
            "SpatialReshapeHeightFeaturesOp", "FlattenOp", "DepthToSpaceOp", "FeaturePermuteOp", "PaddingOp",
            "ReshapeOp", "SpatialTransposeOp", "FlatToFramesOp"):
    OPS[_t_] = gather


def avgpool(rec, x):
    """AvgPoolOp bit-exact: per element lossy(x * kq / 2^s) (accumulator element), then the plain window sum.
    VALID windows (SAME pads with the quantized padding constant first)."""
    kh, kw = (int(v) for v in np.asarray(_a(rec, "kernel_size")).ravel()[:2])
    sh, sw = (int(v) for v in np.asarray(_a(rec, "strides")).ravel()[:2])
    pad = _a(rec, "padding", "VALID")
    if str(pad).upper() not in ("VALID", "PADDINGTYPE.VALID"):
        raise NotImplementedError(f"avgpool padding {pad}")
    kq = rec["kernel_q"]
    v = I.quant_elem(x * 2.0 ** -int(rec["pre_acc_shift"]) * _per_c(kq, x), *lossy(rec))
    n, c, h, w = v.shape
    u = torch.nn.functional.unfold(v.reshape(n * c, 1, h, w), (kh, kw), stride=(sh, sw))  # (n*c, kh*kw, L)
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    return u.sum(1).reshape(n, c, oh, ow)


OPS["AvgPoolOp"] = avgpool


def split_low(rec, x):
    """SplitPrecisionLow: x & (2^b - 1)."""
    b = int(np.asarray(_a(rec, "low_bits", 8)).ravel()[0])
    return torch.remainder(x, 2 ** b)  # == bitwise and for the non-negative two's-complement view


def split_high(rec, x, low=None):
    """SplitPrecisionHigh: (x & ~(2^b - 1)) >> b == floor(x / 2^b)."""
    b = int(np.asarray(_a(rec, "low_bits", 8)).ravel()[0])
    return torch.div(x, 2 ** b, rounding_mode="floor")


def reduce_max(rec, x):
    """ReduceMaxOp: max over channel groups, or over H / W when the recorded output has those collapsed."""
    o = np.asarray(rec["out0"]).shape  # NHWC
    n, c, h, w = x.shape
    if o[3] != c:
        g = o[3]
        return x.reshape(n, g, c // g, h, w).amax(2)
    y = x
    if o[1] == 1 and h != 1:
        y = y.amax(2, keepdim=True)
    if o[2] == 1 and w != 1:
        y = y.amax(3, keepdim=True)
    return y


def reduce_sum_axes(rec, x):
    """ReduceSumOp / ReduceMeanOp over channels (groups) or over H / W (recorded output shape decides)."""
    o = np.asarray(rec["out0"]).shape
    n, c, h, w = x.shape
    if o[3] != c or (o[1] == h and o[2] == w):
        return reduce_sum(rec, x)
    kq = rec["kernel_q"]
    v = I.quant_elem(x * 2.0 ** -int(rec["pre_acc_shift"]) * _per_c(kq, x), *lossy(rec))
    if o[1] == 1 and h != 1:
        v = v.sum(2, keepdim=True)
    if o[2] == 1 and w != 1:
        v = v.sum(3, keepdim=True)
    return I.quant_elem(v, *lossy(rec))


def mock_conv(rec, x):
    """MockConvOp: per-channel kq, pre-acc shift (APU-style round), accumulator element."""
    kq = _per_c(rec["kernel_q"], x)
    s = torch.full(x.shape, int(rec["pre_acc_shift"]), dtype=torch.long, device=x.device)
    v = I.apu_round_shift(x.long() * kq.long(), s, 3, True)
    return I.hw_lossy(v, *lossy(rec)).to(torch.float64)


def ew_mult(rec, a, b):
    """ElementwiseMultOp: (a * b) / 2^ms through the op's output element."""
    if a.shape != b.shape:
        rep = np.asarray(_a(rec, "input_repeats", [[1, 1, 1], [1, 1, 1]])).reshape(2, -1)
        a, b = I.apply_repeats(a, rep[0].tolist()), I.apply_repeats(b, rep[1].tolist())
    return I.ew_mult_acc(a, b, int(np.asarray(rec["_multiplier_shift"]).ravel()[0]), lossy(rec))


def conv_stripped(rec, x):
    """ConvStrippedOp: the MAC accumulator of the layer's conv (record from capture.py's conv hook, injected by the
    chain builder as rec["convrec"])."""
    from graph import conv_params
    r = rec["convrec"]
    p = conv_params(r, x.device)
    acc = I.mac_conv(x, p["k"], p["pre_acc_shift"], tuple(p["strides"].tolist()), tuple(p["dilation"].tolist()),
                     p["pad"].tolist(), p["pad_value"], groups=int(p["groups"]))
    return I.hw_lossy(acc, int(r["acc_bits"]), bool(r["acc_signed"]), bool(r["acc_wrap"]))


def add_bias_deconv(rec, x):
    """AddBiasDeconvOp: bias add on the s*s*C pre-depth-to-space channels (bias tiled per sub-pixel)."""
    b = np.asarray(rec["bias"], np.float64).ravel()
    c = x.shape[1]
    if b.size != c and c % b.size == 0:
        b = np.tile(b, c // b.size)
    return I.hw_lossy(x + _per_c(b, x), *lossy(rec))


def maxpool(rec, x):
    """MaxPoolOp: VALID or TF-SAME (padding never wins: -inf), kernel / strides from the record."""
    kh, kw = (int(v) for v in np.asarray(_a(rec, "pool_size")).ravel()[:2])
    sh, sw = (int(v) for v in np.asarray(_a(rec, "strides")).ravel()[:2])
    pad = str(_a(rec, "padding", "SAME")).upper()
    if "SAME" in pad:
        n, c, h, w = x.shape
        oh, ow = -(-h // sh), -(-w // sw)
        ph, pw = max((oh - 1) * sh + kh - h, 0), max((ow - 1) * sw + kw - w, 0)
        t, b_, l, r = ph // 2, ph - ph // 2, pw // 2, pw - pw // 2  # TF SAME: extra at the end (SE)
        if "NW" in str(_a(rec, "stride_align", "SE")).upper():
            t, b_, l, r = b_, t, r, l
        x = torch.nn.functional.pad(x, (l, r, t, b_), value=float("-inf"))
    return torch.nn.functional.max_pool2d(x, (kh, kw), (sh, sw))


OPS.update({"SplitPrecisionLow": split_low, "SplitPrecisionHigh": split_high, "ReduceMaxOp": reduce_max,
            "ReduceSumOp": reduce_sum_axes, "ReduceMeanOp": reduce_sum_axes, "MockConvOp": mock_conv,
            "ElementwiseMultOp": ew_mult, "ConvStrippedOp": conv_stripped, "AddBiasDeconvOp": add_bias_deconv,
            "MaxPoolOp": maxpool})


def resize_nearest(rec, x):
    """ResizeNearestNeighborOp: TF nearest (asymmetric): out[i] = in[floor(i * in / out)], any ratio; output size
    from the recorded output."""
    o = np.asarray(rec["out0"]).shape  # NHWC
    n, c, h, w = x.shape
    ih = torch.as_tensor(np.floor(np.arange(o[1]) * h / o[1]).astype(np.int64), device=x.device)
    iw = torch.as_tensor(np.floor(np.arange(o[2]) * w / o[2]).astype(np.int64), device=x.device)
    return x[:, :, ih][:, :, :, iw]


OPS["ResizeNearestNeighborOp"] = resize_nearest


def padding(rec, x):
    """PaddingOp: constant pad [top, bottom, left, right, (features)] with the input zero point (a real 0)."""
    p = [int(v) for v in np.asarray(_a(rec, "padding")).ravel()]
    zp = np.asarray(rec.get("input_zero_points", [0]), np.float64).ravel()
    v = float(zp[0])
    y = torch.nn.functional.pad(x, (p[2], p[3], p[0], p[1]), value=v)
    if len(p) > 5 and (p[4] or p[5]):
        y = torch.nn.functional.pad(y, (0, 0, 0, 0, p[4], p[5]), value=v)
    return y


OPS["PaddingOp"] = padding

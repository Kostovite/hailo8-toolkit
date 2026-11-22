"""Bit-exact torch implementation of the Hailo-8 conv datapath.

LAYER_IN -> MAC (per-product banker's shift) -> 16b wrapping accumulator -> bias add -> APU piecewise-linear
activation -> LAYER_OUT. Semantics are ported from acceleras call_bit_exact (verified against the chip).

All tensors are NCHW integer-valued; arithmetic is carried out in float64 where values exceed 2**24, otherwise
in float32, so standard cuDNN convolutions give exact integer results.
"""
import os
import torch
import torch.nn.functional as F


def wrap(x, bits, signed):
    lo = -(2 ** (bits - 1)) if signed else 0
    return torch.remainder(x - lo, 2 ** bits) + lo


def clip(x, bits, signed, symmetric=False):
    if signed:
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        if symmetric:
            lo += 1
    else:
        lo, hi = 0, 2 ** bits - 1
    return x.clamp(lo, hi)


def hw_lossy(x, bits, signed, wraparound, symmetric=False):
    """acceleras hw_simulation_by_lossy_element: optional wraparound, then saturate."""
    if wraparound:
        x = wrap(x, bits, signed)
    return clip(x, bits, signed, symmetric)


def bankers_int_shift(p, s):
    """acceleras bankers_round_int_shift: round(p / 2**s), ties to even, floor-division semantics."""
    h = 2 ** (s - 1)
    div_plus = torch.div(p + h, 2 ** s, rounding_mode="floor")
    odd = torch.remainder(div_plus, 2)
    tie = (torch.remainder(p, 2 ** s) == h).to(p.dtype)
    return div_plus - odd * tie


def _residue_table(s, dtype, device):
    """G[a, b] = (p+h) mod 2^s + 2^s * tie(p)  with p = a*b mod 2^(s+1).

    sum_k bankers(p_k, s) == (sum_k p_k + K*h - sum_k G[x_k mod M, w_k mod M]) / 2^s, M = 2^(s+1).
    """
    m = 2 ** (s + 1)
    a = torch.arange(m, device=device, dtype=torch.int64)
    p = torch.remainder(a[:, None] * a[None, :], m)
    h = 2 ** (s - 1)
    r = torch.remainder(p + h, 2 ** s)
    q_odd = torch.div(torch.remainder(p + h, m), 2 ** s, rounding_mode="floor")
    tie = (torch.remainder(p, 2 ** s) == h).to(torch.int64) * q_odd
    return (r + 2 ** s * tie).to(dtype)


# Integer-valued convs. cuDNN float32 may pick FFT/Winograd: |err| > .5 on big partial sums, and the pick depends on
# the batch size (flat3d a16 conv: batch 8 != batch 1). HQ_CONV=gemm (default): float32 with cuDNN OFF = im2col + GEMM,
# exact for integer partial sums < 2**24 in any order; f64: float64 (exact, 1/32-rate on Maxwell); cudnn: old behaviour.
HQ_CONV = os.environ.get("HQ_CONV", "gemm")
CONV_DTYPE = torch.float64 if HQ_CONV == "f64" else torch.float32


def _conv2d(x, k, **kw):
    if HQ_CONV == "gemm" and x.is_cuda:
        with torch.backends.cudnn.flags(enabled=False):
            return F.conv2d(x, k, **kw)
    return F.conv2d(x, k, **kw)


def _iconv(x, k, **kw):
    """Integer-valued conv in CONV_DTYPE, rounded to undo Winograd/FFT float error, returned as float64.
    Only for small-magnitude operands (residue features); use nib_conv for activations x kernels."""
    return torch.round(_conv2d(x.to(CONV_DTYPE), k.to(CONV_DTYPE), **kw)).to(torch.float64)


def nib_conv(x, k, **kw):
    """Exact integer conv in float32 regardless of summation order: x = 16*hi + lo (x >= 0, 8-bit), each half-conv
    is bounded by 15 * |k| * taps < 2**24 (a single fp32 conv over 512 taps of 255*127 is NOT exact)."""
    if float(x.abs().max()) > 255 or float(k.abs().max()) > 128:
        return _wide_conv(x, k, **kw)
    hi = torch.div(x, 16, rounding_mode="floor")
    lo = x - 16 * hi
    kk = k.to(CONV_DTYPE)
    return 16 * _iconv(hi, kk, **kw) + _iconv(lo, kk, **kw)


def _wide_conv(x, k, **kw):
    """16-bit operands (a16 inputs, a16_w16 kernels): x in 4 nibbles (top one signed) x k in 2 bytes (top one signed);
    each partial conv is bounded by 15 * 255 * taps < 2**24 and the float64 recombination is exact."""
    xs, r = [], x
    for sh in (12, 8, 4):
        d = torch.div(r, 2 ** sh, rounding_mode="floor")
        xs.append((d, sh))
        r = r - d * 2 ** sh
    xs.append((r, 0))
    kh = torch.div(k, 256, rounding_mode="floor")
    taps = k.shape[1] * k.shape[2] * k.shape[3]
    dt = CONV_DTYPE if 15 * 255 * taps < 2 ** 24 else torch.float64  # float64 beyond 4386 taps
    ks = [(kh.to(dt), 8), ((k - 256 * kh).to(dt), 0)]
    total = 0
    for xd, xsh in xs:
        for kd, ksh in ks:
            total = total + torch.round(_conv2d(xd.to(dt), kd, **kw)).to(torch.float64) * 2.0 ** (xsh + ksh)
    return total


def nib_matmul(d, w):
    """Exact integer matmul in float32: split d into 4-bit halves (attn @ V reaches 784 * 255 * 255 = 5.1e7)."""
    hi = torch.div(d, 16, rounding_mode="floor")
    lo = d - 16 * hi
    ww = w.to(CONV_DTYPE)
    mm = lambda a: torch.round(torch.matmul(a.to(CONV_DTYPE), ww)).to(torch.float64)
    return 16 * mm(hi) + mm(lo)


_FACTOR_CACHE = {}


def _residue_factors(s, device):
    """Exact integer factorization G = A @ B (A: M x r with entries in {-1,0,1,2}, B: r x M rows of G).

    Lets the residue sum use r = 2/3/5 channels per input channel instead of M = 4/8/16.
    """
    key = (s, str(device))
    if key not in _FACTOR_CACHE:
        g = _residue_table(s, torch.float64, "cpu")
        rows = []
        for i in range(g.shape[0]):
            if int(torch.linalg.matrix_rank(g[rows + [i]])) > len(rows):
                rows.append(i)
        b = g[rows]
        a = torch.linalg.lstsq(b.T, g.T).solution.T
        a_int = torch.round(a)
        assert torch.equal(a_int @ b, g), "residue factorization is not integral"
        _FACTOR_CACHE[key] = (a_int.to(device), b.to(device))
    return _FACTOR_CACHE[key]


def mac_conv(x, k, pre_acc_shift, stride, dilation, pad, pad_value, groups=1):
    """Hailo MAC: sum_k bankers(x_k * w_k, s). x: (N,Cin,H,W) ints, k: (Cout,Cin/groups,kh,kw) ints.

    pad = (top, bottom, left, right), filled with pad_value (the input zero point).
    """
    s = int(pre_acc_shift)
    dt = torch.float64
    x = F.pad(x.to(dt), (pad[2], pad[3], pad[0], pad[1]), value=float(pad_value))
    kd = k.to(dt)
    total = nib_conv(x, kd, stride=stride, dilation=dilation, groups=groups)
    if s == 0:
        return total
    m = 2 ** (s + 1)
    n_taps = kd.shape[1] * kd.shape[2] * kd.shape[3]
    fa, fb = _residue_factors(s, x.device)
    r = fa.shape[1]
    x_low = torch.remainder(x, m).long()
    w_low = torch.remainder(kd, m).long()
    # input features A[x_low] (N, Cin*r, H, W); kernel B[:, w_low] so that sum_j A[a,j]B[j,b] = G[a,b]
    xo = fa[x_low].permute(0, 1, 4, 2, 3).reshape(x.shape[0], -1, x.shape[2], x.shape[3])
    co, ci, kh, kw = kd.shape
    gk = fb[:, w_low]  # (r, Cout, Cin/g, kh, kw)
    gk = gk.permute(1, 2, 0, 3, 4).reshape(co, ci * r, kh, kw)
    resid = _iconv(xo, gk, stride=stride, dilation=dilation, groups=groups)
    return (total + n_taps * 2 ** (s - 1) - resid) / 2 ** s


SDK_EDGES = os.environ.get("HAILOQAT_SDK_EDGES") == "1"
SDK_MATMUL = os.environ.get("HAILOQAT_SDK_MATMUL") == "1"  # 1: read the matmul weights operand as uint8, as SDK_BIT_EXACT does


def apu_round_shift(v, shift, max_round_bits, signed):
    """acceleras bankers_round_with_shift on int64: truncate below the top `max_round_bits` dropped bits,
    then ties-to-even. `shift` is an int64 tensor broadcastable to v."""
    br = torch.clamp(shift, max=max_round_bits)
    if signed:
        sign, to_b = None, torch.bitwise_right_shift(v, shift - br)
    else:
        sign, to_b = torch.sign(v), torch.bitwise_right_shift(v.abs(), shift - br)
    floor_x = torch.bitwise_right_shift(to_b, br)
    odd = floor_x & 1
    mod = to_b - torch.bitwise_left_shift(floor_x, br)
    twice = 2 * mod
    base = torch.bitwise_left_shift(torch.ones_like(br), br)
    out = floor_x + (twice > base).to(v.dtype) + (twice == base).to(v.dtype) * odd
    return out if sign is None else sign * out


def apu(acc, thr, mant, shift, offs, final_shift, mant_signed, clip1, clip2, out, post_shift1_rounding=6):
    """Hailo APU piecewise-linear activation, integer arithmetic. acc: (N,C,H,W); thr: (P+1,C); mant/shift/offs: (P,C).

    clip1/clip2/out are (bits, signed, wraparound) tuples; out is saturated symmetric iff signed.
    """
    n, c, h, w = acc.shape
    g_ = thr.shape[1] if thr.dim() == 2 else 1
    if g_ != c:  # one table per quantization group (e.g. per attention head): groups cover contiguous channels
        assert c % g_ == 0, (c, g_)
        rep = lambda tab: tab.reshape(tab.shape[0], -1).repeat_interleave(c // g_, 1)  # noqa: E731
        thr, mant, shift, offs = rep(thr), rep(mant), rep(shift), rep(offs)
    a = acc.long().permute(1, 0, 2, 3).reshape(c, -1)  # C x NHW; int64: 32-bit (a16_w16) accumulators overflow int32 in mant*acc
    t = thr.to(torch.float64).T.contiguous()  # C x (P+1)
    af = a.to(torch.float64)
    idx = torch.searchsorted(t[:, 1:-1].contiguous(), af.contiguous(), right=True)  # segment index 0..P-1
    # silicon extends the first/last segment past the table ends (chip map, 451k softmax rows); SDK_BIT_EXACT zeroes
    # them instead (wrong at APU inputs -32768 / 32767). sdk_edges=1 reproduces the SDK.
    valid = (af >= t[:, :1]) & (af < t[:, -1:]) if SDK_EDGES else torch.ones_like(af, dtype=torch.bool)
    g = lambda tab: torch.gather(tab.long().T.contiguous(), 1, idx) * valid.long()
    m, sh, of = g(mant), g(shift), g(offs)
    v = apu_round_shift(m * a, sh, post_shift1_rounding, mant_signed)
    v = hw_lossy(v, *clip1)
    v = hw_lossy(v + of, *clip2)
    v = apu_round_shift(v, torch.full_like(v, int(final_shift)), int(final_shift), False)
    v = hw_lossy(v, out[0], out[1], out[2], symmetric=out[1])
    return v.reshape(c, n, h, w).permute(1, 0, 2, 3).to(torch.float64)


def conv_layer(x, p):
    """Full Hailo conv layer from a capture record dict p (torch tensors). Returns (acc, bias_out, act_out)."""
    acc = mac_conv(x, p["k"], p["pre_acc_shift"], tuple(p["strides"].tolist()), tuple(p["dilation"].tolist()),
                   p["pad"].tolist(), p["pad_value"], groups=int(p["groups"]))
    acc = hw_lossy(acc, int(p["acc_bits"]), bool(p["acc_signed"]), bool(p["acc_wrap"]))
    b = hw_lossy(acc + p["bias"].to(acc.dtype)[None, :, None, None],
                 int(p["bias_bits"]), bool(p["bias_signed"]), bool(p["bias_wrap"]))
    y = apu(b, p["thr"], p["mant"], p["shift"], p["offs"], int(p["final_shift"]), bool(p["mant_signed"]),
            (int(p["clip1_bits"]), bool(p["clip1_signed"]), bool(p["clip1_wrap"])),
            (int(p["clip2_bits"]), bool(p["clip2_signed"]), bool(p["clip2_wrap"])),
            (int(p["out_bits"]), bool(p["out_signed"]), bool(p["out_wrap"])))
    return acc, b, y


def _apu_tail(acc, p):
    """bias add (16b wrapping) + APU, shared by every MAC-based layer."""
    b = hw_lossy(acc + p["bias"].to(acc.dtype)[None, :, None, None],
                 int(p["bias_bits"]), bool(p["bias_signed"]), bool(p["bias_wrap"]))
    y = apu(b, p["thr"], p["mant"], p["shift"], p["offs"], int(p["final_shift"]), bool(p["mant_signed"]),
            (int(p["clip1_bits"]), bool(p["clip1_signed"]), bool(p["clip1_wrap"])),
            (int(p["clip2_bits"]), bool(p["clip2_signed"]), bool(p["clip2_wrap"])),
            (int(p["out_bits"]), bool(p["out_signed"]), bool(p["out_wrap"])))
    return b, y


def concat_groups(layers, n):
    """A concat feeding a grouped matmul's weight input (the DFC materializes the zero-point compensation as
    linear + reduce_sum + concat when it cannot fuse it) is laid out group-interleaved: [a_g0, b_g0, a_g1, b_g1, ...]."""
    for v in layers.values():
        if v["type"] == "matmul" and len(v["input"]) > 1 and v["input"][1] == n:
            return int(v["params"].get("groups", 1))
    return 1


def concat_features(ins, groups=1):
    if groups == 1:
        return torch.cat(ins, dim=1)
    parts = [x.reshape(x.shape[0], groups, x.shape[1] // groups, *x.shape[2:]) for x in ins]
    y = torch.cat(parts, dim=2)
    return y.reshape(y.shape[0], -1, *y.shape[3:])


def feature_split(x, layer):
    """feature_splitter: consecutive channel slices, one per output edge (no requantization; SliceOp is a copy)."""
    sizes = [int(s[-1]) for s in layer["output_shapes"]]
    return list(torch.split(x, sizes, dim=1))


def layer_inputs(out, layers, n):
    """Inputs of layer n; a multi-output producer (feature_splitter) stores a list indexed like its output edges."""
    ins, seen = [], {}
    for i in layers[n]["input"]:
        y = out[i]
        if isinstance(y, list):
            k = seen.get(i, 0)
            seen[i] = k + 1
            y = y[[j for j, o in enumerate(layers[i]["output"]) if o == n][k]]
        ins.append(y)
    return ins


def apply_repeats(x, repeats):
    """acceleras repeat_inputs: tf.repeat per NHWC dim (h, w, c) == repeat_interleave on NCHW dims (2, 3, 1)."""
    for nhwc_dim, nchw_dim in ((0, 2), (1, 3), (2, 1)):
        r = int(repeats[nhwc_dim])
        if r != 1:
            x = x.repeat_interleave(r, nchw_dim)
    return x


def ew_add_layer(x0, x1, p):
    """Hailo elementwise add/sub: each input * per-channel int factor, APU-style rounded shift, 16b wrap, sum.
    ew_sub is the same op with the factor sign folded into ew_k."""
    if "ew_repeats" in p:
        x0, x1 = apply_repeats(x0, p["ew_repeats"][0].tolist()), apply_repeats(x1, p["ew_repeats"][1].tolist())
    k = p["ew_k"].long()
    s = torch.full(x0.shape, int(p["ew_shift"]), dtype=torch.long, device=x0.device)
    acc_cfg = (int(p["ewacc_bits"]), bool(p["ewacc_signed"]), bool(p["ewacc_wrap"]))
    terms = []
    for x, kk in ((x0, k[0]), (x1, k[1])):
        v = apu_round_shift(x.long() * kk[None, :, None, None], s, 3, True)
        terms.append(hw_lossy(v, *acc_cfg))
    acc = (terms[0] + terms[1]).to(torch.float64)
    b, y = _apu_tail(acc, p)
    return acc, b, y


def quant_elem(x, bits, signed, wraparound):
    """acceleras QuantElement in bit-exact mode: round half to even, optional wraparound, saturate."""
    return hw_lossy(torch.round(x), bits, signed, wraparound)


def mac_matmul(d, w, s):
    """sum_k bankers(d[..., i, k] * w[..., k, j], s) exactly. d: (..., I, K), w: (..., K, J) integer-valued.

    Same residue factorization as mac_conv: one plain matmul + one rank-r correction matmul.
    """
    dd, wd = d.to(torch.float64), w.to(torch.float64)
    total = nib_matmul(dd, wd)
    if s == 0:
        return total
    m = 2 ** (s + 1)
    fa, fb = _residue_factors(s, d.device)  # A: M x r, B: r x M
    r = fa.shape[1]
    da = fa[torch.remainder(dd, m).long()]  # (..., I, K, r)
    wb = fb[:, torch.remainder(wd, m).long()]  # (r, ..., K, J)
    da = da.reshape(*d.shape[:-1], d.shape[-1] * r)  # (..., I, K*r) k-major
    wb = wb.movedim(0, -2).reshape(*w.shape[:-2], w.shape[-2] * r, w.shape[-1])  # (..., K*r, J)
    resid = torch.round(torch.matmul(da.to(CONV_DTYPE), wb.to(CONV_DTYPE))).to(torch.float64)
    k = d.shape[-1]
    return (total + k * 2 ** (s - 1) - resid) / 2 ** s


def mock_conv(x, kq, s, outq):
    """MockConvOp bit-exact: x * k, APU-style rounded shift (3 bits), saturate to the accumulator element."""
    v = apu_round_shift(x.long() * kq.long(), torch.full_like(x, int(s), dtype=torch.long), 3, True)
    return hw_lossy(v, *outq)


def conv_add_term(l3, factor, s, feed_repeat, cfg):
    """HailoConvAdd residual into the accumulator: wrap(round(L3 * factor / 2**s)) * feed_repeat (float32 math,
    as acceleras ElementwiseAddOp.call_hw_sim in bit-exact mode).

    factor is the INTEGER per-channel multiplier round(desired / feed_repeat) set by enforce_encoding(); a freshly
    built acceleras model still holds the float pre-quantization value, which is off by up to 84 LSB."""
    v = (l3.to(torch.float32) * factor.to(torch.float32)[None, :, None, None]) / torch.tensor(2.0 ** s, dtype=torch.float32)
    return quant_elem(v.to(torch.float64), *cfg) * float(feed_repeat)


def conv_add_layer(x, l3, p):
    """HailoConvAdd: MAC conv + integer-factor residual in the accumulator, then bias + APU."""
    acc = mac_conv(x, p["k"], p["pre_acc_shift"], tuple(p["strides"].tolist()), tuple(p["dilation"].tolist()),
                   p["pad"].tolist(), p["pad_value"], groups=int(p["groups"]))
    acc = hw_lossy(acc, int(p["acc_bits"]), bool(p["acc_signed"]), bool(p["acc_wrap"]))
    term = conv_add_term(l3, torch.round(p["ca_factor"].to(torch.float64)), float(p["ca_shift"]), float(p["ca_feed"]),
                         (int(p["acc_bits"]), True, True))
    if os.environ.get("HQ_CONVADD_SUM", "f32") == "f32":  # HSim: f32(f32(acc) + f32(term)); differs from integer math
        s_ = (acc.to(torch.float32) + term.to(torch.float32)).to(torch.float64)  # only once |acc| > 2**24 (a16 layers)
    else:  # "exact": pure integer sum (candidate for what the silicon does; check against the chip)
        s_ = acc + term
    pre = hw_lossy(s_, int(p["acc_bits"]), bool(p["acc_signed"]), bool(p["acc_wrap"]))
    b, y = _apu_tail(pre, p)
    return acc, pre, b, y


def matmul_layer_acc(x0, x1, groups, transpose, zp_comp_added, zp_rank, feed_repeat, zp0, s, accq, windows=(1, 1),
                     x1_bits=None):
    """Windowed matmul (acceleras _input_windows (wh, ww)): H and W are split into wh x ww contiguous blocks, each an
    independent matmul over its own tokens; the windows are folded into the batch and unfolded afterwards."""
    wh, ww = int(windows[0]), int(windows[1])
    if wh == 1 and ww == 1:
        return _matmul_layer_acc(x0, x1, groups, transpose, zp_comp_added, zp_rank, feed_repeat, zp0, s, accq, x1_bits)

    def fold(x):
        n, c, h, w = x.shape
        return x.reshape(n, c, wh, h // wh, ww, w // ww).permute(0, 2, 4, 1, 3, 5).reshape(n * wh * ww, c, h // wh, w // ww)
    n, _, h, w = x0.shape
    y = _matmul_layer_acc(fold(x0), fold(x1), groups, transpose, zp_comp_added, zp_rank, feed_repeat, zp0, s, accq, x1_bits)
    c = y.shape[1]
    return y.reshape(n, wh, ww, c, h // wh, w // ww).permute(0, 3, 1, 4, 2, 5).reshape(n, c, h, w)


def _matmul_layer_acc(x0, x1, groups, transpose, zp_comp_added, zp_rank, feed_repeat, zp0, s, accq, x1_bits=None):
    """acceleras MatmulOp bit-exact accumulator (input_windows 1x1x1, input tiles 1). NCHW in, NCHW out.

    Per group: data (T, f0) @ weights (f1, T') with per-product round(d*w/2^s), plus the zero-point compensation
    residue sum_r round(zp0 * zpc_r / 2^s) * feed_r added per output column; then the accumulator element.
    Output channel = g * J + j (acceleras _prepare_matmul_output).
    """
    n, c0, h, w = x0.shape
    t = h * w
    d = x0.permute(0, 2, 3, 1).reshape(n, t, groups, c0 // groups).permute(0, 2, 1, 3)  # N,g,T,f0
    c1 = x1.shape[1]
    wt = x1.permute(0, 2, 3, 1).reshape(n, x1.shape[2] * x1.shape[3], groups, c1 // groups).permute(0, 2, 1, 3)
    zpc = None
    if zp_comp_added:
        zpc, wt = wt[..., -zp_rank:], wt[..., :-zp_rank]  # zpc: N,g,T',rank
    if x1_bits == 8 and not SDK_MATMUL:  # silicon reads an 8-bit weights operand as int8 (uint8 > 127 wraps negative)
        wt = wrap(wt, 8, True)
    if transpose:
        wt = wt.transpose(-1, -2)  # N,g,f1,T'
    elif zp_comp_added:
        raise NotImplementedError("zp compensation on a non-transposed matmul")
    acc = mac_matmul(d, wt, int(s))  # N,g,T,J
    if zpc is not None:
        fr = torch.as_tensor(feed_repeat, dtype=torch.float64, device=x0.device).reshape(-1)
        res = torch.zeros_like(acc[:, :, :1, :])
        for r in range(zp_rank):
            term = quant_elem(float(zp0) * zpc[..., r].to(torch.float64) / 2.0 ** int(s), *accq)  # N,g,T'
            res = res + (term * fr[r if fr.numel() > 1 else 0])[:, :, None, :]
        acc = acc + res
    acc = quant_elem(acc, *accq)
    j = acc.shape[-1]
    return acc.permute(0, 2, 1, 3).reshape(n, t, groups * j).permute(0, 2, 1).reshape(n, groups * j, h, w)


def reduce_max_layer(x, groups):
    """Max over each channel group (reduce_axes=[features]). Pure integer."""
    n, c, h, w = x.shape
    return x.reshape(n, groups, c // groups, h, w).amax(2)


def reduce_sum_acc(x, kq, s, groups, accq):
    """acceleras ReduceSumOp bit-exact: round(x * kq / 2^s) per element (accumulator element), sum per channel
    group, then the accumulator element again (default call_bit_exact _quantize)."""
    n, c, h, w = x.shape
    kq = torch.as_tensor(kq, dtype=torch.float64, device=x.device)
    kq = kq.reshape(1, -1, 1, 1) if kq.numel() > 1 else kq
    v = quant_elem(x.to(torch.float64) * 2.0 ** -int(s) * kq, *accq)
    v = v.reshape(n, groups, c // groups, h, w)
    if os.environ.get("HQ_REDUCE_SUM", "f32") == "f32":  # HSim sums in float32 (tree order): differs from exact
        return quant_elem(v.to(torch.float32).sum(2).to(torch.float64), *accq)  # integers once |sum| > 2**24 (davit)
    return quant_elem(v.sum(2), *accq)  # "exact" integer sum


def ew_mult_acc(a, b, ms, multq):
    """acceleras ElementwiseMultOp bit-exact: (a * b) / 2^ms through the op's output element (inputs pre-repeated)."""
    return quant_elem(a.to(torch.float64) * b.to(torch.float64) / 2.0 ** int(ms), *multq)


def f32_quantize(x, sc, zp):
    """IEEE float32 f32(f32(x) / f32(sc)) + f32(zp), as HSim / HailoRT quantize inputs. CUDA float32 division is not
    correctly rounded (1 ulp -> a rounding tie flips 1 of ~10k 16-bit inputs); the float64 quotient rounded to float32
    is exact (53 >= 2*24 + 2 bits), same for the add."""
    import numpy as _np
    import torch as _t
    xf = x.float().double()
    q = (xf / float(_np.float32(sc))).float().double()
    return (q + float(_np.float32(zp))).float().double()

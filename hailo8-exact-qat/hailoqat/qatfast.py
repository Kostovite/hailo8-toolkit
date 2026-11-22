"""Trainable Hailo-8 integer network (QAT) for YOLO26-style .hn graphs.

Forward = Hailo integer arithmetic (APU tables, wraps, clips, conv-add, C2PSA attention); the only approximation
is per-product MAC rounding replaced by rounding the sum (unbiased, ~0.1% accumulator noise). Backward = straight-
through gradients of a linear surrogate per op (MAC: k/2^s; APU: slope of the hit segment, 0 where saturated).
Trainable: integer kernels and biases of every MAC layer (latent floats, STE-rounded). Everything else is frozen,
as it will be in the HEF. Evaluation always goes through graph26.run (fully exact) with export_ints().
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from torch.utils.cpp_extension import load

import atomic as A
import graph26 as G26
import intops as I
from qchain import QChain
from graph import conv_params, to_t
import graph26 as G
from graph26 import _t, lossy_of

_EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hq_ext")
# HQ_EXT_DIR: build/load an alternate kernel source (DIR/hq.cu) for A/B tests without touching the live extension
_EXT_DIR = os.environ.get("HQ_EXT_DIR", _EXT)
_PRE = os.path.join(_EXT, "prebuilt", "hq_ext.so")  # AOT build (hq_ext/build_prebuilt.sh): sm_52 + sm_86 + PTX


def _load_ext():
    if _EXT_DIR == _EXT and os.path.exists(_PRE) and os.environ.get("HQ_EXT_JIT") != "1":
        import importlib.util
        try:
            spec = importlib.util.spec_from_file_location("hq_ext", _PRE)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod
        except (ImportError, OSError) as e:  # built against another torch: fall back to the JIT build
            print(f"[qatfast] prebuilt hq_ext unusable ({str(e)[:80]}); JIT build", flush=True)
    return load("hq_ext" if _EXT_DIR == _EXT else "hq_ext_alt", [os.path.join(_EXT_DIR, "hq.cu")],
                extra_cuda_cflags=["-O3", "-arch=sm_52"], build_directory=_EXT_DIR, verbose=False)


hq = _load_ext()
_EMPTY = {}


def _empty(dev):
    if dev not in _EMPTY:
        _EMPTY[dev] = torch.empty(0, device=dev)
    return _EMPTY[dev]


def _cfg(c):
    return [int(c[0]), int(bool(c[1])), int(bool(c[2]))] if c else []


def ste(exact, surrogate):
    """Value of `exact` (bit-for-bit: s - s is exactly 0 in IEEE), gradient of `surrogate`.
    The usual `s + (e - s).detach()` is NOT exact in float32 when |s| ~ 1e4 and breaks downstream integer ops."""
    return exact.detach() + (surrogate - surrogate.detach())


def ste_round(x):
    return ste(torch.round(x), x)


def lossy_ste(x, cfg, sym=False):
    """hw_lossy forward; identity gradient inside the representable range, 0 where it saturates."""
    bits, signed, wrap = cfg
    y = I.hw_lossy(x.detach(), bits, signed, wrap, sym)
    if wrap:  # wraparound element: modular, derivative 1 (masking would zero the gradient of every wrapped value)
        return ste(y, x)
    lo, hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)
    inside = ((x.detach() >= lo) & (x.detach() <= hi)).to(x.dtype)
    return ste(y, x * inside)


class APU(nn.Module):
    """Exact APU forward; gradient = mant / 2^(shift + final_shift) of the hit segment, 0 where the output saturates."""

    def __init__(self, rec, dev):
        super().__init__()
        for k in ("thr", "mant", "shift", "offs"):
            self.register_buffer(k, _t(rec[k], dev))
        c = int(np.asarray(rec.get("act_in", rec.get("in0", np.zeros((1, 1, 1, 1))))).shape[-1])
        g_ = self.thr.shape[1] if self.thr.dim() == 2 else 1
        if g_ != c and c % g_ == 0:  # one table per quantization group (or one shared table): expand to channels
            for k in ("thr", "mant", "shift", "offs"):
                setattr(self, k, getattr(self, k).reshape(getattr(self, k).shape[0], -1).repeat_interleave(c // g_, 1))
        self.final_shift = int(rec["final_shift"])
        self.mant_signed = bool(rec["mant_signed"])
        self.clip1, self.clip2, self.out = lossy_of(rec, "clip1"), lossy_of(rec, "clip2"), lossy_of(rec, "out")
        big = float(2 ** 30)  # APU inputs are <= 16 bits, so clamping the thresholds here keeps every comparison
        self.register_buffer("thr_i", torch.ceil(self.thr.T.clamp(-big, big)).int().contiguous())
        self.register_buffer("mant_i", self.mant.T.int().contiguous())
        self.register_buffer("shift_i", self.shift.T.int().contiguous())
        self.register_buffer("offs_i", self.offs.T.int().contiguous())
        assert int(self.shift_i.min()) >= 0 and self.final_shift >= 0
        self.trainable = False

    def enable_training(self, m_radius=1e9, o_radius=1e9):
        """Make the piecewise table trainable: one mantissa and one offset per segment, shared by every channel of the
        layer (the hardware stores one table per channel group). Bounded by the 10-bit mantissa field, 1.5x the original
        offset magnitude (>= 4095) and optional radii around the level-0 table."""
        for t in (self.mant_i, self.offs_i):
            assert bool((t == t[:1]).all()), "per-channel APU tables are not supported"
        m0, o0 = self.mant_i[0].float(), self.offs_i[0].float()
        mhw = 1023.0
        mlo = torch.full_like(m0, -mhw if self.mant_signed else 0.0)
        olim = max(4095.0, 1.5 * float(o0.abs().max()))
        self.register_buffer("m_lo", torch.maximum(mlo, m0 - m_radius))
        self.register_buffer("m_hi", torch.minimum(torch.full_like(m0, mhw), m0 + m_radius))
        self.register_buffer("o_lo", torch.maximum(torch.full_like(o0, -olim), o0 - o_radius))
        self.register_buffer("o_hi", torch.minimum(torch.full_like(o0, olim), o0 + o_radius))
        self.mant_lat = nn.Parameter(m0.clone())
        self.offs_lat = nn.Parameter(o0.clone())
        self.trainable = True

    def tables(self):
        """Current integer (mant, offs), each C x P."""
        if not self.trainable:
            return self.mant_i, self.offs_i
        c = self.mant_i.shape[0]
        m = torch.round(torch.minimum(torch.maximum(self.mant_lat.detach(), self.m_lo), self.m_hi))
        o = torch.round(torch.minimum(torch.maximum(self.offs_lat.detach(), self.o_lo), self.o_hi))
        return m.int()[None].expand(c, -1).contiguous(), o.int()[None].expand(c, -1).contiguous()

    def lat(self):
        return (self.mant_lat, self.offs_lat) if self.trainable else (None, None)

    def args(self):
        m, o = self.tables()
        return (self.thr_i, m, self.shift_i, o, self.final_shift, self.mant_signed,
                _cfg(self.clip1), _cfg(self.clip2), _cfg(self.out))

    def fused(self, x, bias=None, bias_cfg=()):
        """[bias add + lossy] + APU in one kernel; gradient flows to x and, if enabled, to the APU table."""
        return ValFn.apply(x, self, bias, bias_cfg, *self.lat())

    def forward(self, acc):
        with torch.no_grad():
            y = I.apu(acc.detach(), self.thr, self.mant, self.shift, self.offs, self.final_shift, self.mant_signed,
                      self.clip1, self.clip2, self.out)
            n, c, h, w = acc.shape
            a = acc.detach().permute(1, 0, 2, 3).reshape(c, -1)
            t = self.thr.T.contiguous()
            idx = torch.searchsorted(t[:, 1:-1].contiguous(), a.contiguous(), right=True)
            m = torch.gather(self.mant.T.contiguous(), 1, idx)
            sh = torch.gather(self.shift.T.contiguous(), 1, idx)
            slope = (m * torch.pow(2.0, -(sh + self.final_shift))).reshape(c, n, h, w).permute(1, 0, 2, 3)
            bits, signed, _ = self.out
            lo, hi = (-(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)
            slope = slope * ((y > lo) & (y < hi)).to(slope.dtype)
        return ste(y.to(acc.dtype), acc * slope.to(acc.dtype))


def exact_mac(xp, k, total, s, stride, dil, groups):
    """sum_k bankers(x_k w_k, s) in float32 (exact: |partial sums| < 2**24), reusing the plain conv `total`."""
    if s == 0:
        return total
    m = 2 ** (s + 1)
    fa, fb = I._residue_factors(s, xp.device)
    r = fa.shape[1]
    xa = fa.float()[torch.remainder(xp, m).long()].permute(0, 1, 4, 2, 3).reshape(xp.shape[0], -1, xp.shape[2], xp.shape[3])
    co, ci, kh, kw = k.shape
    gk = fb.float()[:, torch.remainder(k, m).long()].permute(1, 2, 0, 3, 4).reshape(co, ci * r, kh, kw)
    resid = torch.round(I._conv2d(xa, gk, stride=stride, dilation=dil, groups=groups))
    return (torch.round(total) + ci * kh * kw * 2 ** (s - 1) - resid) / 2 ** s


def exact_matmul(d, w, s):
    """mac_matmul in float32 with the nibble-split total (exact in any summation order)."""
    total = I.nib_matmul(d, w).float()
    if s == 0:
        return total
    m = 2 ** (s + 1)
    fa, fb = I._residue_factors(s, d.device)
    r = fa.shape[1]
    da = fa.float()[torch.remainder(d, m).long()].reshape(*d.shape[:-1], d.shape[-1] * r)
    wb = fb.float()[:, torch.remainder(w, m).long()].movedim(0, -2).reshape(*w.shape[:-2], w.shape[-2] * r, w.shape[-1])
    resid = torch.round(torch.matmul(da, wb))
    return (total + d.shape[-1] * 2 ** (s - 1) - resid) / 2 ** s


_FACTOR_C = {}


def residue_factors_c(s, device):
    """G = 1 * G[0, :] + A @ B with integer A (entries in {-1..2}) and B: the weight-only row term is a per-output-
    channel constant (sum over the kernel's taps of G[0, k mod M]), so the conv only needs rank(G - 1 G[0]) channels
    per input channel: 1/2/4 instead of 2/3/5 for s = 1/2/3."""
    key = (s, str(device))
    if key not in _FACTOR_C:
        g = I._residue_table(s, torch.float64, "cpu")
        g0 = g[0].clone()
        gp = g - g0[None]
        rows = []
        for i in range(gp.shape[0]):
            if int(torch.linalg.matrix_rank(gp[rows + [i]])) > len(rows):
                rows.append(i)
        b = gp[rows]
        a = torch.round(torch.linalg.lstsq(b.T, gp.T).solution.T)
        assert torch.equal(a @ b + g0[None], g), "constant-row residue factorization is not integral"
        _FACTOR_C[key] = (a.float().to(device), b.float().to(device), g0.float().to(device))
    return _FACTOR_C[key]


def _apu_grads(ctx, gy):
    if not ctx.want_t:
        return None, None
    gm, go = hq.apu_tgrad(gy.contiguous(), ctx.aux_a, ctx.aux_p, ctx.apu.shift_i, ctx.apu.final_shift)
    return gm.sum(0), go.sum(0)


class ValFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, apu, bias, bias_cfg, m_lat=None, o_lat=None):
        e = _empty(x.device)
        want = x.requires_grad
        ctx.want_t = m_lat is not None and m_lat.requires_grad
        y, gs, _, _, aa, ap = hq.tail(x.contiguous(), e, 0, 0.0, 0, e, e, 1.0, 1.0, [], False,
                                      e if bias is None else bias.contiguous(), _cfg(bias_cfg), *apu.args(), want, e,
                                      e, ctx.want_t)
        ctx.want, ctx.apu, ctx.gs, ctx.aux_a, ctx.aux_p = want, apu, gs, aa, ap
        return y

    @staticmethod
    def backward(ctx, gy):
        gm, go = _apu_grads(ctx, gy)
        return (gy * ctx.gs if ctx.want else None), None, None, None, gm, go


class RsMulFn(torch.autograd.Function):
    """lossy(bankers_round_shift(x * k[c], s, 3)) with gradient k/2^s inside the representable range."""

    @staticmethod
    def forward(ctx, x, k, s, cfg):
        want = x.requires_grad
        v, gs = hq.rshift_mul(x.contiguous(), k.contiguous(), int(s), _cfg(cfg), want)
        if want:
            ctx.save_for_backward(gs)
        return v

    @staticmethod
    def backward(ctx, gv):
        (gs,) = ctx.saved_tensors
        return gv * gs, None, None, None


class MacFn(torch.autograd.Function):
    """Exact MAC layer (per-product banker's shift, conv-add, 16b accumulator, bias, APU) in 1 cuDNN conv + 2 fused
    kernels when 255 * max sum|k| + residue bound < 2^24 (single conv over [x, A[x mod M]] with kernel [k, -B[k mod M]]),
    else the nibble split: [hi; lo] conv + residue conv. Backward = the qatnet surrogate d acc/d z with
    z = conv(xp, k)/2^s, without running z forward."""

    @staticmethod
    def forward(ctx, xp, k, b, res, mod, m_lat=None, o_lat=None):
        xp, k = xp.contiguous(), k.contiguous()
        s = mod.s
        e = _empty(xp.device)
        kw = dict(stride=mod.stride, dilation=mod.dil, groups=mod.groups)
        co, ci, kh, kwd = k.shape
        cst = float(ci * kh * kwd * 2 ** (s - 1)) if s > 0 else 0.0
        R = e
        wo = e
        if s > 0 and mod.single:
            kmod = torch.remainder(k, 2 ** (s + 1)).long()
            gk = mod.fbc[:, kmod].permute(1, 2, 0, 3, 4)
            kcat = torch.cat([k[:, :, None], -gk], 2).reshape(co, ci * (mod.rc + 1), kh, kwd)
            src = I._conv2d(hq.expand_cat(xp, mod.fac, s, mod.flag), kcat, **kw)
            wo = mod.g0c[kmod].sum((1, 2, 3)).contiguous()
            mode = 2
        else:
            hl, xa = hq.expand(xp, mod.fa if s > 0 else e, s, mod.planes)
            src = I._conv2d(hl, k, **kw)
            if s > 0:
                gk = mod.fb[:, torch.remainder(k, 2 ** (s + 1)).long()].permute(1, 2, 0, 3, 4).reshape(co, ci * mod.r, kh, kwd)
                R = I._conv2d(xa, gk, **kw).contiguous()
            mode = 1 if mod.planes == 2 else 3
        want = any(t is not None and t.requires_grad for t in (xp, k, b, res))
        ctx.want_t = m_lat is not None and m_lat.requires_grad
        ctx.apu = mod.apu
        if res is not None:
            ca_s, feed = float(mod.conv_add[1]), float(mod.conv_add[2])
            resc, f, sca_inv = res.contiguous(), mod.ca_factor.contiguous(), 2.0 ** -ca_s
        else:
            resc, f, sca_inv, feed = e, e, 1.0, 1.0
        y, gs, code, part, ctx.aux_a, ctx.aux_p = hq.tail(
            src.contiguous(), R, mode, cst, s, resc, f, sca_inv, feed, _cfg(mod.acc), True, b.contiguous(),
            _cfg(mod.bias_cfg), *mod.apu.args(), want, mod.flag, wo, ctx.want_t)
        ovf = part.sum() / y.numel()
        if want:
            ctx.save_for_backward(xp, k, gs, code)
            ctx.mod, ctx.has_res, ctx.n = mod, res is not None, y.numel()
        return y, ovf

    @staticmethod
    def backward(ctx, gy, govf):
        xp, k, gs, code = ctx.saved_tensors
        mod = ctx.mod
        want_res = ctx.has_res and ctx.needs_input_grad[3]
        ca_s, feed = (float(mod.conv_add[1]), float(mod.conv_add[2])) if ctx.has_res else (0.0, 1.0)
        gz, grad_res, part = hq.mac_bwd(gy.contiguous(), gs, code,
                                        govf.float().contiguous() if govf is not None else _empty(gy.device), mod.s, want_res,
                                        mod.ca_factor if want_res else _empty(gy.device), feed, 2.0 ** -ca_s,
                                        bool(ctx.needs_input_grad[2]))
        grad_b = part.sum((0, 2)) if ctx.needs_input_grad[2] else None
        if not want_res:
            grad_res = None
        gx, gk, _ = torch.ops.aten.convolution_backward(
            gz, xp, k, None, list(mod.stride), [0, 0], list(mod.dil), False, [0, 0], mod.groups,
            [ctx.needs_input_grad[0], ctx.needs_input_grad[1], False])
        gm, go = _apu_grads(ctx, gy)
        return gx, gk, grad_b, grad_res, None, gm, go


class MACConv(nn.Module):
    """conv / dw / normalization (+ optional conv-add residual) -> bias -> APU, with trainable int kernel and bias."""

    def __init__(self, r, dev, conv_add=None):
        super().__init__()
        p = {**to_t(r, dev), **conv_params(r, dev)}
        self.kernel = nn.Parameter(p["k"].float().contiguous())
        self.bias = nn.Parameter(p["bias"].float().contiguous())
        kmax = float(p["k"].abs().max())
        self.kmin, self.kmax = -128.0, 127.0
        self.s = int(p["pre_acc_shift"])
        self.stride, self.dil = tuple(p["strides"].tolist()), tuple(p["dilation"].tolist())
        self.pad, self.pad_value = [int(v) for v in p["pad"].tolist()], float(p["pad_value"])
        self.groups = int(p["groups"])
        self.acc = (int(p["acc_bits"]), bool(p["acc_signed"]), bool(p["acc_wrap"]))
        self.bias_cfg = (int(p["bias_bits"]), bool(p["bias_signed"]), bool(p["bias_wrap"]))
        self.apu = APU(r, dev)
        self.conv_add = conv_add  # (int factor tensor, shift, feed)
        if conv_add is not None:
            self.register_buffer("ca_factor", torch.round(conv_add[0].float()))
        self.overflow = torch.zeros((), device=dev)
        if self.s > 0:
            fa, fb = I._residue_factors(self.s, dev)
            self.register_buffer("fa", fa.float().contiguous())
            self.register_buffer("fb", fb.float().contiguous())
            self.r = int(fa.shape[1])
            fac, fbc, g0c = residue_factors_c(self.s, dev)
            self.register_buffer("fac", fac.contiguous())
            self.register_buffer("fbc", fbc.contiguous())
            self.register_buffer("g0c", g0c.contiguous())
            self.rc = int(fac.shape[1])
            taps = int(np.prod(p["k"].shape[1:]))
            self.res_bound = taps * float((fac.abs() @ fbc.abs()).max())
        else:
            self.res_bound = 0.0
        self.single, self.unsafe, self.flag = False, False, None
        # 16-bit (a16) producers feed inputs up to 32767: split into 4 nibble planes and never use the single conv
        self.planes = 4 if "conv_in" in r.files and float(np.asarray(r["conv_in"]).max()) > 255.0 else 2

    def int_kernel(self):
        return ste_round(self.kernel.clamp(self.kmin, self.kmax))

    def int_bias(self):
        return ste_round(self.bias.clamp(-32768.0, 32767.0))

    @torch.no_grad()
    def set_ints(self, k, b):
        """Load integer kernel / bias (export_ints() layout) into the latent parameters."""
        self.kernel.data.copy_(k.to(self.kernel.device, torch.float32).reshape(self.kernel.shape))
        self.bias.data.copy_(b.to(self.bias.device, torch.float32).reshape(self.bias.shape))

    def forward(self, x, res=None):
        xp = F.pad(x, (self.pad[2], self.pad[3], self.pad[0], self.pad[1]), value=self.pad_value) if any(self.pad) else x
        y, self.overflow = MacFn.apply(xp, self.int_kernel(), self.int_bias(), res, self, *self.apu.lat())
        return y


def wide_apu_tables(rec, dev):
    """int64 thresholds (ceil: integer inputs) + int32 mant / shift / offs, channel-major, for hq.wide_tail."""
    t = torch.as_tensor(np.ceil(np.asarray(rec["thr"], np.float64)), dtype=torch.int64, device=dev).T.contiguous()
    g = lambda k: torch.as_tensor(np.asarray(rec[k], np.float64), dtype=torch.int32, device=dev).T.contiguous()
    return (t, g("mant"), g("shift"), g("offs"), int(rec["final_shift"]), bool(rec["mant_signed"]),
            list(lossy_of(rec, "clip1")), list(lossy_of(rec, "clip2")), list(lossy_of(rec, "out")))


class WideFn(torch.autograd.Function):
    """a16_w16 conv in two fp32 cuDNN convs (input digit planes batch-stacked x the kernel's high / low byte), then
    hq.wide_tail: mod-2^32 combine = the 32-bit wrapping accumulator, 32-bit bias, int64 APU, backward slope.
    Backward: d acc / d z with z = conv(xp, k) (pre_acc_shift is 0 on these layers), cuDNN convolution_backward."""

    @staticmethod
    def forward(ctx, xp, k, b, mod):
        xp, k = xp.contiguous(), k.contiguous()
        kh = torch.floor(k / 256.0)
        kw = dict(stride=mod.stride, dilation=mod.dil, groups=mod.groups)
        xd = hq.expand_wide(xp, mod.P, mod.bits)
        hi = I._conv2d(xd, kh, **kw).contiguous()
        lo = I._conv2d(xd, k - 256.0 * kh, **kw).contiguous()
        y, gs = hq.wide_tail(hi, lo, _empty(xp.device), 0, mod.P, mod.bits, mod.s, list(mod.acc), b.contiguous(),
                             mod.bfac, mod.brep, list(mod.bias_cfg), *mod.tabs, mod.flag)
        ctx.save_for_backward(xp, k, gs)
        ctx.mod = mod
        return y

    @staticmethod
    def backward(ctx, gy):
        xp, k, gs = ctx.saved_tensors
        mod = ctx.mod
        gz = (gy * gs).contiguous()
        gx, gk, _ = torch.ops.aten.convolution_backward(
            gz, xp, k, None, list(mod.stride), [0, 0], list(mod.dil), False, [0, 0], mod.groups,
            [ctx.needs_input_grad[0], ctx.needs_input_grad[1], False])
        gb = gz.sum((0, 2, 3)) * mod.BS if ctx.needs_input_grad[2] else None  # b is the integer i, acc += BS * i
        return gx, gk, gb, None


class WideConv(MACConv):
    """a16_w16 conv: 16-bit inputs x 16-bit kernels, 32-bit accumulator and bias (precision_mode=a16_w16).

    Forward is the float64/int64 reference arithmetic (intops.conv_layer), so it is exact for any operand width; these
    layers are few and run at a fraction of the fused path's speed. Kernel latents are in 8-bit-equivalent units
    (kernel / 256); bias latents are the stored 16-bit integer i of bias = step * i (bias_lattice.py)."""
    KS = 256.0

    def __init__(self, r, dev, conv_add=None):
        super().__init__(r, dev, conv_add)
        assert conv_add is None, "conv-add on an a16_w16 layer"
        # storable lattice: bias = f32(f32(factor * i) * repeats), |i| <= imax (annotate the capture with bias_lattice.py)
        assert "bias_factor" in r.files, "a16_w16 capture without bias_factor: run hailoqat/bias_lattice.py HAR CAPTURE"
        self.bfac, self.brep = float(r["bias_factor"]), float(r["bias_repeats"])
        self.BS, self.bimax = self.bfac * self.brep, float(r["bias_imax"])
        self.kernel.data /= self.KS
        self.bias.data /= self.BS
        self.planes = 4  # never the single fused conv
        self.tabs = wide_apu_tables(r, dev)
        taps = int(np.prod(self.kernel.shape[1:]))
        # input digits of `bits` bits x kernel bytes: every partial conv |sum| <= (2^bits - 1) * 255 * taps < 2^24
        self.P, self.bits = (2, 8) if 255 * 255 * taps < 2 ** 24 else (4, 4)
        assert 15 * 255 * taps < 2 ** 24, f"a16_w16 layer with {taps} taps: digit planes not exact in fp32"

    def int_kernel(self):
        return ste_round((self.kernel * self.KS).clamp(-32768.0, 32767.0))

    def int_bias_i(self):
        """The stored 16-bit bias integer i (bias = BS * i)."""
        return ste_round(self.bias.clamp(-self.bimax, self.bimax))

    def int_bias(self):
        """Accumulator bias as stored: float32(float32(factor * i) * repeats), returned in float64."""
        i = self.int_bias_i()
        return ((i.float() * self.bfac) * self.brep).double() + (i - i.detach()).double() * self.BS

    @torch.no_grad()
    def set_ints(self, k, b):
        self.kernel.data.copy_(k.to(self.kernel.device, torch.float32).reshape(self.kernel.shape) / self.KS)
        self.bias.data.copy_((b.to(self.bias.device, torch.float64).reshape(self.bias.shape) / self.BS).float())

    def forward(self, x, res=None):
        xp = F.pad(x, (self.pad[2], self.pad[3], self.pad[0], self.pad[1]), value=self.pad_value) if any(self.pad) else x
        self.overflow = torch.zeros((), device=x.device)
        return WideFn.apply(xp, self.int_kernel(), self.int_bias_i(), self)


class _Rec(dict):
    """An NpzFile read into memory once (NpzFile[key] re-reads and decompresses the zip on every access)."""

    def __init__(self, npz):
        super().__init__({k: npz[k] for k in npz.files})

    @property
    def files(self):
        return list(self.keys())


class QATNet(nn.Module):
    """Builds trainable MAC layers from graph26.Params; non-MAC ops reuse the exact integer kernels with STE."""

    def __init__(self, hn, P, dev):
        super().__init__()
        P.conv = {k: v if isinstance(v, _Rec) else _Rec(v) for k, v in P.conv.items()}
        self.hn, self.P, self.dev = hn, P, dev
        self._cache = {}
        self._flags, self.flag_events = None, []
        layers = hn["layers"]
        self.order = self._order(layers)
        self.mac = nn.ModuleDict()
        self.apus = nn.ModuleDict()
        self.chains = nn.ModuleDict()
        self.chain_apus = nn.ModuleDict()  # chain APUs: exact float64 forward, tables frozen (not in `apus`)
        self._g26layer = set()  # layers moved to graph26 at run time (wide lossy elements)
        self._g26conv = set()  # layers QATNet cannot train (a16 conv+add): exact float64 graph26 forward, frozen
        for n in self.order:
            L = layers[n]
            t, prm = L["type"], L.get("params", {})
            key = n.replace("/", "__")
            if t not in ("conv", "dw", "normalization", "batch_norm", "dense") and G26.use_chain(n, L, P):
                self.chains[key] = QChain(G26.build_chain(n, L, P, layers), self._apu_factory(key, dev), dev)
                continue
            if t in ("conv", "dw", "normalization", "batch_norm", "dense"):
                ca = None
                if prm.get("elementwise_add"):
                    ew = P.op_by_type(n, "ElementwiseAddOp")
                    ca = (_t(ew["elwa_factor"], dev), float(ew["pre_acc_shift"]), float(ew["elwa_feed_repeat"]))
                wide = int(P.conv[n]["acc_bits"]) > 16
                try:
                    self.mac[key] = (WideConv if wide else MACConv)(P.conv[n], dev, ca)
                except AssertionError as e:  # e.g. a 16-bit bias outside the trainable lattice: exact frozen chain
                    if ca is not None:  # a16_w16 conv + fused add: exact frozen graph26 conv_add_layer (no training)
                        print(f"[qatfast] {n}: {str(e)[:60]} -> frozen exact graph26 conv-add", flush=True)
                        self._g26conv.add(key)
                        continue
                    print(f"[qatfast] {n}: {str(e)[:80]} -> frozen-weight atomic chain", flush=True)
                    self.chains[key] = QChain(G26.build_chain(n, L, P, layers), self._apu_factory(key, dev), dev)
            elif t in ("ew_add", "ew_sub", "activation", "matmul", "reduce_sum", "ew_mult"):
                rec = P.conv[n] if t in ("ew_add", "ew_sub") else P.op_by_type(n, "ActivationOp")
                self.apus[key] = APU(rec, dev)

    def _apu_factory(self, key, dev):
        def make(opname, rec):
            k = key + "__" + opname.rsplit("/", 1)[-1]
            self.chain_apus[k] = APU(rec, dev)
            return k
        return make

    @staticmethod
    def _order(layers):
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
        return order

    def _c(self, rec, name):
        """Device float32 copy of a capture constant, made once: per-forward npz reads + H2D copies synced the stream."""
        key = (id(rec), name)
        hit = self._cache.get(key)
        if hit is None or hit[0] is not rec:
            hit = (rec, torch.as_tensor(np.asarray(rec[name], np.float32), device=self.dev))
            self._cache[key] = hit
        return hit[1]

    def _bias(self, rec, x):
        b = self._c(rec, "bias")
        b = b.reshape(1, -1, 1, 1) if b.numel() > 1 else b
        return lossy_ste(x + b, lossy_of(rec))

    def _mock(self, rec, x):
        kq = self._c(rec, "kernel_q")
        kq = kq.reshape(1, -1, 1, 1) if kq.numel() > 1 else kq
        s = int(rec["pre_acc_shift"])
        return RsMulFn.apply(x, kq.reshape(-1), s, lossy_of(rec))

    def _g26_step(self, n, layers, out):
        """One layer on the exact float64 graph26 path (dedicated handler, else its captured atomic-op chain)."""
        import graph26 as G26
        L = layers[n]
        t, prm = L["type"], L.get("params", {})
        ins = [x.double() for x in I.layer_inputs(out, layers, n)]
        with torch.no_grad():
            try:
                y = G26._dispatch(n, L, t, prm, ins, self.P, layers, None, self.dev, None)
            except Exception:  # noqa: BLE001
                y = G26._chain(n, L, ins, self.P, layers, self.dev)
        return y.float() if torch.is_tensor(y) else [v.float() for v in y]

    @torch.no_grad()
    def _refresh(self):
        """One host sync per forward: per-layer single-conv bound from the current integer kernels, and the exactness
        flags raised by the previous forward (a flagged layer goes back to the split path for good)."""
        mods = list(self.mac.values())
        if not mods:  # no 8-bit MAC layers (all-16-bit nets run every conv on the exact frozen path)
            return
        if self._flags is None:
            self._flags = torch.zeros(len(mods), dtype=torch.int32, device=self.dev)
            for i, m in enumerate(mods):
                m.flag = self._flags[i:i + 1]
        sums = torch.stack([m.int_kernel().abs().sum((1, 2, 3)).max() for m in mods])
        host = torch.cat([sums, self._flags.float()]).cpu()
        sums, flags = host[:len(mods)], host[len(mods):]
        for i, (k, m) in enumerate(self.mac.items()):
            if flags[i]:
                m.unsafe = True
                m.planes = 4  # covers an input range the capture did not show
                self.flag_events.append((k, int(flags[i])))
                print(f"[qatfast] exactness flag {int(flags[i])} on {k}: layer moved to the split path", flush=True)
            m.single = (not m.unsafe) and m.planes == 2 and m.s > 0 and 255.0 * float(sums[i]) + m.res_bound < 2 ** 24
        if flags.any():
            self._flags.zero_()

    def forward(self, img_u8):
        """img_u8: (N, H, W, 3) uint8-valued float tensor. Returns {layer: int-valued tensor}."""
        self._refresh()
        layers, P = self.hn["layers"], self.P
        out = {}
        free = not torch.is_grad_enabled() and os.environ.get("G26_KEEP_ALL") != "1"  # inference: drop after last use
        if free and not hasattr(self, "_last"):
            self._keep = {i for v in layers.values() if v["type"] == "output_layer" for i in v["input"]}
            self._keep |= {i for v in layers.values() if v["type"] == "postprocess" for i in v["input"]}  # chip outputs of a host NMS
            self._last = {}
            for k_, m_ in enumerate(self.order):
                for i_ in layers[m_]["input"]:
                    self._last[i_] = k_
        prev = None
        for k_step, n in enumerate(self.order):
            if free and prev is not None:
                for i_ in layers[prev]["input"]:
                    if self._last.get(i_) == k_step - 1 and i_ not in self._keep:
                        out.pop(i_, None)
            prev = n
            if n in self._g26layer:  # layers the 24-bit float32 datapath cannot hold: exact float64 graph26, frozen
                out[n] = self._g26_step(n, layers, out)
                continue
            try:
                L = layers[n]
                t, prm = L["type"], L.get("params", {})
                if t in ("output_layer", "postprocess"):  # postprocess (NMS) runs on the host, not the chip
                    continue
                key = n.replace("/", "__")
                ins = I.layer_inputs(out, layers, n)
                if key in self.chains:
                    y = self.chains[key]([x.to(torch.float64) for x in ins], self.chain_apus)
                    if t == "const_input" and torch.is_tensor(y) and y.shape[0] == 1 and img_u8.shape[0] > 1:
                        y = y.expand(img_u8.shape[0], *y.shape[1:])  # per-frame constant: broadcast to the batch
                    out[n] = y
                    continue
                if t == "input_layer":
                    y = img_u8.permute(0, 3, 1, 2).float()
                    po = [v for k, v in P.ops.items() if k.rsplit("/", 1)[0] == n]
                    sc = float(np.asarray(po[-1].get("output_scale", 1.0)).ravel()[0]) if po else 1.0
                    zp = float(np.asarray(po[-1].get("output_zero_point", 0.0)).ravel()[0]) if po else 0.0
                    if abs(sc - 1.0) > 1e-9 or zp != 0.0:  # host-side input quantization round(x / scale + zp)
                        bits = int(np.asarray(po[-1].get("outq_bits", 8)).ravel()[0]) if po else 8  # UINT16 inputs use 0..65535
                        y = torch.round(I.f32_quantize(y, sc, zp)).clamp(0, 2 ** bits - 1).float()  # IEEE float32 = HSim
                elif t == "dense":
                    fl = [v for k, v in P.ops.items() if k.rsplit("/", 1)[0] == n and v["type"] == "FlattenOp"]
                    x = A.gather(fl[0], ins[0].double()).float() if fl and "perm" in fl[0] else ins[0]
                    y = self.mac[key](x if x.dim() == 4 else x[:, :, None, None])
                elif key in self._g26conv:
                    r = P.conv[n]; p = {**to_t(r, self.dev), **conv_params(r, self.dev)}
                    ew = P.op_by_type(n, "ElementwiseAddOp")
                    p.update(ca_factor=_t(ew["elwa_factor"], self.dev), ca_shift=torch.tensor(float(ew["pre_acc_shift"])),
                             ca_feed=torch.tensor(float(ew["elwa_feed_repeat"])))
                    with torch.no_grad():
                        y = I.conv_add_layer(ins[0].double(), ins[1].double(), p)[3].float()
                elif t in ("conv", "dw", "normalization", "batch_norm"):
                    y = self.mac[key](ins[0], ins[1] if prm.get("elementwise_add") else None)
                elif t in ("ew_add", "ew_sub"):
                    r = P.conv[n]
                    k = self._c(r, "ew_k")
                    s = int(r["ew_shift"])
                    x0, x1 = ins
                    if "ew_repeats" in r.files:
                        x0, x1 = I.apply_repeats(x0, r["ew_repeats"][0].tolist()), I.apply_repeats(x1, r["ew_repeats"][1].tolist())
                    cfg = (int(r["ewacc_bits"]), bool(r["ewacc_signed"]), bool(r["ewacc_wrap"]))
                    if cfg[0] > 24:  # 32-bit accumulators (a16 layers): exact float64 path, linear surrogate gradient
                        p = to_t(r, self.dev)
                        with torch.no_grad():
                            _, _, ye = I.ew_add_layer(ins[0].double(), ins[1].double(), p)
                        m = np.asarray(r["mant"], np.float64) * 2.0 ** -np.asarray(r["shift"], np.float64)
                        g = float(np.abs(m).mean()) * 2.0 ** -float(r["final_shift"]) if m.size else 1.0
                        sur = (x0 * k[0][None, :, None, None] + x1 * k[1][None, :, None, None]) * (2.0 ** -s * g)
                        y = ste(ye.float(), sur)
                    else:
                        acc = RsMulFn.apply(x0, k[0], s, cfg) + RsMulFn.apply(x1, k[1], s, cfg)
                        y = self.apus[key].fused(acc, self._c(r, "bias"),
                                                 (int(r["bias_bits"]), bool(r["bias_signed"]), bool(r["bias_wrap"])))
                elif t == "argmax":  # channel index of the max (HSim: first index on ties; reverse_order = last)
                    x_ = ins[0].flip(1) if prm.get("reverse_order") else ins[0]
                    y = torch.argmax(x_, dim=1, keepdim=True)
                    y = (x_.shape[1] - 1 - y if prm.get("reverse_order") else y).to(ins[0].dtype)
                elif t == "feature_splitter":
                    y = I.feature_split(ins[0], L)
                elif t == "shortcut":
                    y = ins[0]
                elif t == "concat":
                    y = (torch.cat(ins, dim={'spatial_h': 2, 'spatial_w': 3}[prm['concat_axis']])
                         if prm.get('concat_axis') in ('spatial_h', 'spatial_w')  # HN concat_axis; default = features
                         else I.concat_features(ins, I.concat_groups(layers, n)))
                elif t == "maxpool":
                    kh, kw = prm["kernel_shape"][1:3]
                    ph, pw = kh - 1, kw - 1
                    y = F.max_pool2d(F.pad(ins[0], (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=0.0),
                                     (kh, kw), tuple(prm["strides"][1:3]))
                elif t == "resize":
                    y = ins[0].repeat_interleave(int(prm["resize_h_ratio_list"][0]), 2)
                    y = y.repeat_interleave(int(prm["resize_w_ratio_list"][0]), 3)
                elif t == "format_conversion":
                    oh, ow = L["output_shapes"][0][1:3]
                    y = ins[0].reshape(ins[0].shape[0], ins[0].shape[1], oh, ow)
                elif t == "activation" and int(P.op_by_type(n, "MockConvOp")["outq_bits"]) > 24:
                    y = self._wide_activation(n, ins[0])  # precision_change between a16_w16 and int8 layers
                elif t == "activation":
                    br = P.op_by_type(n, "AddBiasOp")
                    y = self.apus[key].fused(self._mock(P.op_by_type(n, "MockConvOp"), ins[0]),
                                             self._c(br, "bias").reshape(-1), lossy_of(br))
                elif t == "matmul":
                    m = P.op_by_type(n, "MatmulOp")
                    y = self.apus[key].fused(self._matmul(m, ins[0], ins[1]))
                elif t == "reduce_max":
                    g = int(prm["groups"])
                    x = ins[0]
                    y = x.reshape(x.shape[0], g, x.shape[1] // g, x.shape[2], x.shape[3]).amax(2)
                elif t == "reduce_sum":
                    rs = P.op_by_type(n, "ReduceSumOp")
                    br = P.op_by_type(n, "AddBiasOp")
                    y = self.apus[key].fused(self._reduce_sum(rs, ins[0], int(prm["groups"])),
                                             self._c(br, "bias").reshape(-1), lossy_of(br))
                elif t == "ew_mult":
                    mks, bss = P.ops_by_type(n, "MockConvOp"), P.ops_by_type(n, "AddBiasOp")
                    mu = P.op_by_type(n, "ElementwiseMultOp")
                    a = self._bias(bss[0], self._mock(mks[0], ins[0]))
                    b = self._bias(bss[1], self._mock(mks[1], ins[1]))
                    rep = prm.get("input_repeats", [[1, 1, 1], [1, 1, 1]])
                    a, b = I.apply_repeats(a, rep[0]), I.apply_repeats(b, rep[1])
                    ms = int(mu["_multiplier_shift"])
                    sur = a * b / 2.0 ** ms
                    y = self.apus[key].fused(lossy_ste(ste(torch.round(sur.detach()), sur), lossy_of(mu)))
                else:
                    raise NotImplementedError(f"{n}: {t}")
                out[n] = y
            except RuntimeError as e_:
                if "datapath supports" not in str(e_):
                    raise
                self._g26layer.add(n)
                print(f"[qatfast] {n}: {str(e_)[:80]} -> exact float64 graph26 for this layer (frozen)", flush=True)
                out[n] = self._g26_step(n, layers, out)
        # a feature_splitter's slices are only intermediate: every caller gets one tensor per layer
        return {k: v for k, v in out.items() if not isinstance(v, list)}

    def _wide_activation(self, n, x):
        """Standalone activation with a > 24-bit mock / bias (precision_change): hq.wide_tail mock mode, no trainable
        state; gradient = kq / 2^s * APU slope."""
        P, dev = self.P, self.dev
        mk, br, ar = P.op_by_type(n, "MockConvOp"), P.op_by_type(n, "AddBiasOp"), P.op_by_type(n, "ActivationOp")
        key = ("wide_act", n)
        if key not in self._cache:
            c = x.shape[1]
            bias = torch.as_tensor(np.broadcast_to(np.asarray(br["bias"], np.float32).reshape(-1), (c,)).copy(), device=dev)
            self._cache[key] = (bias, wide_apu_tables(ar, dev), torch.zeros(1, dtype=torch.int32, device=dev))
        bias, tabs, flag = self._cache[key]
        kq = self._c(mk, "kernel_q").reshape(-1).contiguous()
        s = int(mk["pre_acc_shift"])
        y, gs = hq.wide_tail(x.detach().contiguous(), _empty(dev), kq, 1, 1, 0, s, list(lossy_of(mk)), bias, 1.0,
                             1.0, list(lossy_of(br)), *tabs, flag)
        kq4 = kq.reshape(1, -1, 1, 1) if kq.numel() > 1 else kq
        return ste(y, x * kq4 / 2.0 ** s * gs)

    def _reduce_sum(self, rs, x, g):
        kq = self._c(rs, "kernel_q")
        kq = kq.reshape(1, -1, 1, 1) if kq.numel() > 1 else kq
        s = int(rs["pre_acc_shift"])
        cfg = lossy_of(rs)
        sur = x * kq * 2.0 ** -s
        v = lossy_ste(ste(torch.round(sur.detach()), sur), cfg)
        return lossy_ste(v.reshape(v.shape[0], g, v.shape[1] // g, v.shape[2], v.shape[3]).sum(2), cfg)

    def _matmul(self, m, x0, x1):
        """Windowed matmul (acceleras _input_windows (wh, ww)): windows folded into the batch, unfolded after."""
        wh, ww = (int(v) for v in np.asarray(m.get("_input_windows", [1, 1, 1])).ravel()[:2])
        if wh == 1 and ww == 1:
            return self._matmul_core(m, x0, x1)

        def fold(x):
            n, c, h, w = x.shape
            return x.reshape(n, c, wh, h // wh, ww, w // ww).permute(0, 2, 4, 1, 3, 5).reshape(n * wh * ww, c, h // wh, w // ww)
        n, _, h, w = x0.shape
        y = self._matmul_core(m, fold(x0), fold(x1))
        c = y.shape[1]
        return y.reshape(n, wh, ww, c, h // wh, w // ww).permute(0, 3, 1, 4, 2, 5).reshape(n, c, h, w)

    def _matmul_core(self, m, x0, x1):
        groups, transpose = int(m["_groups"]), bool(m["_transpose_matmul_input"])
        zp_added, rank = bool(m["_zp_comp_added"]), int(m.get("zp_comp_rank", 1))
        s, cfg = int(m["_multiplier_shift"]), lossy_of(m)
        zp0 = float(np.asarray(m["input_zero_points"][0]).ravel()[0])
        fr = np.asarray(m.get("feed_repeat", 1), np.float64).reshape(-1)
        n, c0, h, w = x0.shape
        tt = h * w
        d = x0.permute(0, 2, 3, 1).reshape(n, tt, groups, c0 // groups).permute(0, 2, 1, 3)
        c1 = x1.shape[1]
        wt = x1.permute(0, 2, 3, 1).reshape(n, x1.shape[2] * x1.shape[3], groups, c1 // groups).permute(0, 2, 1, 3)
        zpc = None
        if zp_added:
            zpc, wt = wt[..., -rank:], wt[..., :-rank]
        if transpose:
            wt = wt.transpose(-1, -2)
        sur = torch.matmul(d, wt) / 2.0 ** s
        with torch.no_grad():
            acc = exact_matmul(d.detach(), wt.detach(), s)
        acc = ste(acc, sur)
        if zpc is not None:
            for r in range(rank):
                tsur = zp0 * zpc[..., r] / 2.0 ** s
                term = lossy_ste(ste(torch.round(tsur.detach()), tsur), cfg) * float(fr[r if fr.size > 1 else 0])
                acc = acc + term[:, :, None, :]
        acc = lossy_ste(acc, cfg)
        j = acc.shape[-1]
        return acc.permute(0, 2, 1, 3).reshape(n, tt, groups * j).permute(0, 2, 1).reshape(n, groups * j, h, w)

    def enable_apu_training(self, m_radius=1e9, o_radius=1e9):
        for m in self.mac.values():
            m.apu.enable_training(m_radius, o_radius)
        for a in self.apus.values():
            a.enable_training(m_radius, o_radius)

    def apu_modules(self):
        """{key: APU} for every APU (MAC layers by their mac key, the rest by their apus key)."""
        out = {k: m.apu for k, m in self.mac.items()}
        out.update({"apu:" + k: a for k, a in self.apus.items()})
        return out

    @torch.no_grad()
    def export_apu(self):
        """{key: (mant[P], offs[P])} integer tables of every trainable APU."""
        return {k: tuple(t[0].cpu() for t in a.tables()) for k, a in self.apu_modules().items() if a.trainable}

    @torch.no_grad()
    def load_apu(self, tabs):
        for k, (m, o) in tabs.items():
            a = self.apu_modules()[k]
            c = a.mant_i.shape[0]
            a.mant_i.copy_(m.to(a.mant_i.device).int()[None].expand(c, -1))
            a.offs_i.copy_(o.to(a.offs_i.device).int()[None].expand(c, -1))
            if a.trainable:
                a.mant_lat.copy_(m.float())
                a.offs_lat.copy_(o.float())

    def overflow_penalty(self):
        return sum(m.overflow for m in self.mac.values())

    @torch.no_grad()
    def export_ints(self):
        """{layer: (int kernel OIHW, int bias)} for graph26 evaluation and HAR write-back."""
        return {k.replace("__", "/"): (m.int_kernel().detach().round(), m.int_bias().detach().round())
                for k, m in self.mac.items()}

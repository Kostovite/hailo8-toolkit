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

import intops as I
from graph import conv_params, to_t
from graph26 import _t, lossy_of


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
    lo, hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)
    inside = ((x.detach() >= lo) & (x.detach() <= hi)).to(x.dtype)
    return ste(y, x * inside)


class APU(nn.Module):
    """Exact APU forward; gradient = mant / 2^(shift + final_shift) of the hit segment, 0 where the output saturates."""

    def __init__(self, rec, dev):
        super().__init__()
        for k in ("thr", "mant", "shift", "offs"):
            self.register_buffer(k, _t(rec[k], dev))
        self.final_shift = int(rec["final_shift"])
        self.mant_signed = bool(rec["mant_signed"])
        self.clip1, self.clip2, self.out = lossy_of(rec, "clip1"), lossy_of(rec, "clip2"), lossy_of(rec, "out")

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
    resid = torch.round(F.conv2d(xa, gk, stride=stride, dilation=dil, groups=groups))
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

    def int_kernel(self):
        return ste_round(self.kernel.clamp(self.kmin, self.kmax))

    def int_bias(self):
        return ste_round(self.bias.clamp(-32768.0, 32767.0))

    def forward(self, x, res=None):
        k = self.int_kernel()
        xp = F.pad(x, (self.pad[2], self.pad[3], self.pad[0], self.pad[1]), value=self.pad_value)
        z = F.conv2d(xp, k, stride=self.stride, dilation=self.dil, groups=self.groups) / 2.0 ** self.s
        with torch.no_grad():  # exact per-product rounding: the sum-then-round shortcut drifts 6+ LSB by layer 3
            total = I.nib_conv(xp.detach(), k.detach(), stride=self.stride, dilation=self.dil, groups=self.groups).float()
            acc_ex = exact_mac(xp.detach(), k.detach(), total, self.s, self.stride, self.dil, self.groups)
        acc = ste(acc_ex, z)
        if res is not None:
            f = self.ca_factor[None, :, None, None]
            s, feed = float(self.conv_add[1]), float(self.conv_add[2])
            term_sur = res * f / 2.0 ** s
            acc = acc + ste(torch.round(term_sur.detach()), term_sur) * feed
        self.overflow = F.relu(acc.abs() - 2 ** (self.acc[0] - 1)).mean()
        acc = lossy_ste(acc, self.acc)
        b = lossy_ste(acc + self.int_bias()[None, :, None, None], self.bias_cfg)
        return self.apu(b)


class QATNet(nn.Module):
    """Builds trainable MAC layers from graph26.Params; non-MAC ops reuse the exact integer kernels with STE."""

    def __init__(self, hn, P, dev):
        super().__init__()
        self.hn, self.P, self.dev = hn, P, dev
        layers = hn["layers"]
        self.order = self._order(layers)
        self.mac = nn.ModuleDict()
        self.apus = nn.ModuleDict()
        for n in self.order:
            L = layers[n]
            t, prm = L["type"], L.get("params", {})
            key = n.replace("/", "__")
            if t in ("conv", "dw", "normalization"):
                ca = None
                if prm.get("elementwise_add"):
                    ew = P.op_by_type(n, "ElementwiseAddOp")
                    ca = (_t(ew["elwa_factor"], dev), float(ew["pre_acc_shift"]), float(ew["elwa_feed_repeat"]))
                self.mac[key] = MACConv(P.conv[n], dev, ca)
            elif t in ("ew_add", "ew_sub", "activation", "matmul", "reduce_sum", "ew_mult"):
                rec = P.conv[n] if t in ("ew_add", "ew_sub") else P.op_by_type(n, "ActivationOp")
                self.apus[key] = APU(rec, dev)

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

    def _bias(self, rec, x):
        b = _t(rec["bias"], self.dev).float()
        b = b.reshape(1, -1, 1, 1) if b.numel() > 1 else b
        return lossy_ste(x + b, lossy_of(rec))

    def _mock(self, rec, x):
        kq = _t(rec["kernel_q"], self.dev).float()
        kq = kq.reshape(1, -1, 1, 1) if kq.numel() > 1 else kq
        s = int(rec["pre_acc_shift"])
        with torch.no_grad():
            ex = I.apu_round_shift(x.detach().long() * kq.long(), torch.full(x.shape, s, dtype=torch.long,
                                                                              device=self.dev), 3, True).float()
        return lossy_ste(ste(ex, x * kq / 2.0 ** s), lossy_of(rec))

    def forward(self, img_u8):
        """img_u8: (N, H, W, 3) uint8-valued float tensor. Returns {layer: int-valued tensor}."""
        layers, P = self.hn["layers"], self.P
        out = {}
        for n in self.order:
            L = layers[n]
            t, prm = L["type"], L.get("params", {})
            if t == "output_layer":
                continue
            key = n.replace("/", "__")
            ins = [out[i] for i in L["input"]]
            if t == "input_layer":
                y = img_u8.permute(0, 3, 1, 2).float()
            elif t in ("conv", "dw", "normalization"):
                y = self.mac[key](ins[0], ins[1] if prm.get("elementwise_add") else None)
            elif t in ("ew_add", "ew_sub"):
                r = P.conv[n]
                k = _t(r["ew_k"], self.dev).float()
                s = int(r["ew_shift"])
                x0, x1 = ins
                if "ew_repeats" in r.files:
                    x0, x1 = I.apply_repeats(x0, r["ew_repeats"][0].tolist()), I.apply_repeats(x1, r["ew_repeats"][1].tolist())
                cfg = (int(r["ewacc_bits"]), bool(r["ewacc_signed"]), bool(r["ewacc_wrap"]))
                terms = []
                for x, kk in ((x0, k[0]), (x1, k[1])):
                    sur = x * kk[None, :, None, None] / 2.0 ** s
                    with torch.no_grad():
                        ex = I.apu_round_shift(x.detach().long() * kk.long()[None, :, None, None],
                                               torch.full(x.shape, s, dtype=torch.long, device=self.dev), 3, True).float()
                    terms.append(lossy_ste(ste(ex, sur), cfg))
                acc = terms[0] + terms[1]
                b = lossy_ste(acc + _t(r["bias"], self.dev).float()[None, :, None, None],
                              (int(r["bias_bits"]), bool(r["bias_signed"]), bool(r["bias_wrap"])))
                y = self.apus[key](b)
            elif t == "concat":
                y = torch.cat(ins, dim=1)
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
            elif t == "activation":
                y = self.apus[key](self._bias(P.op_by_type(n, "AddBiasOp"), self._mock(P.op_by_type(n, "MockConvOp"), ins[0])))
            elif t == "matmul":
                m = P.op_by_type(n, "MatmulOp")
                y = self.apus[key](self._matmul(m, ins[0], ins[1]))
            elif t == "reduce_max":
                g = int(prm["groups"])
                x = ins[0]
                y = x.reshape(x.shape[0], g, x.shape[1] // g, x.shape[2], x.shape[3]).amax(2)
            elif t == "reduce_sum":
                rs = P.op_by_type(n, "ReduceSumOp")
                y = self.apus[key](self._bias(P.op_by_type(n, "AddBiasOp"), self._reduce_sum(rs, ins[0], int(prm["groups"]))))
            elif t == "ew_mult":
                mks, bss = P.ops_by_type(n, "MockConvOp"), P.ops_by_type(n, "AddBiasOp")
                mu = P.op_by_type(n, "ElementwiseMultOp")
                a = self._bias(bss[0], self._mock(mks[0], ins[0]))
                b = self._bias(bss[1], self._mock(mks[1], ins[1]))
                rep = prm.get("input_repeats", [[1, 1, 1], [1, 1, 1]])
                a, b = I.apply_repeats(a, rep[0]), I.apply_repeats(b, rep[1])
                ms = int(mu["_multiplier_shift"])
                sur = a * b / 2.0 ** ms
                y = self.apus[key](lossy_ste(ste(torch.round(sur.detach()), sur), lossy_of(mu)))
            else:
                raise NotImplementedError(f"{n}: {t}")
            out[n] = y
        return out

    def _reduce_sum(self, rs, x, g):
        kq = _t(rs["kernel_q"], self.dev).float()
        kq = kq.reshape(1, -1, 1, 1) if kq.numel() > 1 else kq
        s = int(rs["pre_acc_shift"])
        cfg = lossy_of(rs)
        sur = x * kq * 2.0 ** -s
        v = lossy_ste(ste(torch.round(sur.detach()), sur), cfg)
        return lossy_ste(v.reshape(v.shape[0], g, v.shape[1] // g, v.shape[2], v.shape[3]).sum(2), cfg)

    def _matmul(self, m, x0, x1):
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

    def overflow_penalty(self):
        return sum(m.overflow for m in self.mac.values())

    @torch.no_grad()
    def export_ints(self):
        """{layer: (int kernel OIHW, int bias)} for graph26 evaluation and HAR write-back."""
        return {k.replace("__", "/"): (m.int_kernel().detach().round(), m.int_bias().detach().round())
                for k, m in self.mac.items()}

"""Host-side op vs on-chip approximation: what each costs in throughput and accuracy.

Model: 224x224x3 -> conv s2 32 -> conv s2 64 -> [OP on 56x56x64] -> conv 64 -> conv s2 128 -> 1x1 conv 32.
  chip : one HEF with OP rewritten on-chip (onnx_hailo_compat + pwl_inject)
  host : HEF part1 (to OP input) -> OP in onnxruntime on the CPU -> HEF part2 (from OP output)
Accuracy: SQNR of the final output vs float, bit-exact emulation (chip) / emulated parts + float host op (host).

PREC=max16 builds the 16-bit variants (chip_16.hef, part1_16.hef = 16-bit boundary output, part2_16.hef = 16-bit
input) and writes result_16.json.

Usage (general env, export):  python host_vs_chip.py export OP OUT_DIR
       (DFC env, build + score): python host_vs_chip.py build OUT_DIR
"""
import json
import math
import os
import sys

import numpy as np

OPS = ["Sin", "Erf", "Floor", "Atan", "TopK", "Hardmax", "ReduceProd", "GridSample", "Bicubic", "SiLU", "Identity"]


def export(op, out):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(0)
    os.makedirs(out, exist_ok=True)

    class Op(nn.Module):
        def forward(self, x):
            if op == "Sin":
                return torch.sin(x)
            if op == "Erf":
                return torch.erf(x)
            if op == "Floor":
                return torch.floor(x)
            if op == "Atan":
                return torch.atan(x)
            if op == "SiLU":
                return F.silu(x)
            if op == "Identity":
                return x
            if op == "TopK":  # top-4 channels per pixel, zero elsewhere (a sparsifying gate)
                v, i = x.topk(4, dim=1)
                return torch.zeros_like(x).scatter(1, i, v)
            if op == "Hardmax":
                return F.one_hot(x.argmax(1), x.shape[1]).permute(0, 3, 1, 2).float() * x
            if op == "ReduceProd":
                return x * torch.prod(torch.sigmoid(x[:, :8]), dim=1, keepdim=True)
            if op == "GridSample":
                g = torch.stack(torch.meshgrid(torch.linspace(-1, 1, x.shape[2]), torch.linspace(-1, 1, x.shape[3]),
                                               indexing="ij"), -1)[None].flip(-1) * 0.9
                return F.grid_sample(x, g, align_corners=False)
            if op == "Bicubic":
                return F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode="bicubic")
            raise ValueError(op)

    def blk(ci, co, s):
        return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1), nn.BatchNorm2d(co), nn.ReLU())

    part1 = nn.Sequential(blk(3, 32, 2), blk(32, 64, 2)).eval()
    part2 = nn.Sequential(blk(64, 64, 1), blk(64, 128, 2), nn.Conv2d(128, 32, 1)).eval()
    for m in list(part1.modules()) + list(part2.modules()):
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-0.2, 0.2)
            m.running_var.uniform_(0.5, 2)
    full = nn.Sequential(part1, Op(), part2).eval()
    x = torch.rand(1, 3, 224, 224) * 255
    kw = dict(opset_version=17, dynamo=False)
    norm = lambda m: nn.Sequential(Norm(), m)  # noqa: E731

    class Norm(nn.Module):
        def forward(self, x):
            return x / 255.0

    torch.onnx.export(norm(full), x, f"{out}/full.onnx", input_names=["x"], output_names=["y"], **kw)
    torch.onnx.export(norm(part1), x, f"{out}/part1.onnx", input_names=["x"], output_names=["f"], **kw)
    f = part1(x / 255.0)
    torch.onnx.export(Op().eval(), f, f"{out}/op.onnx", input_names=["f"], output_names=["g"], **kw)
    torch.onnx.export(part2, Op()(f), f"{out}/part2.onnx", input_names=["g"], output_names=["y"], **kw)
    json.dump({"op": op}, open(f"{out}/meta.json", "w"))
    print("exported", op, "->", out)


def sq(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(10 * np.log10((a ** 2).sum() / max(((a - b) ** 2).sum(), 1e-30)))


def build(out):
    """DFC env: compile chip variant (compat + pwl) and the two host parts; emulate; write result.json."""
    import onnxruntime as ort
    from hailo_sdk_client import ClientRunner, InferenceContext
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import onnx_hailo_compat as C
    import pwl_inject as PW
    op = json.load(open(f"{out}/meta.json"))["op"]
    rng = np.random.default_rng(1)
    calib = (rng.random((64, 3, 224, 224)) * 255).astype(np.float32)
    test = (np.random.default_rng(2).random((8, 3, 224, 224)) * 255).astype(np.float32)
    nhwc = lambda a: np.transpose(a, (0, 2, 3, 1))  # noqa: E731
    ref = np.concatenate([ort.InferenceSession(f"{out}/full.onnx").run(None, {"x": t[None]})[0] for t in test])
    res = {"op": op}
    script = "model_optimization_flavor(optimization_level=0, compression_level=0)\n"
    prec = os.environ.get("PREC", "a8_w8")  # a8_w8, or max16 = 16-bit wherever the layer supports it (MILP)
    sfx = os.environ.get("SFX", "" if prec == "a8_w8" else "_16")
    res["prec"] = prec

    def dfc(path, cal, name, fit_json=None):
        r = ClientRunner(hw_arch="hailo8")
        r.translate_onnx_model(path, name)
        if fit_json:
            PW.apply_pwl(r, fit_json)
        alls = script
        if prec != "a8_w8":
            import probe
            by = {}
            for n_, m_ in probe.supported_layers(path, name, prec, nhwc(cal))[0]:
                by.setdefault(m_, []).append(n_)
            for m_, ns in by.items():
                alls += f"quantization_param([{', '.join(ns)}], precision_mode={m_})\n"
            res[f"{name}_modes"] = {m_: len(ns) for m_, ns in by.items()}
        r.load_model_script(alls)
        r.optimize(nhwc(cal))
        hef = r.compile()
        open(f"{out}/{name}{sfx}.hef", "wb").write(hef)
        return r

    def emu(r, data):
        with r.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
            return np.asarray(r.infer(ctx, nhwc(data)))

    # chip variant
    try:
        C.run(f"{out}/full.onnx", f"{out}/full_c.onnx", C.ALL, check=False)
        fit = None
        import onnx
        unary = {n.op_type for n in onnx.load(f"{out}/full_c.onnx").graph.node} & set(PW.FUNCS)
        unary -= {"Exp", "Log", "Reciprocal", "Sqrt"}  # native on the chip
        if os.environ.get("PWL_SILU") == "1":  # replace Hailo's built-in SiLU by our fit
            unary.add("SiLU")
        src = f"{out}/full_c.onnx"
        if unary:
            np.save(f"{out}/calib{sfx}.npy", calib[:16])
            PW.fit(src, f"{out}/calib{sfx}.npy", f"{out}/fit{sfx}.json", unary, int(os.environ.get("PIECES", "9")))
            PW.mark(src, f"{out}/fit{sfx}.json", f"{out}/full_p{sfx}.onnx")
            src, fit = f"{out}/full_p{sfx}.onnx", f"{out}/fit{sfx}.json"
        r = dfc(src, calib, "chip", fit)
        res["chip_sqnr_db"] = sq(nhwc(ref), emu(r, test).reshape(nhwc(ref).shape))
    except Exception as e:  # noqa: BLE001
        res["chip_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    # host variant
    try:
        if os.environ.get("SKIP_HOST") == "1":
            raise RuntimeError("skipped (SKIP_HOST=1)")
        s1, so = ort.InferenceSession(f"{out}/part1.onnx"), ort.InferenceSession(f"{out}/op.onnx")
        f_cal = np.concatenate([s1.run(None, {"x": t[None]})[0] for t in calib])
        g_cal = np.concatenate([so.run(None, {"f": t[None]})[0] for t in f_cal])
        r1 = dfc(f"{out}/part1.onnx", calib, "part1")
        r2 = dfc(f"{out}/part2.onnx", g_cal, "part2")
        f_q = np.transpose(emu(r1, test), (0, 3, 1, 2)).astype(np.float32)  # part1 on chip (dequantized)
        g_h = np.concatenate([so.run(None, {"f": t[None]})[0] for t in f_q])  # op on the host, float
        y = emu(r2, g_h)
        res["host_sqnr_db"] = sq(nhwc(ref), y.reshape(nhwc(ref).shape))
        res["boundary_MB_per_frame"] = round((f_q[0].size + g_h[0].size) / 1e6, 3)  # uint8 out + uint8 in
    except Exception as e:  # noqa: BLE001
        res["host_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    json.dump(res, open(f"{out}/result{sfx}.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3])
    else:
        build(sys.argv[2])

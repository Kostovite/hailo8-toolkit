"""Build one probe with the stock DFC flow and compare chip, SDK_BIT_EXACT emulator and float on the same frames.

Usage (Hailo DFC 3.33.1 + HailoRT 4.23 env, a Hailo-8 attached):
  python run.py DIR [--alls "line; line"] [--a16] [--fix-dynmm] [--smx-headroom full] [--amps 0.5,1,2]
                    [--calib 64] [--frames 8] [--device 0000:03:00.0] [--u16-boundary]

DIR holds m.onnx (from models.py). Steps:
  1. translate_onnx_model(hw_arch=hailo8)
  2. optimize: model_optimization_flavor(optimization_level=0, compression_level=0) + --alls,
     calibration = N(0, 1) frames (seed 1)
  3. compile (single call, stock allocator) -> DIR/m.hef, DIR/q.har, DIR/auto.alls
  4. test frames: N(0, 1) * amp (seed 0), quantized to the HEF's raw input format
  5. the same raw frames -> chip (batch, and frame by frame), SDK_BIT_EXACT emulator, onnxruntime float
  6. report per amplitude: % output elements chip != emulator, max |diff| in output LSB, SNR vs float (dB)
"""
import argparse
import json
import os
import re
import sys
import tarfile

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--alls", default="", help="extra model-script lines: a .alls file, or text with ';' between lines")
ap.add_argument("--a16", action="store_true", help="every layer a16_w16, input layer included (UINT16 input)")
ap.add_argument("--fix-dynmm", action="store_true", help="workaround #1: force_range_out on dynamic-matmul operands")
ap.add_argument("--smx-headroom", default="", help="workaround #6: size softmax denominators for the worst case")
ap.add_argument("--amps", default="1.0")
ap.add_argument("--calib", type=int, default=64)
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--device", default=os.environ.get("DEVICE", ""))
ap.add_argument("--u16-boundary", action="store_true", help="#11: one raw input element swept over 30000-40000")
ap.add_argument("--no-compile", action="store_true", help="reuse DIR/m.hef + DIR/q.har")
ap.add_argument("--build-only", action="store_true", help="steps 1-3 only (no chip needed)")
A = ap.parse_args()
D = os.path.abspath(A.dir)
os.chdir(D)  # one private working directory per compile: the DFC writes errors.pb / logs into the cwd


def snr(ref, x):
    return float(10 * np.log10((ref ** 2).mean() / (((x - ref) ** 2).mean() + 1e-20)))


def layers(runner):
    return runner.get_hn_dict()["layers"]


def build():
    from hailo_sdk_client import ClientRunner
    if A.smx_headroom:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import smx_headroom
        smx_headroom.install(A.smx_headroom)
    import onnx
    dims = [d.dim_value for d in onnx.load("m.onnx").graph.input[0].type.tensor_type.shape.dim]
    calib = np.random.default_rng(1).normal(0, 1, (A.calib, dims[2], dims[3], dims[1])).astype(np.float32)

    def quantize(extra):
        r = ClientRunner(hw_arch="hailo8")
        r.translate_onnx_model(os.path.join(D, "m.onnx"), "probe")
        script = "model_optimization_flavor(optimization_level=0, compression_level=0)\n"
        if A.a16:
            ls = [k.split("/")[-1] for k, v in layers(r).items()]  # input layer included -> UINT16 input
            script += f"quantization_param([{', '.join(ls)}], precision_mode=a16_w16)\n"
        script += "".join(x.strip() + "\n" for x in extra if x.strip())
        r.load_model_script(script)
        r.optimize(calib)
        return r, script

    extra = (open(A.alls).read().splitlines() if os.path.isfile(A.alls) else A.alls.split(";"))
    r, script = quantize(extra)
    if A.fix_dynmm:  # [0, 4 x calibrated max] on every producer of a dynamic matmul's second operand: its scale is
        # range/255 while the matmul reads int8, so values above HALF the range wrap (2x left 4.3% wrong, 4x exact)
        L, P = layers(r), r.get_params_translated()
        fix = []
        for k, v in L.items():
            if v["type"] == "matmul" and v.get("params", {}).get("dynamic_weights"):
                s = v["input"][1]
                while L[s]["type"] == "format_conversion":
                    s = L[s]["input"][0]
                lo, hi = (float(x) for x in np.asarray(P[f"{s}/limvals_out:0"]).ravel()[:2])
                fix.append(f"quantization_param({s.split('/')[-1]}, force_range_out=[{min(lo, 0):.6g}, {4 * hi:.6g}])")
        print("fix-dynmm:", fix)
        r, script = quantize(extra + fix)
    open("script.alls", "w").write(script)
    r.save_har("q.har")
    hef = r.compile()
    open("m.hef", "wb").write(hef)
    r.save_har("c.har")
    with tarfile.open("c.har") as t:
        nm = next((x for x in t.getnames() if x.endswith(".auto.alls")), None)
        auto = t.extractfile(nm).read().decode() if nm else ""
    open("auto.alls", "w").write(auto)
    for ln in auto.splitlines():  # the compiler's own decisions that the reports refer to
        if re.search(r"defuse\(|manual_scs_selection", ln):
            print("auto.alls:", ln.strip())


def emulate(har, xf):
    from hailo_sdk_client import ClientRunner, InferenceContext
    r = ClientRunner(har=har)
    with r.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
        return np.asarray(r.infer(ctx, xf))


def chip(hef, ii, X, one_by_one):
    from hailo_platform import (ConfigureParams, FormatType, HailoStreamInterface, InferVStreams, InputVStreamParams,
                                OutputVStreamParams, VDevice)
    p = VDevice.create_params()
    if A.device:
        p.device_ids = [A.device]
    vd = VDevice(p)
    try:
        ng = vd.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        # FormatType.AUTO rejects uint16 arrays on a UINT16 input (report #21): give the format explicitly
        ip = InputVStreamParams.make(ng, format_type=FormatType.UINT16 if X.dtype == np.uint16 else FormatType.UINT8)
        op = OutputVStreamParams.make(ng, format_type=FormatType.AUTO)
        with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
            run = lambda x: np.asarray(next(iter(pipe.infer({ii.name: x}).values()))).copy()  # noqa: E731
            batch = run(X)
            single = np.concatenate([run(X[i:i + 1]) for i in range(len(X))]) if one_by_one else None
            return batch, single
    finally:
        vd.release()


def main():
    if not A.no_compile:
        build()
    if A.build_only:
        return
    from hailo_platform import HEF
    import onnxruntime as ort
    hef = HEF("m.hef")
    ii, oi = hef.get_input_vstream_infos()[0], hef.get_output_vstream_infos()[0]
    u16 = "16" in str(ii.format.type)
    hi_in, hi_out = (65535 if u16 else 255), (65535 if "16" in str(oi.format.type) else 255)
    qi, qo = ii.quant_info, oi.quant_info
    deq_in = lambda q: (q.astype(np.float32) - qi.qp_zp) * qi.qp_scale  # noqa: E731
    deq_out = lambda q: (q.astype(np.float32) - qo.qp_zp) * qo.qp_scale  # noqa: E731
    sess = ort.InferenceSession("m.onnx", providers=["CPUExecutionProvider"])
    res = {"dir": D, "contexts": hef.get_network_groups_infos()[0].name, "input": str(ii.format.type),
           "output": str(oi.format.type), "rows": []}

    if A.u16_boundary:  # all zeros (raw = zp) except one element in the middle
        H, W, C = ii.shape
        vals = [30000, 32000, 32767, 33000, 35000, 40000, 50000, 65535]
        X = np.full((len(vals), H, W, C), int(round(qi.qp_zp)), np.uint16)
        for i, v in enumerate(vals):
            X[i, H // 2, W // 2, 0] = v
        e = emulate("q.har", deq_in(X))
        c, _ = chip(hef, ii, X, False)
        eq = np.clip(np.round(e.reshape(c.shape) / qo.qp_scale + qo.qp_zp), 0, hi_out)
        for i, v in enumerate(vals):
            res["rows"].append({"raw_input": v, "chip_out": int(c[i, H // 2, W // 2, 0]),
                                "emu_out": int(eq[i, H // 2, W // 2, 0]),
                                "elements_differing": int((c[i].astype(int) != eq[i].astype(int)).sum())})
            print(json.dumps(res["rows"][-1]))
        json.dump(res, open("result.json", "w"), indent=1)
        return

    for amp in (float(a) for a in A.amps.split(",")):
        rng = np.random.default_rng(0)
        X = np.clip(np.round(rng.normal(0, amp, (A.frames, *ii.shape)) / qi.qp_scale + qi.qp_zp), 0, hi_in)
        X = X.astype(np.uint16 if u16 else np.uint8)
        xf = deq_in(X)
        F = np.concatenate([sess.run(None, {"x": xf[i:i + 1].transpose(0, 3, 1, 2)})[0] for i in range(len(X))])
        F = F.transpose(0, 2, 3, 1)
        e = emulate("q.har", xf).reshape(F.shape)
        c, c1 = chip(hef, ii, X, True)
        c, c1 = c.reshape(F.shape).astype(np.int64), c1.reshape(F.shape).astype(np.int64)
        eq = np.clip(np.round(e / qo.qp_scale + qo.qp_zp), 0, hi_out).astype(np.int64)
        dd = np.abs(c - eq)
        row = {"amp": amp, "chip_ne_emu_pct": round(100 * float((dd > 0).mean()), 2), "max_diff_lsb": int(dd.max()),
               "snr_chip_db": round(snr(F, deq_out(c)), 1), "snr_emu_db": round(snr(F, deq_out(eq)), 1),
               "chip_batch_eq_1by1": bool((c == c1).all())}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    json.dump(res, open("result.json", "w"), indent=1)


if __name__ == "__main__":
    main()

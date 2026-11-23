"""cve.py HAR HEF DEV: same float inputs (seeded) through the Hailo-8 (FLOAT32 vstreams: host quantize/dequantize)
and the SDK_BIT_EXACT emulator of HAR; print per-output max/mean |diff| and the fraction of exactly equal values."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
import numpy as np
HAR, HEFP, dev = sys.argv[1], sys.argv[2], int(sys.argv[3])
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams, InputVStreamParams,
                            OutputVStreamParams, VDevice, Device)
hef = HEF(HEFP)
ins = {i.name: tuple(i.shape) for i in hef.get_input_vstream_infos()}
rng = np.random.default_rng(0)
X = {n: rng.normal(0, 1, (2,) + s).astype(np.float32) for n, s in ins.items()}
p = VDevice.create_params(); p.device_ids = [Device.scan()[dev]]
with VDevice(p) as vd:
    ng = vd.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32); op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
        HW = pipe.infer(X)
from hailo_sdk_client import ClientRunner, InferenceContext
r = ClientRunner(har=HAR)
with r.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
    # emulator input order = HN input layers; match by the vstream name suffix
    names = [n for n in r.get_hn_dict()["layers"] if r.get_hn_dict()["layers"][n]["type"] == "input_layer"]
    data = {n: X[next(k for k in X if k.split("/")[-1] == n.split("/")[-1])] for n in names}
    E = r.infer(ctx, data)
E = E if isinstance(E, list) else [E]
for (k, h), e in zip(HW.items(), E):
    h, e = np.asarray(h, np.float64), np.asarray(e, np.float64).reshape(np.asarray(h).shape)
    d = np.abs(h - e)
    print(f"{k}: shape {h.shape} max|d| {d.max():.3g} mean|d| {d.mean():.3g} exact {np.mean(d == 0):.4f} range {h.min():.3g}..{h.max():.3g}")

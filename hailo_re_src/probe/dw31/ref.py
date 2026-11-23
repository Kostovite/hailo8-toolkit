"""ref.py HEF DEV: run the dw31 sub-HEF on Hailo-8 with seeded float inputs and compare with a float64 reference of
dw31 (depthwise 3x3 VALID + bias + elementwise add of the 2nd input) from the full HAR's native weights."""
import io, sys, tarfile, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams, InputVStreamParams,
                            OutputVStreamParams, VDevice, Device)
HEFP, dev = sys.argv[1], int(sys.argv[2])
G = "/mnt/storage/lib_project/models/Hailo_model/convert/runs/layout_g3/q.har"
with tarfile.open(G) as t:
    z = np.load(io.BytesIO(t.extractfile("layout.npz").read()))
K = z["layout/dw31/kernel:0"].astype(np.float64)   # (3, 3, 64, 1)
b = z["layout/dw31/bias:0"].astype(np.float64)
hef = HEF(HEFP); ins = {i.name: tuple(i.shape) for i in hef.get_input_vstream_infos()}
print("inputs", ins, "kernel", K.shape, "bias", b.shape)
rng = np.random.default_rng(0)
X = {n: rng.uniform(-0.4, 3.0, (2,) + s).astype(np.float32) for n, s in ins.items()}  # inside the calibrated input ranges (-0.47..11.6 / 9.48)
p = VDevice.create_params(); p.device_ids = [Device.scan()[dev]]
with VDevice(p) as vd:
    ng = vd.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32); op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
        Y = np.asarray(next(iter(pipe.infer(X).values())), np.float64)
a = next(v for n, v in X.items() if ins[n][0] == 226).astype(np.float64)   # padded main input
r = next(v for n, v in X.items() if ins[n][0] == 224).astype(np.float64)   # residual (conv14)
ref = np.clip(sum(a[:, i:i + 224, j:j + 224, :] * K[i, j, :, 0] for i in range(3) for j in range(3)) + b + r, -0.637489, 9.414776)  # output range
d = np.abs(Y - ref); rel = d.max() / np.abs(ref).max()
print(f"chip vs float64 ref: max|d| {d.max():.4g}  mean|d| {d.mean():.4g}  max rel {rel:.3g}  ref range {ref.min():.3g}..{ref.max():.3g}")
print("per-channel max|d| (first 16):", np.round(d.max(axis=(0, 1, 2))[:16], 4))

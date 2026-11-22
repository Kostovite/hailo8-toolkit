"""Who is right, chip or emulator? Both vs the float ONNX model (onnxruntime) on the same dequantized input frames.
Also runs the emulator one frame at a time, to expose batch effects in the emulator itself."""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
import chip_vs_emu as cv
import onnxruntime as ort
from hailo_platform import HEF, VDevice
from hailo_sdk_client import ClientRunner, InferenceContext
d = sys.argv[1]; hef = HEF(f"{d}/m.hef"); ii = hef.get_input_vstream_infos()[0]; o = hef.get_output_vstream_infos()[0]
X = cv.frames(hef)[ii.name]
xf = (X.astype(np.float32) - ii.quant_info.qp_zp) * ii.quant_info.qp_scale
s = ort.InferenceSession(f"{d}/m.onnx")
F = np.concatenate([s.run(None, {s.get_inputs()[0].name: xf[i:i + 1].transpose(0, 3, 1, 2)})[0] for i in range(len(xf))])
F = F.transpose(0, 2, 3, 1) if F.ndim == 4 else F
deq = lambda q: (q.astype(np.float32) - o.quant_info.qp_zp) * o.quant_info.qp_scale
z = np.load(f"{d}/{cv.REF}")["arr_0"].reshape(F.shape)
runner = ClientRunner(har=f"{d}/q.har")
with runner.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
    z1 = np.concatenate([np.asarray(runner.infer(ctx, xf[i:i + 1])) for i in range(len(xf))]).reshape(F.shape)
p = VDevice.create_params(); p.device_ids = [os.environ.get("DEVICE", "0000:03:00.0")]; vd = VDevice(p)
try:
    c = deq(np.asarray(next(iter(cv.run16(vd, hef, {ii.name: X}).values())))).reshape(F.shape)
    c1 = np.concatenate([deq(np.asarray(next(iter(cv.run16(vd, hef, {ii.name: X[i:i + 1]}).values())))) for i in range(len(X))]).reshape(F.shape)
finally:
    vd.release()
def snr(a):
    return [round(float(10 * np.log10((F[i] ** 2).mean() / (((a[i] - F[i]) ** 2).mean() + 1e-12))), 1) for i in range(len(F))]
for nm, a in (("chip batch", c), ("chip 1-by-1", c1), ("emu batch", z), ("emu 1-by-1", z1)):
    print(f"{nm:12s} SNR dB vs float per frame:", snr(a))
print("emu batch == emu 1-by-1:", np.allclose(z, z1), " chip batch == chip 1-by-1:", np.allclose(c, c1))

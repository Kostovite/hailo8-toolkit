"""Chip SNR vs float (onnxruntime) at several input amplitudes: frames ~ N(0, a), quantized like the HEF input.
Usage: python snr_amp.py DIR [a1 a2 ...]   (DEVICE env; takes the chip lock)"""
import fcntl, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chip_vs_emu as cv
import onnxruntime as ort
from hailo_platform import HEF, VDevice
d = sys.argv[1]; amps = [float(a) for a in sys.argv[2:]] or [0.25, 0.5, 1.0, 2.0]
hef = HEF(f"{d}/m.hef"); ii = hef.get_input_vstream_infos()[0]; o = hef.get_output_vstream_infos()[0]; q = ii.quant_info
s = ort.InferenceSession(f"{d}/m.onnx"); dev = os.environ.get("DEVICE", "0000:03:00.0")
lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p); res = []
try:
    for a in amps:
        rng = np.random.default_rng(7)
        X = np.clip(np.round(rng.normal(0, a, (16, *ii.shape)) / q.qp_scale + q.qp_zp), 0, 255).astype(np.uint8)
        xf = (X.astype(np.float32) - q.qp_zp) * q.qp_scale
        F = np.concatenate([s.run(None, {s.get_inputs()[0].name: xf[i:i + 1].transpose(0, 3, 1, 2)})[0] for i in range(len(xf))]).transpose(0, 2, 3, 1)
        c = (np.asarray(next(iter(cv.run16(vd, hef, {ii.name: X}).values()))).astype(np.float32) - o.quant_info.qp_zp) * o.quant_info.qp_scale
        c = c.reshape(F.shape)
        res.append(round(float(10 * np.log10((F ** 2).mean() / ((c - F) ** 2).mean())), 2))
finally:
    vd.release()
print(os.path.basename(d), " ".join(f"a={a}:{r}dB" for a, r in zip(amps, res)))

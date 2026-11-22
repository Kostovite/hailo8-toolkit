"""LM block on the chip vs float: real token embeddings (DIR/calib.npy, NCHW) -> quantized input -> HEF on the chip
-> dequantized output, compared with onnxruntime float of DIR/m.onnx (per-token cosine, SNR) and, when the HEF
covers ALL layers, next-token top-1 agreement through the model's final norm + tied lm_head on the host.

Usage (DFC env): python llm_check.py MODEL_DIR HEF [SNAPSHOT_DIR]
"""
import fcntl
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/port")
import exact_check as ec  # noqa: E402
from hailo_platform import HEF, VDevice  # noqa: E402

d, hef_path = sys.argv[1], sys.argv[2]
X = np.load(f"{d}/calib.npy").astype(np.float32)  # [S, H, gh, gw]
hef = HEF(hef_path)
ii, oi = hef.get_input_vstream_infos()[0], hef.get_output_vstream_infos()[0]
xq = np.clip(np.round(X.transpose(0, 2, 3, 1) / ii.quant_info.qp_scale + ii.quant_info.qp_zp), 0, 255).astype(np.uint8)
xd = ((xq.astype(np.float32) - ii.quant_info.qp_zp) * ii.quant_info.qp_scale).transpose(0, 3, 1, 2)
s = ort.InferenceSession(f"{d}/m.onnx")
F = np.concatenate([s.run(None, {"x": xd[i:i + 1]})[0] for i in range(len(xd))])  # float on the same input
lk = open("/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_" + os.environ.get("DEVICE", "0000:03:00.0").replace(":", "_") + ".lock", "w")
fcntl.flock(lk, fcntl.LOCK_EX)
p = VDevice.create_params()
p.device_ids = [os.environ.get("DEVICE", "0000:03:00.0")]
vd = VDevice(p)
try:
    c = np.asarray(next(iter(ec.run(vd, hef_path, lambda h: {ii.name: xq}).values())))
finally:
    vd.release()
    fcntl.flock(lk, fcntl.LOCK_UN)
C = ((c.astype(np.float32) - oi.quant_info.qp_zp) * oi.quant_info.qp_scale).reshape(F.shape[0], *F.shape[2:],
                                                                                    F.shape[1]).transpose(0, 3, 1, 2)
n_tok = F.shape[2] * F.shape[3]
f_t = F.reshape(F.shape[0], F.shape[1], n_tok).transpose(0, 2, 1)  # [S, N, H]
c_t = C.reshape(C.shape[0], C.shape[1], n_tok).transpose(0, 2, 1)
cos = (f_t * c_t).sum(-1) / (np.linalg.norm(f_t, axis=-1) * np.linalg.norm(c_t, axis=-1) + 1e-9)
snr = 10 * np.log10((f_t ** 2).sum() / ((f_t - c_t) ** 2).sum())
print(f"chip vs float: SNR {snr:.1f} dB, per-token cosine mean {cos.mean():.4f} min {cos.min():.4f}")
if len(sys.argv) > 3:
    import glob
    import json
    from safetensors.numpy import load_file
    snap = sys.argv[3].rstrip("/") + "/"
    cfg = json.load(open(snap + "config.json"))
    sd = load_file(glob.glob(snap + "*.safetensors")[0])
    wn = sd["model.norm.weight"].astype(np.float32)
    E = sd["model.embed_tokens.weight"].astype(np.float32)  # tied lm_head

    def logits(t):
        return (t / np.sqrt((t ** 2).mean(-1, keepdims=True) + cfg["rms_norm_eps"]) * wn) @ E.T
    lf, lc = logits(f_t), logits(c_t)
    top_f, top_c = lf.argmax(-1), lc.argmax(-1)
    t5 = np.mean([top_f[i, j] in np.argsort(-lc[i, j])[:5] for i in range(len(top_f)) for j in range(n_tok)])
    print(f"next-token top-1 agreement {np.mean(top_f == top_c):.3f}, float top-1 in chip top-5 {t5:.3f}")

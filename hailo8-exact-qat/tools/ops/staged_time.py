"""StagedRunner on the real chip: throughput vs chunk size and agreement with the per-frame float pipeline.

Usage (hailo env): [SFX=_16] [DEVICE=cuda:0 (source gpu_env.sh)] [BACKEND=torch] python staged_time.py CASE_DIR
  (hvc/<op>: part1/part2 HEFs + host.onnx; BACKEND=torch uses the op's torch function below on DEVICE)
"""
import json
import os
import sys
import time

import numpy as np

from host_split import StagedRunner

d = sys.argv[1]
SFX = os.environ.get("SFX", "")
DEVICE = os.environ.get("DEVICE", "cpu")
BACKEND = os.environ.get("BACKEND", "onnx")


def torch_fn(op):
    """The hvc ops in torch, NHWC in/out (host_vs_chip.export)."""
    import torch
    import torch.nn.functional as F
    nchw = lambda f: lambda x: f(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()  # noqa: E731

    def topk(x):
        v, i = x.topk(4, dim=1)
        return torch.zeros_like(x).scatter(1, i, v)

    def grid(x):
        g = torch.stack(torch.meshgrid(torch.linspace(-1, 1, x.shape[2]), torch.linspace(-1, 1, x.shape[3]),
                                       indexing="ij"), -1)[None].flip(-1) * 0.9
        return F.grid_sample(x, g.to(x.device).expand(len(x), -1, -1, -1), align_corners=False)
    return {"Identity": lambda x: x, "Sin": torch.sin, "SiLU": F.silu, "Erf": torch.erf, "Atan": torch.atan,
            "TopK": nchw(topk), "GridSample": nchw(grid)}[op]
x = (np.random.default_rng(0).random((2048, 224, 224, 3)) * 255).astype(np.float32)
res = {}
ref = None
for K in (64, 256, 1024):
    host = torch_fn(json.load(open(f"{d}/meta.json"))["op"]) if BACKEND == "torch" else f"{d}/host.onnx"
    r = StagedRunner(f"{d}/part1{SFX}.hef", host, f"{d}/part2{SFX}.hef", chunk=K, device=DEVICE)
    r.run(x[:K])  # warm-up
    t = time.perf_counter()
    y = r.run(x)
    res[f"staged_{K}_fps"] = len(x) / (time.perf_counter() - t)
    res["host_batched"] = r.batched
    if ref is None:  # per-frame float pipeline (HailoRT dequant/quant), first 64 frames
        from hailo_platform import FormatType, InputVStreamParams, OutputVStreamParams
        p1, p2 = dict(r.p1), dict(r.p2)
        p1["op"] = OutputVStreamParams.make(p1["ng"], format_type=FormatType.FLOAT32)
        p2["ip"] = InputVStreamParams.make(p2["ng"], format_type=FormatType.FLOAT32)
        import onnxruntime as ort
        plain = ort.InferenceSession(f"{d}/host.onnx", providers=["CPUExecutionProvider"])
        f = r._infer(p1, x[:64])
        g = np.concatenate([plain.run(None, {"f": f[i:i + 1]})[0] for i in range(len(f))])
        ref = r._infer(p2, np.ascontiguousarray(g, np.float32))
    a, b = ref.astype(np.float64).ravel(), y[:64].astype(np.float64).ravel()
    res["agree_vs_float_path_db"] = float(10 * np.log10((a ** 2).sum() / max(((a - b) ** 2).sum(), 1e-30)))
    r.close()
print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in res.items()}))
json.dump(res, open(f"{d}/staged{SFX}_{BACKEND}_{DEVICE.replace(':', '')}.json", "w"), indent=1)

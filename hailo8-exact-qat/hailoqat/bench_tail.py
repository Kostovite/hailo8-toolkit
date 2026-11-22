"""Micro-benchmark of the fused MAC tail kernel on one real layer (default best/conv1, 448x448x64, bs 2)."""
import json
import sys
import time

import torch

sys.path.insert(0, ".")
import graph26  # noqa: E402
import qatfast  # noqa: E402

ROOT = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"
name = sys.argv[1] if len(sys.argv) > 1 else "best/conv1"
want = (sys.argv[2] if len(sys.argv) > 2 else "1") == "1"
dev = torch.device("cuda")
hn = json.load(open(f"{ROOT}/yolo26m_v14sec_L0.hn"))
P = graph26.Params(f"{ROOT}/capture_y26g", dev)
net = qatfast.QATNet(hn, P, dev).to(dev)
mod = net.mac[name.replace("/", "__")]
_, h, w, c = hn["layers"][name]["output_shapes"][0]
e = qatfast._empty(dev)
src = torch.randint(-20000, 20000, (2, c, h, w), device=dev).float()
flag = torch.zeros(1, dtype=torch.int32, device=dev)
b = mod.int_bias().detach().contiguous()
wo = torch.zeros(c, device=dev)
args = (src, e, 2, 0.0, mod.s, e, e, 1.0, 1.0, qatfast._cfg(mod.acc), True, b, qatfast._cfg(mod.bias_cfg),
        *mod.apu.args(), want, flag, wo)
for _ in range(3):
    qatfast.hq.tail(*args)
torch.cuda.synchronize()
n = 50
t = time.time()
for _ in range(n):
    qatfast.hq.tail(*args)
torch.cuda.synchronize()
dt = (time.time() - t) / n
el = src.numel()
byts = el * (4 + 4 + (4 + 1 if want else 0))
print(f"{name} {el / 1e6:.1f}M el: {dt * 1e3:.3f} ms  {el / dt / 1e9:.2f} Gel/s  {byts / dt / 1e9:.0f} GB/s  P={mod.apu.mant_i.shape[1]}")

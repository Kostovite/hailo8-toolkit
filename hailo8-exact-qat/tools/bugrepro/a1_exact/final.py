"""A1 positive test: graph26 (default = silicon rules) vs the chip on the mm_att probe matmul."""
import json, sys, numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import graph26 as G
hn = json.load(open("m.hn")); x = np.transpose(np.load("x.npy"), (0, 2, 3, 1)).astype(np.float64)
c = np.load("chip.npz")["probe|format_conversion3"].astype(np.int64).reshape(1, 100, 100)
P = G.Params("cap", "cpu"); print("silicon-unsigned producers:", G.silicon_weights_operands(P, hn["layers"]))
with torch.no_grad(): out, _ = G.run(hn, G.Params("cap", "cpu"), x, "cpu", check=False)
y = np.transpose(out["probe/matmul1"].numpy(), (0, 2, 3, 1)).reshape(1, 100, 100).astype(np.int64)
print("matmul vs chip: mismatch", int((y != c).sum()), "of", c.size)

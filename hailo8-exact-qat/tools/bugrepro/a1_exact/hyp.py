"""A1: which integer model of the dynamic matmul reproduces the chip bit-exactly (graph26, cont=False)."""
import json, sys, numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import graph26 as G, intops as I
class Z(dict):
    @property
    def files(self): return list(self.keys())
hn = json.load(open("m.hn")); x = np.transpose(np.load("x.npy"), (0, 2, 3, 1)).astype(np.float64)
c = np.load("chip.npz")["probe|format_conversion3"].astype(np.int64).reshape(1, 100, 100)
orig = I._matmul_layer_acc
def go(tag, f0=None, f1=None, k_uns=False, flags=("out_signed", "clip2_signed")):
    def spy(x0, x1, *a):
        return orig(f0(x0) if f0 else x0, f1(x1) if f1 else x1, *a)
    I._matmul_layer_acc = spy; P = G.Params("cap", "cpu")
    if k_uns:
        for k, r in P.ops.items():
            if k.startswith("probe/conv4/"):
                for a in flags:
                    if a in r: r[a] = False
        d = Z({k: P.conv["probe/conv4"][k] for k in P.conv["probe/conv4"].files})
        for a in flags: d[a] = np.array(False)
        P.conv["probe/conv4"] = d
    with torch.no_grad(): out, _ = G.run(hn, P, x, "cpu", cont=False, check=False) if "check" in G.run.__code__.co_varnames else G.run(hn, P, x, "cpu", cont=False)
    y = np.transpose(out["probe/matmul1"].numpy(), (0, 2, 3, 1)).reshape(1, 100, 100).astype(np.int64)
    k4 = out["probe/conv4"]; print(f"{tag:36s} K range {int(k4.min())}..{int(k4.max())}  mismatch vs chip {int((y != c).sum()):5d}  max {int(np.abs(y - c).max()):3d}  mean|d| {float(np.abs(y - c).mean()):.3f}")
go("SDK")
go("Q read as int8", f0=lambda v: I.wrap(v, 8, True))
go("K unsigned 0..255 (conv4 APU)", k_uns=True)
go("K unsigned, read as int8", f1=lambda v: I.wrap(v, 8, True), k_uns=True)
go("only out_signed=False", k_uns=True, flags=("out_signed",))
go("only clip2_signed=False", k_uns=True, flags=("clip2_signed",))

"""P.V: which integer model of matmul2 reproduces the chip (graph26, cont=False)."""
import json, sys, numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import graph26 as G, intops as I
hn = json.load(open("m.hn")); x = np.transpose(np.load("x.npy"), (0, 2, 3, 1)).astype(np.float64)
c = next(iter(np.load("chip.npz").values())).astype(np.int64)  # NHWC
orig = I._matmul_layer_acc
names = []
def go(tag, f0=None, f1=None, only="probe/matmul2"):
    cur = {"n": None}
    def spy(x0, x1, *a):
        return orig(f0(x0) if f0 else x0, f1(x1) if f1 else x1, *a)
    I._matmul_layer_acc = spy
    o_disp = G._dispatch
    def disp(n, *a, **k):  # apply the hypothesis to one matmul only
        I._matmul_layer_acc = spy if n == only else orig
        return o_disp(n, *a, **k)
    G._dispatch = disp
    with torch.no_grad(): out, _ = G.run(hn, G.Params("cap", "cpu"), x, "cpu", check=False)
    G._dispatch = o_disp; I._matmul_layer_acc = orig
    y = out["probe/format_conversion4"].numpy(); y = np.transpose(y, (0, 2, 3, 1)).astype(np.int64)
    print(f"{tag:34s} mismatch vs chip {int((y != c).sum()):5d} of {c.size}  max {int(np.abs(y - c).max()):3d}  mean|d| {float(np.abs(y - c).mean()):.3f}")
u8 = lambda v: torch.remainder(v, 256)
go("SDK")
go("V (2nd) read as uint8", f1=u8)
go("P (1st) read as int8", f0=lambda v: I.wrap(v, 8, True))
go("V clipped >= 0", f1=lambda v: v.clamp(min=0))
go("V abs", f1=lambda v: v.abs())

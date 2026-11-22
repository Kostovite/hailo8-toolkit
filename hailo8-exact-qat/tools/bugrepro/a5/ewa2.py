import pickle, sys, numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import intops as I, atomic as A
P = pickle.load(open("sum3/cap/_all_ops_np.pkl", "rb"))
c = torch.as_tensor(np.transpose(next(iter(np.load("sum3/chip.npz").values())).astype(np.float64), (0, 3, 1, 2)))
ea, ba, r = P["Sum3/ew_add2/elementwise_add_op"], P["Sum3/ew_add2/bias_add_op"], P["Sum3/ew_add2/act_op"]
print([k for k in ea if k.startswith("in")], {k: np.asarray(ea[k]).shape for k in ea if k.startswith("in")})
x0 = np.asarray(ea["in0"], np.float64); x1 = np.asarray(ea["in1"], np.float64) if "in1" in ea else None
kq = np.asarray(ea["kernel_q"], np.float64).reshape(2, -1); b = np.asarray(ba["bias"], np.float64)
zp = [np.asarray(z, np.float64) for z in ea.get("input_zero_points", [0, 0])]
print("in ranges", x0.min(), x0.max(), x1.min() if x1 is not None else None, x1.max() if x1 is not None else None, "zps", [np.asarray(z).ravel()[:2] for z in zp], "bias", b[:3])
acc = x0 * kq[0] + x1 * kq[1]
a_ref = np.asarray(r["in0"], np.float64); print("acc+bias == act_in?", np.allclose(acc + b, a_ref), "max diff", np.abs(acc + b - a_ref).max())
def act(v): return A.activation(r, torch.as_tensor(np.transpose(v, (0, 3, 1, 2))))
def test(tag, v): print(f"{tag:42s} mismatch vs chip {int((act(v) != c).sum())}")
test("ours", a_ref)
for s in (4, 8, 12, 16):  # product computed from byte planes, the low plane's product truncated by s bits
    for nm, f in (("floor", np.floor), ("trunc", np.trunc)):
        lo0, hi0 = x0 % 256, x0 // 256; lo1, hi1 = x1 % 256, x1 // 256
        p0 = hi0 * kq[0] * 256 + f(lo0 * kq[0] / 2 ** s) * 2 ** s
        p1 = hi1 * kq[1] * 256 + f(lo1 * kq[1] / 2 ** s) * 2 ** s
        test(f"byte-split, low products {nm} >> {s}", p0 + p1 + b)
for s in (2, 4, 6, 8, 10):
    test(f"each product floor >> {s}", np.floor(x0 * kq[0] / 2 ** s) * 2 ** s + np.floor(x1 * kq[1] / 2 ** s) * 2 ** s + b)

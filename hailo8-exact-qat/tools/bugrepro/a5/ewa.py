import pickle, sys, numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import intops as I, atomic as A
P = pickle.load(open("sum3/cap/_all_ops_np.pkl", "rb"))
c = torch.as_tensor(np.transpose(next(iter(np.load("sum3/chip.npz").values())).astype(np.float64), (0, 3, 1, 2)))
r = P["Sum3/ew_add2/act_op"]; a = torch.as_tensor(np.transpose(np.asarray(r["in0"], np.float64), (0, 3, 1, 2)))
ea = P["Sum3/ew_add2/elementwise_add_op"]
print("ew_add2 act_in range", float(a.min()), float(a.max()), "kernel_q", np.asarray(ea["kernel_q"]).reshape(2, -1)[:, :3].tolist(), "mant", np.asarray(r["mant"]).ravel()[:2], "shift", np.asarray(r["shift"]).ravel()[:2])
base = A.activation(r, a)
print("baseline", int((base != c).sum()))
for k in range(1, 12):
    print(k, "floor", int((A.activation(r, torch.floor(a / 2 ** k) * 2 ** k) != c).sum()), "round", int((A.activation(r, torch.round(a / 2 ** k) * 2 ** k) != c).sum()))
bad = torch.nonzero(base != c)[:6]
for n, ch, h, w in bad.tolist():
    m = float(np.asarray(r["mant"]).reshape(-1)[ch]); sh = float(np.asarray(r["shift"]).reshape(-1)[ch]); of = float(np.asarray(r["offs"]).reshape(-1)[ch])
    v = float(a[n, ch, h, w]); print(" bad: act_in", v, "exact out", (v * m / 2 ** sh + of) / 2 ** int(r["final_shift"]), "emu", float(base[n, ch, h, w]), "chip", float(c[n, ch, h, w]))

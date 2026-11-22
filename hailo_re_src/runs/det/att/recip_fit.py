"""Fit hailoqat's APU to the chip reciprocal map: acc = reduce_sum(-2*x)/2^s, bias, act; compare per row sum S."""
import pickle, sys, numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import atomic as A, intops as I
P = pickle.load(open("capture_mTPC/_all_ops.pkl", "rb"))
rs, rb, ra = (P[f"best/reduce_sum_softmax1/{k}"] for k in ("reduce_sum_op", "bias_add_op", "act_op"))
for k in ("kernel_q", "pre_acc_shift", "bias"):
    print(k, np.asarray(rs.get(k, rb.get(k))).ravel()[:4])
print({k: v for k, v in A.lossy(rs) and {"acc": A.lossy(rs), "bias": A.lossy(rb), "c1": A.lossy(ra, "clip1"), "c2": A.lossy(ra, "clip2"), "out": A.lossy(ra, "out")}.items()})
print("thr", np.asarray(ra["thr"]).reshape(-1, 4)[:, 0], "shift", np.asarray(ra["shift"]).reshape(-1, 4)[:, 0])
d = "runs/det/att"; S = np.concatenate([np.load(f"{d}/r{i}.npz")["S"] for i in (1, 2, 3)]); C = np.concatenate([np.load(f"{d}/r{i}.npz")["C"] for i in (1, 2, 3)])
u, inv = np.unique(S, return_inverse=True)
x = torch.zeros(len(u), 784, 1, 4, dtype=torch.float64)  # N, C(=784 keys), H, W(=4 heads) -> spread S over keys
q, r = np.divmod(u, 784); base = torch.as_tensor(q, dtype=torch.float64)[:, None].repeat(1, 784)
base += (torch.arange(784)[None, :] < torch.as_tensor(r)[:, None]).double()
assert (base.max() <= 255)
x = base.repeat(1, 4)[:, :, None, None]  # N, 4 groups x 784 keys, 1, 1
acc = A.reduce_sum(rs, x); b = A.add_bias(rb, acc)
def act(b, mode):
    thr = A._t(ra["thr"], b); mant, sh, of = A._t(ra["mant"], b), A._t(ra["shift"], b), A._t(ra["offs"], b)
    if mode == "edge":  # chip hypothesis: clamp into the first/last segment instead of zeroing
        lo, hi = thr[0].clone(), thr[-1].clone(); thr = thr.clone(); thr[0] = -2.0 ** 40; thr[-1] = 2.0 ** 40
    return I.apu(b, thr, mant, sh, of, int(ra["final_shift"]), bool(ra["mant_signed"]), A.lossy(ra, "clip1"), A.lossy(ra, "clip2"), A.lossy(ra, "out"))
print("apu in at S=32768..32771:", b[(u >= 32768) & (u <= 32771)][:, 0, 0, 0].tolist() if ((u >= 32768) & (u <= 32771)).any() else "-")
for mode in ("sdk", "edge"):
    y = act(b, mode)[:, 0, 0, 0].numpy().astype(np.int64)[inv]
    bad = y != C
    print(mode, "rows", C.size, "mismatch", int(bad.sum()), "bad S:", np.unique(S[bad])[:20].tolist())

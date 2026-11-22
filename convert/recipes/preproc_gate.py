"""preproc_gate.py orient|flat OUT_DIR PREFIX CKPT
orient: decide_ll decisions from PREFIX_0 vs torch ref_0 on 2000 eval canvases; PASS if diffs <= MAX_DIFF (default 0).
flat:   mesh |PREFIX - ref| in % of frame (x50) mean <= FLAT_TOL (default .05) and GT error delta per group printed."""
import os, sys, numpy as np, torch, torch.nn.functional as F
kind, out, pre, ck = sys.argv[1:5]; E = "/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/eval"; torch.set_num_threads(8)
if kind == "orient":
    sys.path.insert(0, "/mnt/storage/lib_project/models/layout_teacher/train"); import train_orient as T
    net = T.OriNet(); net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)); net.eval()
    d = np.load(E + "/orient_eval.npz"); nh, nw, k = d["nh"], d["nw"], d["k"]
    def dec(f):
        r = []
        with torch.no_grad():
            for i in range(len(f)):
                x = torch.from_numpy(np.ascontiguousarray(f[i, :nh[i] // 32, :nw[i] // 32])).permute(2, 0, 1)[None].float()
                p = F.conv2d(F.avg_pool2d(x, 7, 1), net.fc.weight[:, :, None, None], net.fc.bias).softmax(1)[0].flatten(1).t().numpy()
                r.append(T.decide_ll(p)[0])
        return np.asarray(r)
    a = dec(np.load(f"{out}/ref_0.npy", mmap_mode="r")); b = dec(np.load(f"{out}/{pre}_0.npy", mmap_mode="r")); up = k == 0
    nd = int((a != b).sum()); lim = int(os.environ.get("MAX_DIFF", "0"))
    for nm, z in (("torch", a), (pre, b)):
        print(f"RESULT {nm}: false_turn {int((z[up] != 0).sum())}/{int(up.sum())} correct_turn {int((z[~up] == k[~up]).sum())}/{int((~up).sum())}")
    print(f"RESULT decision_diffs_vs_torch {nd} / {len(k)}"); print("GATE", "PASS" if nd <= lim else "FAIL"); sys.exit(0 if nd <= lim else 2)
d = np.load(E + "/flat_eval.npz"); gt = d["gt"]
G = {"synth_photo": (d["set"] == "synth") & (d["kind"] == 1), "synth_flat": (d["set"] == "synth") & (d["kind"] == 0), "uvdoc": d["set"] == "uvdoc"}
r = np.load(f"{out}/ref_0.npy").transpose(0, 3, 1, 2); m = np.load(f"{out}/{pre}_0.npy").transpose(0, 3, 1, 2)
df = np.abs(m - r).mean((1, 2, 3)) * 50; er, em = np.abs(r - gt).mean((1, 2, 3)) * 50, np.abs(m - gt).mean((1, 2, 3)) * 50
for k_, s in G.items(): print(f"RESULT {k_} n {int(s.sum())} gt_err torch {er[s].mean():.4f} {pre} {em[s].mean():.4f}")
tol = float(os.environ.get("FLAT_TOL", ".05")); print(f"RESULT |{pre}-torch| %frame mean {df.mean():.4f} max {df.max():.4f}")
print("GATE", "PASS" if df.mean() <= tol else "FAIL"); sys.exit(0 if df.mean() <= tol else 2)

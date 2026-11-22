"""flat2 FP-vs-chip gate.  build: synthetic val (train_flat.split va, DS seed 1234, 400) + UVDoc val captures (uv_only, 400)
-> eval/flat_frames.npy (uint8 712x488) + flat_eval.npz (gt mesh, kind, set, FP mesh).  score MESH.npy...: error % of frame."""
import sys, os, numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/layout_teacher/train"); sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/uvdoc/src")
os.chdir("/mnt/storage/lib_project/models/layout_teacher/train")
import train_flat as TF
from model import UVDocnet
OUT = "/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/eval"; UV = "/mnt/storage/lib_project/models/layout_teacher/data_ext/uvdoc/UVDoc_final"
if sys.argv[1] == "build":
    net = UVDocnet(num_filter=32, kernel_size=5); sd = torch.load("../models_flat/flat2.pt", map_location="cpu"); net.load_state_dict(sd.get("model_state", sd)); net.eval()
    tr, va = TF.split(); X, G, K, S = [], [], [], []
    for name, ds in (("synth", TF.DS(va, seed=1234)), ("uvdoc", TF.DS(va, seed=1234, uv=(UV, TF.uv_list(UV)[1]), uv_only=True))):
        for _, (x, g, k) in zip(range(400), ds):
            X.append(x.permute(1, 2, 0).numpy()); G.append(g.numpy()); K.append(k); S.append(name)
    X = np.stack(X); P = []
    with torch.no_grad():
        for i in range(0, len(X), 20):
            P.append(net(torch.from_numpy(X[i:i + 20]).permute(0, 3, 1, 2).float() / 255)[0].numpy())
    np.save(f"{OUT}/flat_frames.npy", X); np.savez(f"{OUT}/flat_eval.npz", gt=np.stack(G), kind=K, set=S, fp=np.concatenate(P)); print("built", X.shape)
else:
    d = np.load(f"{OUT}/flat_eval.npz"); gt, fp = d["gt"], d["fp"]; groups = {"synth photo": (d["set"] == "synth") & (d["kind"] == 1), "synth flat": (d["set"] == "synth") & (d["kind"] == 0), "uvdoc real": d["set"] == "uvdoc"}
    def rep(name, m):                                   # m (N, 2, 45, 31) NCHW
        e = np.abs(m - gt).mean((1, 2, 3)) * 50; dfp = np.abs(m - fp).mean((1, 2, 3)) * 50
        print(f"{name:26s} " + "  ".join(f"{g}: {e[s].mean():.3f}% (p95 {np.percentile(e[s], 95):.2f})" for g, s in groups.items()) + f"   |chip-FP| {dfp.mean():.4f}% max {dfp.max():.3f}")
    rep("FP torch", fp)
    for p in sys.argv[2:]:
        m = np.load(p); m = m.transpose(0, 3, 1, 2) if m.shape[-1] == 2 else m; rep(os.path.basename(p), m)

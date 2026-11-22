"""router_data.py CKPT TRAIN_FEATS EVAL_FEATS OUT_DIR -> calib.npy, eval_x.npy (N,1,1,1024 scaled), ref_<i>.npy (size order: lang, kind, body)"""
import sys, numpy as np, torch
sys.path[:0] = ["/mnt/storage/lib_project/models/Dolphin", "/mnt/storage/lib_project/models/Dolphin/wordlid"]
from region_router import RegionRouter
ck, tr, ev, out = sys.argv[1:5]; s = np.load(f"{out}/host/router_scale.npy")
net = RegionRouter(); net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["router"]); net.eval()
t = np.load(tr).astype(np.float32); e = np.load(ev).astype(np.float32); e = e[: len(e) // 8 * 8]
np.save(f"{out}/calib.npy", (t / s).reshape(-1, 1, 1, 1024)); np.save(f"{out}/eval_x.npy", (e / s).reshape(-1, 1, 1, 1024))
with torch.no_grad(): r = net(torch.from_numpy(e))
for i, k in enumerate(("lang", "kind", "body")): np.save(f"{out}/ref_{i}.npy", r[k].numpy())
print("calib", t.shape, "eval", e.shape, "scaled eval |max|", float(np.abs(e / s).max()))

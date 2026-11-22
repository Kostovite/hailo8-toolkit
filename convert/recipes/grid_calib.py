"""grid_calib.py N OUT_NPY : the stratified GridNet calibration plan of gridnet_hailo/quantize.py (TRAIN shards only),
saved as uint8 NHWC so it can be reused / audited. Run with HPY (hailo env torch), cwd-independent."""
import os, sys
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/gridnet_hailo")
import common as C, numpy as np
from torch.utils.data import DataLoader, Subset
n, out = int(sys.argv[1]), sys.argv[2]; rng = np.random.default_rng(0)
OTHER = [t for t in ("vttrain", "vktrain", "vntrain", "chtrain", "cvtrain", "bktrain") if os.path.exists(os.path.join(C.T.DATA, f"{t}_lab_000.npz"))]
plan = []; g5 = C.T.GridShards("g5train", False); kinds = np.concatenate([l["kind"] for l in g5.lab]); half = n // 2 if OTHER else n
ks, cnt = np.unique(kinds, return_counts=True); share = np.maximum(cnt / cnt.sum(), 1 / 16); share = share / share.sum()
for k, sh in zip(ks, share):
    pool = np.nonzero(kinds == k)[0]; plan.append(("g5train", rng.choice(pool, min(len(pool), int(round(half * sh))), replace=False)))
for t in OTHER:
    m = len(C.T.GridShards(t, False)); plan.append((t, rng.choice(m, min(m, (n - half) // len(OTHER)), replace=False)))
tot = sum(len(ix) for _, ix in plan); cal = np.lib.format.open_memmap(out, "w+", np.uint8, (tot, 896, 896, 1)); o = 0; lo, hi = 1e9, -1e9
for t, ix in plan:
    ds = g5 if t == "g5train" else C.T.GridShards(t, False)
    for b in DataLoader(Subset(ds, sorted(ix.tolist())), batch_size=32, num_workers=16, collate_fn=lambda b: C.T.collate(b, False)):
        x = b["x"].numpy().transpose(0, 2, 3, 1); lo, hi = min(lo, x.min()), max(hi, x.max())
        if hi <= 1.0 + 1e-6: x = x * 255.0
        cal[o:o + len(x)] = np.clip(np.round(x), 0, 255).astype(np.uint8); o += len(x)
    print(f"calib {t:8s} {len(ix):5d}", flush=True)
cal.flush(); print("calib", cal.shape, "raw range", lo, hi, "DATA", C.T.DATA)

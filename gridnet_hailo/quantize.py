"""usage: OPT_LEVEL=0 [CK=ckpt] python quantize.py PARSED_HAR OUT_HAR N_CALIB [EXTRA_ALLS_FILE]
Calibration is STRATIFIED over every data type the model serves, TRAIN shards only (never the val sets the scores use):
half from g5train spread over its region kinds (each kind >= 1/16 of that half), the rest split evenly over the
vertical Han Nom (vttrain), tight-leading Vietnamese (vntrain), CJK horizontal (chtrain) and CJK vertical (cvtrain), tight scan-degraded Han Nom (vktrain).
A random draw from one mix let the rarest inputs set no range at all."""
import os, sys, time
import common as C
import numpy as np
from torch.utils.data import DataLoader, Subset
from hailo_sdk_client import ClientRunner
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
extra = open(sys.argv[4]).read() if len(sys.argv) > 4 else ""
rng = np.random.default_rng(0)
OTHER = [t for t in ("vttrain", "vktrain", "vntrain", "chtrain", "cvtrain", "bktrain") if os.path.exists(os.path.join(C.T.DATA, f"{t}_lab_000.npz"))]
plan = []                                        # (tag, indices)
g5 = C.T.GridShards("g5train", False)
kinds = np.concatenate([l["kind"] for l in g5.lab])
half = n // 2 if OTHER else n
ks, cnt = np.unique(kinds, return_counts=True)
share = np.maximum(cnt / cnt.sum(), 1 / 16); share = share / share.sum()
for k, sh in zip(ks, share):
    pool = np.nonzero(kinds == k)[0]
    plan.append(("g5train", rng.choice(pool, min(len(pool), int(round(half * sh))), replace=False)))
for t in OTHER:
    m = len(C.T.GridShards(t, False))
    plan.append((t, rng.choice(m, min(m, (n - half) // len(OTHER)), replace=False)))
cal = np.zeros((sum(len(ix) for _, ix in plan), 896, 896, 1), np.float32)
o = 0
for t, ix in plan:
    ds = g5 if t == "g5train" else C.T.GridShards(t, False)
    for b in DataLoader(Subset(ds, sorted(ix.tolist())), batch_size=32, num_workers=16,
                        collate_fn=lambda b: C.T.collate(b, False)):
        x = b["x"].numpy().transpose(0, 2, 3, 1)
        cal[o:o + len(x)] = x; o += len(x)
    print(f"calib {t:8s} {len(ix):5d}", flush=True)
cal = cal[:o]
n = len(cal)
print("calib", cal.shape, cal.min(), cal.max())
if os.environ.get("DRY"):
    sys.exit(0)
r = ClientRunner(har=src)
alls = f"""normalization1 = normalization([0.0], [255.0])
model_optimization_flavor(optimization_level={os.environ.get("OPT_LEVEL", "0")}, compression_level=0)
model_optimization_config(calibration, batch_size=8, calibset_size={n})
{extra}"""
print(alls)
r.load_model_script(alls)
t = time.time()
r.optimize(cal)
print(f"optimize took {time.time() - t:.0f} s")
r.save_har(dst)
print("saved", dst)

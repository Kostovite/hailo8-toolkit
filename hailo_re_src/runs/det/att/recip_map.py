"""Map the softmax reciprocal (reduce_sum_softmax1 + inv_pos) chip vs SDK_BIT_EXACT over many row sums, using the
attention sub-model E (ew_add9 -> ... -> ew_sub (exp) + reduce_sum). Row sum = sum of the CHIP's own exp outputs."""
import fcntl, sys
import numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
from hailo_sdk_client import ClientRunner, InferenceContext
d = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/runs/det/att"
z = np.load(f"{d}/ints2.npz"); base = z["best__ew_add9"].astype(np.int64)  # (16, 28, 28, 512)
rng = np.random.default_rng(5); F = []
for k in range(int(sys.argv[1]) if len(sys.argv) > 1 else 64):
    a, b = base[rng.integers(16)], base[rng.integers(16)]
    perm = rng.permutation(784); m = rng.random((28, 28, 1)) < rng.uniform(0.1, 0.9)
    x = np.where(m, a, b.reshape(784, 512)[perm].reshape(28, 28, 512)) * rng.uniform(0.7, 1.3)
    F.append(np.clip(np.round(x), 0, 255))
X = np.stack(F).astype(np.uint8)
hef = HEF(f"{d}/E.hef"); ii = hef.get_input_vstream_infos()[0]; q = ii.quant_info
outs = {o.name: o for o in hef.get_output_vstream_infos()}
r = ClientRunner(har=f"{d}/E_compiled.har")
with r.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
    e = r.infer(ctx, (X.astype(np.float32) - q.qp_zp) * q.qp_scale)
dev = "0000:03:00.0"; lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w")
fcntl.flock(lk, fcntl.LOCK_EX); p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
try: c = cv.run16(vd, hef, {ii.name: X})
finally: vd.release(); fcntl.flock(lk, fcntl.LOCK_UN)
names = list(outs); hn_order = [n for n in names]  # emulator returns outputs in HEF output order
E_ = dict(zip(names, e if isinstance(e, list) else [e]))
kexp = next(n for n in names if "ew_sub" in n); ksum = next(n for n in names if "reduce_sum" in n)
qo = outs[ksum].quant_info
S = np.asarray(c[kexp]).astype(np.int64).reshape(len(X), 784, 4, 784).sum(-1)
Cs = np.asarray(c[ksum]).astype(np.int64).reshape(len(X), 784, 4)
Es = np.clip(np.round(np.asarray(E_[ksum]) / qo.qp_scale + qo.qp_zp), 0, 255).astype(np.int64).reshape(Cs.shape)
qe = outs[kexp].quant_info
Ee = np.clip(np.round(np.asarray(E_[kexp]) / qe.qp_scale + qe.qp_zp), 0, 255).astype(np.int64).reshape(len(X), 784, 4, 784)
print("exp step chip == emu:", bool((np.asarray(c[kexp]).astype(np.int64).reshape(Ee.shape) == Ee).all()))
S, Cs, Es = S.ravel(), Cs.ravel(), Es.ravel()
print(f"rows {S.size}, sums {S.min()}..{S.max()}, mismatching rows {(Cs != Es).sum()}")
bad = np.unique(S[Cs != Es]); print("sums where chip != emu:", bad.tolist()[:60])
for v in sorted(set(bad.tolist()) | {32766, 32767, 32768, 32771, 32772}):
    m = S == v
    if m.any(): print(f"  sum {v}: rows {m.sum()}, wrong {(m & (Cs != Es)).sum()}, chip {sorted(set(Cs[m]))} emu {sorted(set(Es[m]))}")
np.savez(f"{d}/recip_map.npz", S=S, C=Cs, E=Es)

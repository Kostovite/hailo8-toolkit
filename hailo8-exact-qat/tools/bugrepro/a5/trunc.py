"""A5 hypothesis: the silicon truncates the 32-bit accumulator before the APU multiply. Test on the last layer's
captured APU input against the chip output."""
import sys
import numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import intops as I
d, lay = sys.argv[1], sys.argv[2]
r = np.load(f"{d}/cap/{lay}.npz"); c = next(iter(np.load(f"{d}/chip.npz").values())).astype(np.int64)
t = lambda k: torch.as_tensor(np.asarray(r[k], np.float64))
a = torch.as_tensor(np.transpose(r["act_in"], (0, 3, 1, 2)).astype(np.float64))
lo = lambda p: (int(r[p + "_bits"]), bool(r[p + "_signed"]), bool(r[p + "_wrap"]))
def apu(x, **kw):
    y = I.apu(x, t("thr"), t("mant"), t("shift"), t("offs"), int(r["final_shift"]), bool(r["mant_signed"]), lo("clip1"), lo("clip2"), lo("out"), **kw)
    return np.transpose(y.numpy(), (0, 2, 3, 1)).astype(np.int64).reshape(c.shape)
print("baseline mismatch", int((apu(a) != c).sum()), "emu act_out == ours:", bool((apu(a) == r["act_out"].astype(np.int64).reshape(c.shape)).all()))
for k in range(1, 14):
    af = torch.floor(a / 2 ** k) * 2 ** k
    at = torch.trunc(a / 2 ** k) * 2 ** k
    print(f"k={k:2d} floor: {int((apu(af) != c).sum()):5d}   trunc-to-zero: {int((apu(at) != c).sum()):5d}")

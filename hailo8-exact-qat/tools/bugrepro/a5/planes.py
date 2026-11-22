"""A5: 16-bit conv MAC built from 8x8 byte planes; which plane rounding reproduces the chip (QLinearConv probe)."""
import sys, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import intops as I
d = sys.argv[1]; c = next(iter(np.load(f"{d}/chip.npz").values())).astype(np.int64)
r1, r2 = np.load(f"{d}/cap/QLinearConv__conv1.npz"), np.load(f"{d}/cap/QLinearConv__conv2.npz")
print({k: r1[k].tolist() for k in r1.files if r1[k].size <= 4 and k in ("pre_acc_shift", "pad", "pad_value", "strides", "groups", "acc_bits")}, "k", r1["k"].shape)
x = torch.as_tensor(np.transpose(r1["conv_in"], (0, 3, 1, 2)).astype(np.float64)); k = torch.as_tensor(np.transpose(r1["k"], (3, 2, 0, 1)).astype(np.float64))
pv = float(r1["pad_value"]); pad = [int(v) for v in r1["pad"]]  # top, bottom, left, right
xp = F.pad(x, (pad[2], pad[3], pad[0], pad[1]), value=pv)
conv = lambda a, b: F.conv2d(a, b)
xh, xl = torch.div(xp, 256, rounding_mode="floor"), torch.remainder(xp, 256)
kh, kl = torch.div(k, 256, rounding_mode="floor"), torch.remainder(k, 256)
Shh, Shl, Slh, Sll = conv(xh, kh), conv(xh, kl), conv(xl, kh), conv(xl, kl)
acc = Shh * 65536 + (Shl + Slh) * 256 + Sll
ref = torch.as_tensor(np.transpose(r1["conv_out"], (0, 3, 1, 2)).astype(np.float64))
print("plane recombination == captured acc:", bool((I.wrap(acc, 32, True) == ref).all()))
lo = lambda r, p: (int(r[p + "_bits"]), bool(r[p + "_signed"]), bool(r[p + "_wrap"]))
def tail(a):
    t = lambda r, kk: torch.as_tensor(np.asarray(r[kk], np.float64))
    a = I.hw_lossy(I.wrap(a, 32, True) + t(r1, "bias")[None, :, None, None], *lo(r1, "bias"))
    y1 = I.apu(a, t(r1, "thr"), t(r1, "mant"), t(r1, "shift"), t(r1, "offs"), int(r1["final_shift"]), bool(r1["mant_signed"]), lo(r1, "clip1"), lo(r1, "clip2"), lo(r1, "out"))
    acc2 = I.mac_conv(y1, torch.as_tensor(np.transpose(r2["k"], (3, 2, 0, 1)).astype(np.float64)), int(r2["pre_acc_shift"]), (1, 1), (1, 1), [0, 0, 0, 0], float(r2["pad_value"]), groups=int(r2["groups"]))
    a2 = I.hw_lossy(I.hw_lossy(acc2, int(r2["acc_bits"]), bool(r2["acc_signed"]), bool(r2["acc_wrap"])) + torch.as_tensor(np.asarray(r2["bias"], np.float64))[None, :, None, None], *lo(r2, "bias"))
    y2 = I.apu(a2, t(r2, "thr"), t(r2, "mant"), t(r2, "shift"), t(r2, "offs"), int(r2["final_shift"]), bool(r2["mant_signed"]), lo(r2, "clip1"), lo(r2, "clip2"), lo(r2, "out"))
    return np.transpose(y2.numpy(), (0, 2, 3, 1)).astype(np.int64).reshape(c.shape)
print("ours (exact acc):", int((tail(ref) != c).sum()))
for s in (2, 4, 6, 8):
    for nm, f in (("round", torch.round), ("floor", torch.floor)):
        print(f"Sll {nm} >> {s}: {int((tail(Shh * 65536 + (Shl + Slh) * 256 + f(Sll / 2 ** s) * 2 ** s) != c).sum())}   "
              f"(Shl+Slh)*256+Sll {nm} >> {s}: {int((tail(Shh * 65536 + f(((Shl + Slh) * 256 + Sll) / 2 ** s) * 2 ** s) != c).sum())}")
print("drop Sll:", int((tail(Shh * 65536 + (Shl + Slh) * 256) != c).sum()))

"""A5: which APU rounding reproduces the chip on the 16-bit probes (graph26, cont=False)."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import graph26 as G, intops as I
d = sys.argv[1]; hn = json.load(open(f"{d}/m.hn")); x = np.transpose(np.load(f"{d}/x.npy"), (0, 2, 3, 1)).astype(np.float64)
c = next(iter(np.load(f"{d}/chip.npz").values())).astype(np.int64)
orig_apu, orig_rs = I.apu, I.apu_round_shift
def run():
    with torch.no_grad(): out, _ = G.run(hn, G.Params(f"{d}/cap", "cpu"), x, "cpu", check=False)
    lay = [n for n in out if not n.endswith("input_layer1")][-1]
    y = np.transpose(out[lay].numpy(), (0, 2, 3, 1)).astype(np.int64).reshape(c.shape); return int((y != c).sum())
for rb in (6, 7, 8, 10, 12, 16, 24, 40):
    I.apu = lambda *a, _rb=rb, **k: orig_apu(*a, **{**k, "post_shift1_rounding": _rb}); print("post_shift1_rounding", rb, "mismatch", run())
I.apu = orig_apu
def rs_halfup(v, shift, max_round_bits, signed):  # round half away from zero / half up, no truncation
    if signed:
        return torch.div(v + torch.bitwise_left_shift(torch.ones_like(shift), shift) // 2, torch.bitwise_left_shift(torch.ones_like(shift), shift), rounding_mode="floor")
    s = torch.sign(v); a = v.abs()
    return s * torch.div(a + torch.bitwise_left_shift(torch.ones_like(shift), shift) // 2, torch.bitwise_left_shift(torch.ones_like(shift), shift), rounding_mode="floor")
I.apu_round_shift = rs_halfup; print("half-up rounding everywhere: mismatch", run())

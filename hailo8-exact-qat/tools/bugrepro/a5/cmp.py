"""Exact executor vs chip, step 2 (torch env): graph26 from raw pixels (cont=False) vs the chip, per output.
Usage: python cmp.py OUT_DIR"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import graph26 as G
d = sys.argv[1]; hn = json.load(open(f"{d}/m.hn")); x = np.transpose(np.load(f"{d}/x.npy"), (0, 2, 3, 1)).astype(np.float64)
P = G.Params(f"{d}/cap", "cpu")
with torch.no_grad(): out, rep = G.run(hn, P, x, "cpu", cont=True)
print("vs HSim:", sum(b == 0 for _, _, b, _ in rep), "/", len(rep), "layers exact;", [r for r in rep if r[2] != 0][:4])
with torch.no_grad(): out, _ = G.run(hn, G.Params(f"{d}/cap", "cpu"), x, "cpu", check=False)
for k, v in np.load(f"{d}/chip.npz").items():
    name = k.replace("|", "/"); src = hn["layers"][name]["input"][0] if hn["layers"][name]["type"] == "output_layer" else name
    o = next((n for n in hn["layers"] if hn["layers"][n]["type"] == "output_layer" and name.split("/")[-1] in hn["layers"][n]["input"][0]), None)
    lay = src if src in out else next(n for n in out if n.endswith(name.split("/")[-1]))
    y = np.transpose(out[lay].numpy(), (0, 2, 3, 1)).astype(np.int64).reshape(v.shape); c = v.astype(np.int64)
    dd = y - c; np.save(f"{d}/diff.npy", dd); np.save(f"{d}/ours.npy", y)
    print(f"{name}: ours vs chip mismatch {int((dd != 0).sum())} of {dd.size}, values {sorted(set(dd[dd != 0].tolist()))[:10]}")

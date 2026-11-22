"""usage: grid_e2e.py dump N OUTDIR | eval OUTDIR MODE(torch|emu)
GridNet end-to-end on the VN thesis bench: emu mode swaps net.trunk (shared with TableMergerV2) for Hailo-emulated p2/p4."""
import sys, os, json, numpy as np, torch
W = "/mnt/storage/lib_project/models/Dolphin/wordlid"; sys.path.insert(0, W); sys.path.insert(0, W + "/third_party")
from PIL import Image
from gridnet_full import TableStructure
import grits as GR
CK = W + "/data/grid/table_structure_final.pt"; BD = W + "/data/vn_bench"
m = TableStructure.load(CK); B = json.load(open(BD + "/bench.json"))
cmd, a1 = sys.argv[1], sys.argv[2]
def gray(rec): return np.asarray(Image.open(os.path.join(BD, "img", rec["id"] + ".png")).convert("L"))
if cmd == "dump":
    n, out = int(a1), sys.argv[3]; os.makedirs(out, exist_ok=True)
    ix = np.sort(np.random.default_rng(1121).choice(len(B), n, replace=False))
    cv = np.lib.format.open_memmap(out + "/canvas.npy", "w+", np.uint8, (n, 896, 896, 1))
    for j, i in enumerate(ix): cv[j, ..., 0] = m._canvas(gray(B[i]), m.norm_text)[0]
    cv.flush(); np.save(out + "/ix.npy", ix); print("dumped", n); sys.exit()
out, mode = a1, sys.argv[3]; ix = np.load(out + "/ix.npy"); cvs = np.load(out + "/canvas.npy", mmap_mode="r")
if mode == "emu":
    F = [np.load(out + f"/q_{k}.npy", mmap_mode="r") for k in (0, 1)]; cur = [0]
    def fake(x):
        j = cur[0]; assert (x[0, 0].mul(255).round().byte().cpu().numpy() == cvs[j, ..., 0]).all(), "canvas mismatch"
        return tuple(torch.from_numpy(np.asarray(f[j], np.float32)).permute(2, 0, 1)[None].to(x.device, x.dtype) for f in F)
    m.net.trunk.forward = fake
def cells(cs, rows, cols):
    return [{"row_nums": list(range(r, r + h)), "column_nums": list(range(c, c + w)), "bbox": [cols[c], rows[r], cols[c + w], rows[r + h]]} for r, c, h, w in cs]
ok, top, recs = [], [], []
for j, i in enumerate(ix):
    if mode == "emu": cur[0] = j
    rec = B[i]; t = m.table(gray(rec)); gt = {tuple(c) for c in rec["cells"]}
    got = {(c["r"], c["c"], c["rowspan"], c["colspan"]) for c in t["cells"]}
    e = len(t["rows"]) == len(rec["rows"]) and len(t["cols"]) == len(rec["cols"]) and got == gt; ok.append(e)
    P = cells(sorted(got), t["rows"], t["cols"]) if len(t["rows"]) > 1 and len(t["cols"]) > 1 and got else []
    G = cells(sorted(gt), rec["rows"], rec["cols"])
    top.append(GR.grits_top(np.array(GR.cells_to_relspan_grid(G)), np.array(GR.cells_to_relspan_grid(P)))[0] if P else 0.0)
    recs.append({"id": rec["id"], "nr": len(t["rows"]), "nc": len(t["cols"]), "cells": sorted(got), "exact": bool(e)})
json.dump(recs, open(f"{out}/pred_{mode}.json", "w"))
print(f"RESULT {mode} n {len(ix)} exact {np.mean(ok):.4f} GriTS-Top {100*np.mean(top):.2f}")

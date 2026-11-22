"""grid_gate.py CKPT OUT_DIR PREFIX|torch : GridNet end-to-end on the eval canvases; PREFIX_0/1 (p2/p4 NHWC) replace net.trunk
(shared with TableMergerV2). PASS: identical structure to torch on >= MIN_SAME (.99) of tables and exact rate >= torch - .005."""
import sys, os, json, numpy as np, torch
W = "/mnt/storage/lib_project/models/Dolphin/wordlid"; sys.path[:0] = [W, W + "/third_party"]
from PIL import Image
from gridnet_full import TableStructure
import grits as GR
ck, out, pre = sys.argv[1:4]; BD = W + "/data/vn_bench"; B = json.load(open(BD + "/bench.json"))
m = TableStructure.load(ck); ix = np.load(f"{out}/host/bench_ix.npy"); cvs = np.load(f"{out}/eval_x.npy", mmap_mode="r"); cur = [0]
if pre != "torch":
    Fe = [np.load(f"{out}/{pre}_{k}.npy", mmap_mode="r") for k in (0, 1)]
    def fake(x):
        j = cur[0]; assert (x[0, 0].mul(255).round().byte().cpu().numpy() == cvs[j, ..., 0]).all(), "canvas mismatch"
        return tuple(torch.from_numpy(np.asarray(f[j], np.float32)).permute(2, 0, 1)[None].to(x.device, x.dtype) for f in Fe)
    m.net.trunk.forward = fake
def cells(cs, rows, cols): return [{"row_nums": list(range(r, r + h)), "column_nums": list(range(c, c + w)), "bbox": [cols[c], rows[r], cols[c + w], rows[r + h]]} for r, c, h, w in cs]
recs, top = [], []
for j, i in enumerate(ix):
    cur[0] = j; rec = B[i]; t = m.table(np.asarray(Image.open(f"{BD}/img/{rec['id']}.png").convert("L")))
    gt = {tuple(c) for c in rec["cells"]}; got = sorted({(c["r"], c["c"], c["rowspan"], c["colspan"]) for c in t["cells"]})
    e = len(t["rows"]) == len(rec["rows"]) and len(t["cols"]) == len(rec["cols"]) and set(got) == gt
    P = cells(got, t["rows"], t["cols"]) if len(t["rows"]) > 1 and len(t["cols"]) > 1 and got else []
    top.append(GR.grits_top(np.array(GR.cells_to_relspan_grid(cells(sorted(gt), rec["rows"], rec["cols"]))), np.array(GR.cells_to_relspan_grid(P)))[0] if P else 0.0)
    recs.append({"nr": len(t["rows"]), "nc": len(t["cols"]), "cells": [list(c) for c in got], "exact": bool(e)})
ex = float(np.mean([r["exact"] for r in recs])); json.dump(recs, open(f"{out}/e2e_{pre}.json", "w"))
print(f"RESULT {pre} n {len(ix)} exact {ex:.4f} GriTS-Top {100 * np.mean(top):.2f}")
if pre == "torch": sys.exit(0)
T = json.load(open(f"{out}/e2e_torch.json")); same = np.mean([a["nr"] == b["nr"] and a["nc"] == b["nc"] and a["cells"] == b["cells"] for a, b in zip(T, recs)])
tex = float(np.mean([r["exact"] for r in T])); ok = same >= float(os.environ.get("MIN_SAME", ".99")) and ex >= tex - .005
print(f"RESULT identical_structure_vs_torch {same:.4f}  exact torch {tex:.4f} -> {ex:.4f}"); print("GATE", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 2)

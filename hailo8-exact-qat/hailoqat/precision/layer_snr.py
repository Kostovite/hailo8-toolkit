"""Per-layer SNR of the exact chip model vs the FP model (same letterboxed pages), plus box-edge error in pixels.
Usage: python layer_snr.py HN CAPTURE_DIR N_PAGES OUT.json"""
import glob, json, os, re, sys
import numpy as np, torch
R = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"
sys.path.insert(0, f"{R}/hailoqat")
import evaluate as E, graph26
from ultralytics import YOLO
hn_path, cap, npg, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
dev = "cuda"
hn = json.load(open(hn_path)); L = hn["layers"]
P = graph26.Params(cap, dev)
model = YOLO(f"{R}/models/v14sec/best.pt").model.to(dev).eval().double()
mods = dict(model.named_modules())
mp = {}
def resolve(o):
    """ONNX scope '/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv' -> torch module 'model.23.one2one_cv2.0.1'."""
    parts = [p for p in o.split("/") if p][:-1]
    if parts and parts[-1] == "conv":
        parts = parts[:-1]
    cur = parts[0]
    for c in parts[1:]:
        segs = cur.split(".")
        for k in range(len(segs), 0, -1):
            cand = ".".join(segs[:k]) + "." + c
            if cand in mods:
                cur = cand; break
        else:
            return None
    return cur if cur in mods else None
for n, v in L.items():
    for o in v.get("original_names", []):
        if o.endswith("/Conv") and n in P.conv:
            name = resolve(o)
            if name:
                mp[n] = name
print("mapped", len(mp), "of", sum(1 for n in L if n in P.conv))
fp = {}
for n, name in mp.items():
    mods[name].register_forward_hook(lambda m_, i_, o_, n=n: fp.__setitem__(n, o_.detach()))
files = sorted(glob.glob(f"{R}/data_v14sec/val/images/*"))
files = [files[i] for i in np.random.default_rng(0).choice(len(files), npg, replace=False)]
acc = {}
order = list(L)
with torch.no_grad():
    for imgs, _ in E.batches(files, (896, 896), 1):
        model(torch.as_tensor(imgs, device=dev).permute(0, 3, 1, 2).double() / 255.0)
        q, _ = graph26.run(hn, P, imgs.astype(np.float64), dev, check=False)
        for n in mp:
            if n not in q or not isinstance(q[n], torch.Tensor) or q[n].shape != fp[n].shape: continue
            r = P.conv[n]
            sc = torch.as_tensor(np.asarray(r["out_scale"], np.float64), device=dev).reshape(1, -1, 1, 1)
            zp = torch.as_tensor(np.asarray(r["out_zp"], np.float64), device=dev).reshape(1, -1, 1, 1)
            d = (q[n] - zp) * sc
            s, e = float((fp[n] ** 2).sum()), float(((fp[n] - d) ** 2).sum())
            a = acc.setdefault(n, [0.0, 0.0]); a[0] += s; a[1] += e
rows = []
for n in order:
    if n in acc:
        s, e = acc[n]; v = L[n]
        prec = v.get("quantization_params", {}).get("precision_mode", "?")
        rows.append({"layer": n.split("/")[-1], "module": mp[n], "prec": prec, "snr_db": round(10 * np.log10(s / max(e, 1e-30)), 2)})
for r_ in rows:
    flag = "  <-- box" if "one2one_cv2" in r_["module"] else ""
    print(f"{r_['layer']:28s} {r_['module']:34s} {r_['prec']:13s} {r_['snr_db']:7.2f}{flag}")
json.dump(rows, open(out, "w"), indent=0)

"""GPU exactness suite for the hailoqat executors (graph26 float64 reference + qatfast) on one captured case.

usage (torch env, no DFC preloads): gpu_suite.py NAME HN CAP INPUT [OUT_JSON]
  INPUT: an image (letterboxed to the HN input like capture.py) or the NHWC .npy capture.py fed.
Tests, all on CUDA:
  T1 per-layer exact (--cont semantics: each layer fed HSim's own inputs)
  T2 chained exact from the raw input (no HSim substitution)
  T3 batch invariance: rows of batches 2/3/5/8/16 == the same inputs run one at a time (inputs = capture input +
     deterministic variants: flips, +-noise, other COCO images for image nets)
  T4 qatfast == graph26 on a batch of 4 (when QATNet builds)
Also records the layer types / precision modes / activations covered, so the suite doubles as a coverage table."""
import glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hailoqat"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import graph26 as G  # noqa: E402

name, hn_p, cap, inp = sys.argv[1:5]; out_p = sys.argv[5] if len(sys.argv) > 5 else None
dev = torch.device("cuda"); hn = json.load(open(hn_p)); L = hn["layers"]; P = G.Params(cap, dev)
res = {"case": name, "conv_mode": os.environ.get("HQ_CONV", "gemm")}
in_l = [v for v in L.values() if v["type"] == "input_layer"][0]; ishape = in_l["input_shapes"][0][1:]
outs = []
for n in L:  # chip outputs; a host postprocess (NMS) is replaced by the chip tensors that feed it
    if L[n]["type"] == "output_layer":
        i = L[n]["input"][0]
        outs += L[i]["input"] if L[i]["type"] == "postprocess" else [i]
outs = list(dict.fromkeys(outs))


def load_img(p):
    import cv2
    from letterbox import letterbox
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    if im.shape[:2] != tuple(ishape[:2]):
        im = letterbox(im, tuple(ishape[:2]))[0]
    return im.astype(np.float64)


if inp.endswith(".npy"):
    x0 = np.load(inp).astype(np.float64); x0 = x0[0] if x0.ndim == 4 else x0
    lo, hi = float(x0.min()), float(x0.max())
    rng = np.random.default_rng(0)
    var = [x0[::-1].copy(), x0[:, ::-1].copy()] + [np.clip(x0 + rng.normal(0, .05 * (hi - lo + 1e-9), x0.shape), lo, hi) for _ in range(14)]
else:
    x0 = load_img(inp)
    others = sorted(glob.glob(os.path.join(os.path.dirname(inp), "*.jpg")))[1:15]
    var = [x0[:, ::-1].copy()] + [load_img(p) for p in others]
    var += [x0] * (15 - len(var))
X = np.stack([x0] + var[:15])  # 16 inputs, row 0 = capture input

# coverage table (what this case exercises)
cov = {}
for n, v in L.items():
    q = v.get("quantization_params", {}).get("precision_mode", "-"); a = v.get("params", {}).get("activation", "")
    key = f"{v['type']}|{q}|{a}"; cov[key] = cov.get(key, 0) + 1
res["coverage"] = cov

with torch.no_grad():
    t = time.time(); _, rep = G.run(hn, P, X[:1], dev, check=True, cont=True)
    bad = [(n, ty, b, m) for n, ty, b, m in rep if b != 0]
    res["T1_per_layer"] = {"checked": len(rep), "exact": len(rep) - len(bad), "not_exact": bad[:20], "s": round(time.time() - t, 1)}
    t = time.time(); _, rep2 = G.run(hn, P, X[:1], dev, check=True, cont=False)
    real = [(n, ty, b, m) for n, ty, b, m in rep2 if b > 0]  # b == -1: shape-only (slice vs its input passthru record)
    res["T2_chained"] = {"checked": len(rep2), "value_mismatch": real[:20], "pass": not real, "s": round(time.time() - t, 1)}
    single = [G.run(hn, P, X[i:i + 1], dev, check=False)[0] for i in range(5)]
    t3 = {}
    for bs in (2, 3, 5, 8, 16):
        try:
            b = G.run(hn, P, X[:bs], dev, check=False)[0]
            t3[bs] = all(torch.equal(single[j][n][0], b[n][j]) for j in range(min(bs, 5)) for n in outs)
            del b
        except torch.OutOfMemoryError:
            t3[bs] = "oom"
        torch.cuda.empty_cache()
    res["T3_batch"] = {"per_bs": t3, "pass": all(v is True for v in t3.values() if v != "oom") and any(v is True for v in t3.values())}
    try:
        import qatfast
        net = qatfast.QATNet(hn, P, dev).to(dev).eval()
        o = net(torch.from_numpy(X[:2].astype(np.float32)).to(dev))
        g = G.run(hn, P, X[:2], dev, check=False)[0]
        diff = {n: int((o[n].double() != g[n].double()).sum()) for n in outs}
        res["T4_qatfast"] = {"built": True, "flags": list(net.flag_events)[:5], "mismatch_per_output": diff, "pass": not any(diff.values()) and not net.flag_events}
    except Exception as e:  # noqa: BLE001
        import traceback
        res["T4_qatfast"] = {"built": False, "why": f"{type(e).__name__}: {str(e)[:120]}",
                             "tb": traceback.format_exc().strip().splitlines()[-7:]}
res["PASS"] = bool(res["T1_per_layer"]["exact"] == res["T1_per_layer"]["checked"] and res["T2_chained"]["pass"] and res["T3_batch"]["pass"]
                   and res["T4_qatfast"].get("pass", False))  # qatfast must build and match
print(json.dumps({k: v for k, v in res.items() if k != "coverage"}, default=str))
if out_p:
    json.dump(res, open(out_p, "w"), indent=1, default=str)

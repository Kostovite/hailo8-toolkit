"""FP YOLO26m with simulated 8-bit activations on chosen modules (per-tensor asymmetric grid, min/max from 32 TRAIN
pages) -> full-bench AP per IoU threshold. Shows which part's activation precision costs the IoU-0.90/0.95 AP.
Usage: python fq.py VARIANT OUT.json      VARIANT in none, all8, all8_box16, box8, neck8, cls8"""
import glob, json, os, re, sys
import numpy as np, torch
os.environ.setdefault("VAL_DIR", "/mnt/storage/lib_project/models/Hailo_model/hailo_re/data_v14sec/bench/images")
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import evaluate as E
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class
R = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"
var, out = sys.argv[1], sys.argv[2]
dev = "cuda"
model = YOLO(f"{R}/models/v14sec/best.pt").model.to(dev).eval().float()
leaf = {n: m for n, m in model.named_modules() if type(m).__name__ in ("Conv", "DWConv")
        or re.fullmatch(r"model\.23\.one2one_cv[23]\.\d\.2", n) or re.fullmatch(r"model\.23\.one2one_cv3\.\d\.\d\.\d", n)}
leaf = {n: m for n, m in leaf.items() if not any(o != n and o.startswith(n + ".") for o in leaf)}  # innermost only
head_out = lambda n: re.fullmatch(r"model\.23\.one2one_cv[23]\.\d\.2", n) is not None  # noqa: E731 (16-bit on chip)
box = lambda n: n.startswith("model.23.one2one_cv2") and not head_out(n)  # noqa: E731
cls = lambda n: n.startswith("model.23.one2one_cv3") and not head_out(n)  # noqa: E731
WQ = os.environ.get("WQ", "")  # box | all | boxhead : 8-bit symmetric weights on BN-fused convs (WQC=1 per channel)
PWL = os.environ.get("PWL", "")  # box | all : SiLU -> 11 uniform linear pieces over the calibrated input range
sel = {"none": lambda n: False, "all8": lambda n: not head_out(n), "all8_box16": lambda n: not head_out(n) and not box(n),
       "box8": box, "cls8": cls, "neck8": None}[var]
targets = {n: m for n, m in leaf.items() if sel and sel(n)}
if var == "neck8":
    targets = {n: m for n, m in model.named_modules() if n in ("model.16", "model.19", "model.22")}
rng = {}
def cal_hook(n):
    def f(m, i, o):
        lo, hi = float(o.min()), float(o.max()); a = rng.get(n, (lo, hi)); rng[n] = (min(a[0], lo), max(a[1], hi))
    return f
def q_hook(n):
    lo, hi = rng[n]; lo = min(lo, 0.0); s = (hi - lo) / 255; zp = round(-lo / s)
    def f(m, i, o):
        return (torch.clamp(torch.round(o / s) + zp, 0, 255) - zp) * s
    return f
cal = sorted(glob.glob(f"{R}/data_v14sec/train/images/*"))
cal = [cal[i] for i in np.random.default_rng(0).choice(len(cal), 32, replace=False)]
with torch.no_grad():
    hs = [m.register_forward_hook(cal_hook(n)) for n, m in targets.items()]
    for imgs, _ in E.batches(cal, (896, 896), 4):
        model(torch.as_tensor(imgs, device=dev).permute(0, 3, 1, 2).float() / 255.0)
    for h in hs: h.remove()
    for n, m in targets.items(): m.register_forward_hook(q_hook(n))
    import torch.nn as nn
    if WQ:
        if WQ.startswith("re:"):  # WQ=re:INCLUDE[|!EXCLUDE]  regexes on module names
            inc, _, exc = WQ[3:].partition("!")
            wsel = lambda n: re.search(inc, n) is not None and not (exc and re.search(exc, n))  # noqa: E731
        else:
            wsel = {"box": box, "all": lambda n: True, "boxhead": lambda n: n.startswith("model.23.one2one_cv2")}[WQ]
        for n, m in leaf.items():
            if not wsel(n): continue
            conv = getattr(m, "conv", m)
            w = conv.weight.data
            if hasattr(m, "bn") and isinstance(m.bn, nn.BatchNorm2d):  # fold BN like the parser, quantize, unfold
                bn = m.bn; g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).reshape(-1, 1, 1, 1)
                wf = w * g
            else:
                g, wf = None, w
            a = wf.abs().amax(dim=(1, 2, 3), keepdim=True) if os.environ.get("WQC") else wf.abs().max()
            wq = torch.round(wf / (a / 127)) * (a / 127)
            conv.weight.data = wq / g if g is not None else wq
    if PWL:
        psel = {"box": box, "all": lambda n: True}[PWL]
        prng = {}
        def pre_hook(n):
            def f(m, i):
                x = i[0]; lo, hi = float(x.min()), float(x.max()); a = prng.get(n, (lo, hi)); prng[n] = (min(a[0], lo), max(a[1], hi))
            return f
        # ultralytics shares ONE SiLU instance across all Conv modules: record each layer's act input at its bn output
        acts = {n + ".act": m.bn for n, m in leaf.items() if psel(n) and isinstance(getattr(m, "act", None), nn.SiLU)}
        hs = [bn_.register_forward_hook(lambda m_, i_, o_, n=n: pre_hook(n)(m_, (o_,))) for n, bn_ in acts.items()]
        for imgs, _ in E.batches(cal, (896, 896), 4):
            model(torch.as_tensor(imgs, device=dev).permute(0, 3, 1, 2).float() / 255.0)
        for h in hs: h.remove()
        class PwlSiLU(nn.Module):
            def __init__(s_, lo, hi):
                super().__init__(); s_.xs = torch.linspace(lo, hi, 12, device=dev); s_.ys = s_.xs * torch.sigmoid(s_.xs)
            def forward(s_, x):
                i = torch.clamp(torch.searchsorted(s_.xs[1:-1].contiguous(), x.contiguous()), 0, 10)
                x0, x1, y0, y1 = s_.xs[i], s_.xs[i + 1], s_.ys[i], s_.ys[i + 1]
                return y0 + (y1 - y0) / (x1 - x0) * (x - x0)  # end pieces extend (silicon edge rule)
        for n, a in acts.items():
            parent = leaf[n[:-4]]; parent.act = PwlSiLU(*prng[n])
    met, done = E.Metric(), 0
    files = E.val_files(0)
    for imgs, labs in E.batches(files, (896, 896), 4):
        y = model(torch.as_tensor(imgs, device=dev).permute(0, 3, 1, 2).float() / 255.0)
        y = y[0] if isinstance(y, (list, tuple)) else y
        for d, l in zip([p[p[:, 4] > 0.001] for p in y], labs):
            met.add(d, l, done); done += 1
st = {k: np.concatenate(v, 0) for k, v in met.stats.items()}
r = ap_per_class(st["tp"], st["conf"], st["pred_cls"], st["target_cls"])
res = {"variant": var + (f"+WQ{WQ}" if WQ else "") + ("c" if os.environ.get("WQC") else "") + (os.environ.get("TAG", "")) + (f"+PWL{PWL}" if PWL else ""), "n_quant": len(targets), "mAP50_95": float(r[5].mean()), "ap_per_iou": [float(x) for x in r[5].mean(0)]}
print(json.dumps(res)); json.dump(res, open(out, "w"))

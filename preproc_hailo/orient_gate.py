"""orient2 FP-vs-chip gate on held-out pages (runs/orient/val_items.json, 4 variants each = train_orient.ValPages).
  build : venv_hf  -> eval/orient_eval.npz (canvas uint8, k, photo, nh, nw, FP served decision on the variable-shape page)
  score FEAT.npy [FEAT2.npy ...] : host head (valid windows, 7x7 mean, fc, softmax, decide_ll) on chip/emulator features"""
import sys, os, json, numpy as np, cv2, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/storage/lib_project/models/layout_teacher/train"); os.chdir("/mnt/storage/lib_project/models/layout_teacher/train")
import train_orient as T
OUT = "/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/eval"; os.makedirs(OUT, exist_ok=True)
net = T.OriNet(); net.load_state_dict(torch.load("runs/orient2/ori_5000.pt", map_location="cpu")); net.eval()
VAL_N = int(os.environ.get("VAL_N", 50))

def head(f, nh, nw):                                   # f (28,28,256) HWC
    f = torch.from_numpy(np.ascontiguousarray(f[:nh // 32, :nw // 32])).permute(2, 0, 1)[None].float()
    p = F.conv2d(F.avg_pool2d(f, 7, 1), net.fc.weight[:, :, None, None], net.fc.bias).softmax(1)[0]
    return p.detach().flatten(1).t().numpy()

if sys.argv[1] == "build":
    sets = {k: [tuple(x) for x in v[:VAL_N]] for k, v in json.load(open("runs/orient/val_items.json")).items()}
    C, K, PH, NH, NW, FPD, FPC, DOM = [], [], [], [], [], [], [], []
    for dom, items in sets.items():
        ds = T.ValPages(items)
        for j in range(len(ds)):
            img, k, photo = ds[j]; img = np.ascontiguousarray(img)
            h, w = img.shape[:2]; s = 896 / max(h, w)
            nw, nh = max(224, int(round(w * s / 32)) * 32), max(224, int(round(h * s / 32)) * 32)
            x = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            c = np.full((896, 896, 3), 128, np.uint8); c[:min(nh, 896), :min(nw, 896)] = x[:896, :896]
            fpd = T.decide_ll(T.page_probs(net, img))[0]                     # served FP (variable shape)
            with torch.no_grad():
                f = net.body(net.stem(torch.from_numpy(c).permute(2, 0, 1)[None].float() / 255 - .5))[0].permute(1, 2, 0).numpy()
            C.append(c); K.append(k); PH.append(photo); NH.append(nh); NW.append(nw); FPD.append(fpd); DOM.append(dom)
            FPC.append(T.decide_ll(head(f, nh, nw))[0])
        print(dom, len(C), flush=True)
    np.save(f"{OUT}/orient_canvas.npy", np.stack(C))
    np.savez(f"{OUT}/orient_eval.npz", canvas=np.stack(C), k=K, photo=PH, nh=NH, nw=NW, fp_served=FPD, fp_canvas=FPC, dom=DOM)
else:
    d = np.load(f"{OUT}/orient_eval.npz"); k = d["k"]; ref = d["fp_served"]
    def rep(name, dec):
        dec = np.asarray(dec); up = k == 0
        print(f"{name:28s} agree-with-FP {np.mean(dec == ref)*100:6.2f}%  ({int((dec != ref).sum())} pages differ)  "
              f"false turns {int((dec[up] != 0).sum())}/{up.sum()}  right turns {int((dec[~up] == k[~up]).sum())}/{(~up).sum()}  wrong angle {int(((dec[~up] != 0) & (dec[~up] != k[~up])).sum())}")
    rep("FP served (variable shape)", ref); rep("FP on 896 canvas (torch)", d["fp_canvas"])
    for fp in sys.argv[2:]:
        f = np.load(fp); rep(os.path.basename(fp), [T.decide_ll(head(f[i], d["nh"][i], d["nw"][i]))[0] for i in range(len(f))])

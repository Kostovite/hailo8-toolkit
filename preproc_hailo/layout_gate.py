"""layout (PP-DocLayoutV3 r3b/r5) FP-vs-chip gate on train_ppdl3.val_sets() thesis / d4la / photo_thesis / photo_d4la
(flat2 straightening as served).  The host part = the full HF model after the chip cut (decoder_input_proj outputs + mask_feat).
  build CKPT        -> eval/layout_x.npy (uint8 896 letterbox) + layout_gt.pkl + FP AP
  score CKPT PREFIX -> AP from chip/emulator outputs PREFIX_{0..3}.npy (0 = s8 src, 1 = mask_feat, 2 = s16, 3 = s32 by size)"""
import sys, os, pickle, numpy as np, torch, torch.nn as nn
os.environ.setdefault("LAYOUT_FLAT_CKPT", "/mnt/storage/lib_project/models/layout_teacher/models_flat/flat2.pt")
sys.path.insert(0, "/mnt/storage/lib_project/models/layout_teacher/train"); os.chdir("/mnt/storage/lib_project/models/layout_teacher/train")
import train_ppdl3 as TP
E = "/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/eval"; SETS = ("thesis", "d4la", "photo_thesis", "photo_d4la")
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = TP.build_model(); m.load_state_dict(torch.load(sys.argv[2], map_location="cpu")); m.eval().to(dev)

def ap_of(pred_fn, x_or_feats, gts, names):
    out = {}; i0 = 0
    for name in SETS:
        sel = [j for j, n in enumerate(names) if n == name]; G, D = [], []
        for a in range(0, len(sel), 8):
            idx = sel[a:a + 8]
            with torch.no_grad():
                s, c, b = pred_fn(idx)
            for j, k in enumerate(idx):
                G.append(gts[k]); iid = len(G); keep = s[j] > .01
                for sc, kk, q in zip(s[j][keep].tolist(), c[j][keep].tolist(), b[j][keep].tolist()):
                    D.append({"image_id": iid, "category_id": kk + 1, "bbox": [q[0] * 1000, q[1] * 1000, (q[2] - q[0]) * 1000, (q[3] - q[1]) * 1000], "score": sc})
        a_, per = TP.coco_ap(G, D); out[name] = (round(a_, 4), round(per["_box"], 4), round(per["_box75"], 4))
    return out

if sys.argv[1] == "build":
    sets = TP.val_sets(); X, GT, NM = [], [], []
    for name in SETS:
        items, warp = sets[name]; ds = TP.ValData(items, warp)
        for j in range(len(ds)):
            x, b, c = ds[j]; X.append((x.permute(1, 2, 0).numpy() * 255 + .5).astype(np.uint8)); GT.append((b, c)); NM.append(name)
        print(name, len(X), flush=True)
    X = np.stack(X); np.save(f"{E}/layout_x.npy", X); pickle.dump((GT, NM), open(f"{E}/layout_gt.pkl", "wb"))
    fn = lambda idx: TP.predict(m, torch.from_numpy(X[idx]).permute(0, 3, 1, 2).float().to(dev) / 255)
    print("FP (AP, box AP, box AP75):", ap_of(fn, X, GT, NM))
else:
    GT, NM = pickle.load(open(f"{E}/layout_gt.pkl", "rb")); P_ = sys.argv[3]
    F = [np.load(f"{P_}_{i}.npy", mmap_mode="r") for i in range(4)]       # by size: s8 (112x112x256), mask (224x224x32), s16, s32
    core = m.model
    class Stub(nn.Module):
        def forward(s, x, mask): return [None, (x, None)]                  # backbone skipped; x4_feat unused with encoder_outputs
    core.backbone = Stub(); core.encoder_input_proj = nn.ModuleList([nn.Identity()]); core.decoder_input_proj = nn.ModuleList([nn.Identity()] * 3)
    def fn(idx):
        t = lambda a: torch.from_numpy(np.asarray(a[idx], np.float32)).permute(0, 3, 1, 2).to(dev)
        srcs, mask = [t(F[0]), t(F[2]), t(F[3])], t(F[1]); B = len(idx)
        o = m(pixel_values=torch.zeros(B, 3, 896, 896, device=dev), encoder_outputs=(srcs, mask))
        prob = o.logits.sigmoid(); _, Q, C = prob.shape; s, i = prob.flatten(1).topk(300, 1); q = i // C; c = i % C
        b = o.pred_boxes.gather(1, q[..., None].expand(-1, -1, 4))
        return s, c, torch.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)
    print(os.path.basename(P_), "(AP, box AP, box AP75):", ap_of(fn, None, GT, NM))

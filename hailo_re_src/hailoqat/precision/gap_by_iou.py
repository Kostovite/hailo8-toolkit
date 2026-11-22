"""Where the FP -> chip mAP50-95 gap lives: per class and per author on the full bench (saved match stats)."""
import glob, json, os, sys
import numpy as np
os.environ.setdefault("VAL_DIR", "/mnt/storage/lib_project/models/Hailo_model/hailo_re/data_v14sec/bench/images")
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import evaluate as E
from ultralytics.utils.metrics import ap_per_class
O = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/out"
def load(tag):
    zs = [np.load(p) for p in sorted(glob.glob(f"{O}/stats_{tag}_*.npz"))]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
fp, q = load("bench_fp"), load("mb_m_w16top8_qat_rt")
files = E.val_files(0)
author = np.array(["_".join(os.path.basename(f).split("_")[:3]) for f in files])
tgt_img = []  # target_cls has no img id: rebuild per-image target counts from labels
for f in files:
    lab = E.load_labels(f, 1.0, (0, 0), 1, 1); tgt_img.append(np.asarray(lab[:, 0]) if len(lab) else np.zeros(0))
split = json.load(open("/mnt/storage/lib_project/models/Hailo_model/hailo_re/data_v14sec/bench_split.json"))
def ap(s, imgs):
    m = np.isin(s["img"], imgs)
    tc = np.concatenate([tgt_img[i] for i in imgs]) if len(imgs) else np.zeros(0)
    r = ap_per_class(s["tp"][m], s["conf"][m], s["pred_cls"][m], tc)
    return r  # tuple; ap is index 5 (n_cls x 10), unique classes index 6
def summ(s, imgs):
    r = ap_per_class(s["tp"][np.isin(s["img"], imgs)], s["conf"][np.isin(s["img"], imgs)], s["pred_cls"][np.isin(s["img"], imgs)],
                     np.concatenate([tgt_img[i] for i in imgs]))
    return r[5].mean(1), r[6]
allimg = np.arange(len(files))
print("n images", len(files), "fp preds", len(fp["conf"]), "q preds", len(q["conf"]))
a_fp, c_fp = summ(fp, allimg); a_q, c_q = summ(q, allimg)
print("overall mAP50-95 fp %.4f q %.4f gap %.4f" % (a_fp.mean(), a_q.mean(), a_fp.mean() - a_q.mean()))
cnt = np.bincount(np.concatenate(tgt_img).astype(int), minlength=20)
print("per class: cls n_targets fp q gap")
for c, x, y in sorted(zip(c_fp, a_fp, a_q), key=lambda t: -(t[1] - t[2])):
    print(f"  {int(c):2d} {cnt[int(c)]:6d} {x:.4f} {y:.4f} {x - y:+.4f}")
print("per author (>= 20 pages): author pages split fp q gap")
rows = []
for a in np.unique(author):
    imgs = np.where(author == a)[0]
    if len(imgs) < 20: continue
    f_, _ = summ(fp, imgs); q_, _ = summ(q, imgs)
    sp = "sel" if any(a in x for x in json.dumps(split.get("sel", split)).split('"')) else "?"
    rows.append((a, len(imgs), f_.mean(), q_.mean()))
for a, n, x, y in sorted(rows, key=lambda t: -(t[2] - t[3])):
    print(f"  {a:30s} {n:4d} {x:.4f} {y:.4f} {x - y:+.4f}")
print("keys of split json:", list(split)[:5])
print("AP per IoU threshold 0.50..0.95 (all classes): fp / q / gap")
rf = ap_per_class(fp["tp"], fp["conf"], fp["pred_cls"], np.concatenate(tgt_img))
rq = ap_per_class(q["tp"], q["conf"], q["pred_cls"], np.concatenate(tgt_img))
for i, t in enumerate(np.linspace(0.5, 0.95, 10)):
    print(f"  {t:.2f} {rf[5][:, i].mean():.4f} {rq[5][:, i].mean():.4f} {rf[5][:, i].mean() - rq[5][:, i].mean():+.4f}")
for c in (5, 10, 3, 0):
    k = list(rf[6]).index(c)
    print(f"cls {c} gap per IoU:", " ".join(f"{rf[5][k, i] - rq[5][k, i]:+.3f}" for i in range(10)))
from ultralytics import YOLO
print(YOLO("/mnt/storage/lib_project/models/Hailo_model/hailo_re/models/v14sec/best.pt").names)

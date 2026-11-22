"""Paired bootstrap over images for the mAP difference between two evaluations of the same image set.

Each evalgpu.sh run saved per-detection stats with global image ids (out/stats_TAG_k.npz); resampling images with
replacement and recomputing both mAPs on the same resample gives a CI for (B - A).

Usage: python boot_ci.py TAG_A TAG_B [N_BOOT]
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import evaluate as E  # noqa: E402


def load(tag):
    parts = sorted(glob.glob(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/out/stats_{tag}_*.npz"))
    assert parts, tag
    z = [np.load(p) for p in parts]
    return {k: np.concatenate([q[k] for q in z]) for k in z[0].files}


def per_image(s, n_img):
    """Split the flat stats into per-image chunks (predictions by img id; targets by their own img ids)."""
    pred = {i: [] for i in range(n_img)}
    for j, i in enumerate(s["img"]):
        pred[int(i)].append(j)
    return pred


def mAP(s, pred_idx, tgt_cls_by_img, imgs):
    m = E.Metric()
    for i in imgs:
        j = np.asarray(pred_idx[i], dtype=np.int64)
        m.stats["tp"].append(s["tp"][j])
        m.stats["conf"].append(s["conf"][j])
        m.stats["pred_cls"].append(s["pred_cls"][j])
        m.stats["target_cls"].append(tgt_cls_by_img[i])
        m.stats["img"].append(s["img"][j])
        m.stats["rank"].append(s["rank"][j])
    r = m.result()
    return r["mAP50"], r["mAP50_95"]


def main():
    a_tag, b_tag = sys.argv[1], sys.argv[2]
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    A, B = load(a_tag), load(b_tag)
    # targets are stored per image in evaluation order; rebuild per-image target lists from the val labels
    files = E.val_files(0)
    tgt = []
    for f in files:
        lab = E.load_labels(f, 1.0, (0, 0), 1, 1)
        tgt.append(np.asarray(lab[:, 0]) if len(lab) else np.zeros(0))
    n = len(files)
    assert sum(len(t) for t in tgt) == len(A["target_cls"]) == len(B["target_cls"]), "target count mismatch"
    pa, pb = per_image(A, n), per_image(B, n)
    base_a, base_b = mAP(A, pa, tgt, range(n)), mAP(B, pb, tgt, range(n))
    rng = np.random.default_rng(0)
    d = []
    for _ in range(nb):
        imgs = rng.integers(0, n, n)
        ma, mb = mAP(A, pa, tgt, imgs), mAP(B, pb, tgt, imgs)
        d.append((mb[0] - ma[0], mb[1] - ma[1]))
    d = np.array(d)
    for k, name in enumerate(("mAP50", "mAP50-95")):
        lo, hi = np.percentile(d[:, k], [2.5, 97.5])
        print(f"{name}: A {base_a[k]:.4f}  B {base_b[k]:.4f}  B-A {base_b[k] - base_a[k]:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(B>A) {np.mean(d[:, k] > 0):.2f}")


if __name__ == "__main__":
    main()

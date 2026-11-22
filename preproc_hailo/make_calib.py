"""Calibration sets (TRAIN pages only; never MDPBench / bench_v1 / holdout authors / orient val / flat val) -> uint8 npy.
  orient: page as served (long side 896, other side x32, >=224) at the top-left of a 896x896 canvas of gray 128
          (128/255-.5 ~ 0 = the conv zero padding); half clean, half P.photo; turned k x 90, k uniform.
  flat  : train_flat.DS mix (photo / flat / UVDoc train) at 488x712.
usage: make_calib.py orient|flat N"""
import sys, json, random, hashlib, numpy as np, cv2
sys.path.insert(0, "/mnt/storage/lib_project/models/layout_teacher/train")
import photo_data as P
what, n = sys.argv[1], int(sys.argv[2]); OUT = "/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/calib"
r = random.Random(1234); np.random.seed(1234); cv2.setRNGSeed(1234)
if what == "orient":
    import os; os.chdir("/mnt/storage/lib_project/models/layout_teacher/train")
    val = {tuple(x)[0] for v in json.load(open("runs/orient/val_items.json")).values() for x in v}
    m = {k: [it for it in v if it[0] not in val] for k, v in P.manifest().items() if v}
    doms = sorted(m); X = np.full((n, 896, 896, 3), 128, np.uint8); i = 0
    while i < n:
        dom = doms[i % len(doms)]; it = r.choice(m[dom])
        try:
            img, b, c = P.load_sample(*it)
            if i % 2: img = P.photo(img, b, c, r)[0]
        except Exception: continue
        img = np.ascontiguousarray(np.rot90(img, r.randrange(4)))
        h, w = img.shape[:2]; s = 896 / max(h, w)
        nw, nh = max(224, int(round(w * s / 32)) * 32), max(224, int(round(h * s / 32)) * 32)
        x = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        X[i, :min(nh, 896), :min(nw, 896)] = x[:896, :896]; i += 1
        if i % 128 == 0: print(i, flush=True)
else:
    import train_flat as TF, torch
    tr, va = TF.split(); uv = "/mnt/storage/lib_project/models/layout_teacher/data_ext/uvdoc/UVDoc_final"
    ds = TF.DS(tr, seed=1234, uv=(uv, TF.uv_list(uv)[0]), p_uv=.25)
    X = np.zeros((n, 712, 488, 3), np.uint8)
    for i, (x, g, k) in zip(range(n), ds):
        X[i] = x.permute(1, 2, 0).numpy()
        if (i + 1) % 128 == 0: print(i + 1, flush=True)
np.save(f"{OUT}/{what}_{n}.npy", X); print("saved", X.shape)

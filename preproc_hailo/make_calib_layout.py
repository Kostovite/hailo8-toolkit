"""Layout calibration (TRAIN manifest only, = train_ppdl3.Mix with the r5 mix, p_photo .5, photos straightened by flat2 as
served) -> calib/layout_N.npy uint8 896x896 black letterbox.  usage: make_calib_layout.py N"""
import sys, os, numpy as np, torch
os.environ["LAYOUT_FLAT_CKPT"] = "/mnt/storage/lib_project/models/layout_teacher/models_flat/flat2.pt"
sys.path.insert(0, "/mnt/storage/lib_project/models/layout_teacher/train"); os.chdir("/mnt/storage/lib_project/models/layout_teacher/train")
import train_ppdl3 as TP
n = int(sys.argv[1]); W = dict(thesis=.22, doclaynet=.06, d4la=.13, comics=.05, slides=.09, screens=.07, m6doc=.07, badlad=.11, indicdlp=.14, docsynth=.06)
ds = TP.Mix(n, .5, W, seed=4321); X = np.zeros((n, 896, 896, 3), np.uint8)
dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=16, collate_fn=lambda b: b)
i = 0
for batch in dl:
    for x, _ in batch:
        X[i] = (x.permute(1, 2, 0).numpy() * 255 + .5).astype(np.uint8); i += 1
    print(i, flush=True)
np.save(f"/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/calib/layout_{n}.npy", X[:i]); print("saved", i)

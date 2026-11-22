"""mAP of saved bench evaluations restricted to the author-disjoint halves in data_v14sec/bench_split.json.

Usage: VAL_DIR=.../bench/images python split_map.py TAG [TAG...]
"""
import json
import os
import sys

import numpy as np

import boot_ci as B
import evaluate as E

split = json.load(open("/mnt/storage/lib_project/models/Hailo_model/hailo_re/data_v14sec/bench_split.json"))
files = E.val_files(0)
author = ["_".join(os.path.basename(f).split("_")[:3]) for f in files]
tgt = []
for f in files:
    lab = E.load_labels(f, 1.0, (0, 0), 1, 1)
    tgt.append(np.asarray(lab[:, 0]) if len(lab) else np.zeros(0))
for tag in sys.argv[1:]:
    try:
        s = B.load(tag)
    except (AssertionError, FileNotFoundError, OSError) as e:
        print(f"{tag:28s} MISSING ({e})")
        continue
    pi = B.per_image(s, len(files))
    out = []
    for half in ("bench_sel", "bench_test"):
        imgs = [i for i, a in enumerate(author) if a in set(split[half])]
        m50, m = B.mAP(s, pi, tgt, imgs)
        out.append(f"{half} {m50:.4f}/{m:.4f}")
    print(f"{tag:28s} " + "   ".join(out))

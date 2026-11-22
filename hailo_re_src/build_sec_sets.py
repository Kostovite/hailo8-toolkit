"""Build the sec-merged QAT/eval sets for yolo26m_v14_sec.pt by NAME remap (never raw index).

Target taxonomy = the checkpoint's own names: cap catalogue code equ fig foot header list para reference sec
_unused_11 _unused_12 tab. Sources use the 14-class v14 names (sec_1/sec_2/sec_3 separate); sec_* -> sec.
Images are symlinked; labels rewritten. Fails loudly on unknown names, ids out of range or coords outside [0,1].

Usage: python build_sec_sets.py OUT_ROOT
"""
import glob
import os
import sys

import yaml

TARGET = ["cap", "catalogue", "code", "equ", "fig", "foot", "header", "list", "para", "reference", "sec",
          "_unused_11", "_unused_12", "tab"]
MERGE = {"sec_1": "sec", "sec_2": "sec", "sec_3": "sec"}
M = "/mnt/storage/lib_project/models"
SETS = {
    # name: (data.yaml, split, filename prefixes kept or None for all)
    "train": (f"{M}/yolo26m_v18_all/data.yaml", "train", ("HCMUS", "tex")),
    "val": (f"{M}/yolo26m_v18_all/data.yaml", "val", None),
    "bench": (f"{M}/yolo26m_bench_v1_14/data.yaml", None, None),
}


def names_of(data_yaml):
    n = yaml.safe_load(open(data_yaml))["names"]
    return [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)


def remap_table(src_names):
    tab = {}
    for i, name in enumerate(src_names):
        tgt = MERGE.get(name, name)
        if tgt not in TARGET or tgt.startswith("_unused"):
            raise SystemExit(f"unknown/unmapped source class {name!r}")
        tab[i] = TARGET.index(tgt)
    return tab


def build(out_root, name, data_yaml, split, prefixes):
    root = os.path.dirname(data_yaml)
    img_dir = os.path.join(root, "images", split) if split else os.path.join(root, "images")
    lab_dir = img_dir.replace(os.sep + "images", os.sep + "labels", 1)
    tab = remap_table(names_of(data_yaml))
    oi, ol = os.path.join(out_root, name, "images"), os.path.join(out_root, name, "labels")
    os.makedirs(oi, exist_ok=True)
    os.makedirs(ol, exist_ok=True)
    n_img = n_box = 0
    for f in sorted(glob.glob(os.path.join(img_dir, "*"))):
        base = os.path.basename(f)
        if prefixes and not base.startswith(prefixes):
            continue
        stem = base.rsplit(".", 1)[0]
        lp = os.path.join(lab_dir, stem + ".txt")
        lines = []
        if os.path.exists(lp):
            for ln in open(lp).read().split("\n"):
                if not ln.strip():
                    continue
                p = ln.split()
                c = tab[int(p[0])]
                xywh = [float(v) for v in p[1:5]]
                if not all(0.0 <= v <= 1.0 for v in xywh):
                    raise SystemExit(f"{lp}: coord out of [0,1]: {ln}")
                lines.append(" ".join([str(c)] + p[1:5]))
        dst = os.path.join(oi, base)
        if not os.path.lexists(dst):
            os.symlink(f, dst)
        with open(os.path.join(ol, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_box += len(lines)
    print(f"{name}: {n_img} images, {n_box} boxes  <- {img_dir} prefixes={prefixes}")


if __name__ == "__main__":
    out = sys.argv[1]
    for nm, (dy, sp, pref) in SETS.items():
        build(out, nm, dy, sp, pref)
    yaml.safe_dump({"path": out, "train": "train/images", "val": "val/images",
                    "names": {i: n for i, n in enumerate(TARGET)}}, open(os.path.join(out, "data.yaml"), "w"),
                   sort_keys=False)
    print("wrote", os.path.join(out, "data.yaml"))

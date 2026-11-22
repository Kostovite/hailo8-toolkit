"""Harvest every compiled op probe (zoo/ops/*/CASE/m.hef) into one JSONL: ONNX ops -> HN layer types -> HW node
types, per-node costs, and the compiler's rule strings; cases that failed to compile keep their error (a rule too).

Usage (DFC env): python harvest_all.py OUT.jsonl [SET ...]   (default sets: every zoo/ops dir with m.onnx cases)
"""
import glob
import json
import os
import sys
import tarfile

import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import harvest  # noqa: E402

ZOO = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/zoo/ops"


def hn_types(har):
    try:
        with tarfile.open(har) as t:
            nm = next(m for m in t.getnames() if m.endswith(".hn") and not m.endswith((".native.hn", ".fp.hn")))
            hn = json.loads(t.extractfile(nm).read())
        return sorted({v["type"] for v in hn["layers"].values()} - {"input_layer", "output_layer"})
    except Exception:  # noqa: BLE001
        return None


def main(out, sets):
    n = 0
    with open(out, "w") as f:
        for s in sets:
            for d in sorted(glob.glob(f"{ZOO}/{s}/*/")):
                if not os.path.exists(d + "m.onnx"):
                    continue
                rec = {"set": s, "case": os.path.basename(d.rstrip("/"))}
                try:
                    rec["onnx_ops"] = sorted({nd.op_type for nd in onnx.load(d + "m.onnx").graph.node})
                except Exception as e:  # noqa: BLE001
                    rec["onnx_ops"] = f"unreadable: {e}"
                try:
                    r = json.load(open(d + "result.json"))
                except (OSError, ValueError):
                    r = None
                if r:
                    rec.update({k: r.get(k) for k in ("stage", "error", "sqnr_hw_db", "sqnr_native_db", "prec")})
                rec["hn_types"] = hn_types(d + "q.har") if os.path.exists(d + "q.har") else None
                if os.path.exists(d + "m.hef"):
                    try:
                        rec.update(harvest(d + "m.hef", ""))
                    except Exception as e:  # noqa: BLE001
                        rec["harvest_error"] = f"{type(e).__name__}: {e}"[:300]
                f.write(json.dumps(rec) + "\n")
                n += 1
    print("records", n)


if __name__ == "__main__":
    sets = sys.argv[2:] or sorted({p.split("/")[-3] for p in glob.glob(f"{ZOO}/*/*/m.onnx")})
    main(sys.argv[1], sets)

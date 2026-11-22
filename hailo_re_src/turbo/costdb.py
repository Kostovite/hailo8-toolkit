"""Dataset for the context cost model: every sub-compile in turbo/t/subcache (one compiled context each) ->
features from its HN (MAC by layer kind, weights, activations, attention tokens) + the compiler's measured FPS /
latency, best over variants per distinct layer set.

Usage (DFC env): python costdb.py OUT.jsonl
"""
import glob
import json
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part_gen import costs  # noqa: E402

C = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/t/subcache"


def hn_of(har):
    with tarfile.open(har) as t:
        nm = next(m for m in t.getnames() if m.endswith(".hn") and not m.endswith((".native.hn", ".fp.hn")))
        return json.loads(t.extractfile(nm).read())


def features(hn):
    L = hn["layers"]
    c = costs(hn)
    f = {"n": 0, "mac_conv": 0, "mac_dw": 0, "mac_mm": 0, "w": 0, "act": 0, "attn": 0, "n_in": 0, "n_out": 0,
         "in_bytes": 0, "out_bytes": 0, "max_conv": 0, "n_conv": 0, "n_dw": 0, "n_ew": 0, "n_pool": 0, "n_resize": 0,
         "big_o": 0}
    for n, v in L.items():
        t = v["type"]
        if t == "input_layer":
            f["n_in"] += 1
            f["in_bytes"] += int(abs(__import__("math").prod(v["output_shapes"][0][1:])))
            continue
        if t == "output_layer":
            f["n_out"] += 1
            f["out_bytes"] += int(abs(__import__("math").prod(v["input_shapes"][0][1:])))
            continue
        f["n"] += 1
        x = c.get(n, {"mac": 0, "w": 0, "act": 0})
        f["w"] += x["w"]
        f["act"] += x["act"]
        if t == "conv":
            f["mac_conv"] += x["mac"]
            f["n_conv"] += 1
            f["max_conv"] = max(f["max_conv"], x["mac"])
            if v["output_shapes"][0][3] > 64:
                f["big_o"] += x["mac"]
        elif t == "dw":
            f["mac_dw"] += x["mac"]
            f["n_dw"] += 1
        elif t == "matmul":
            f["mac_mm"] += x["mac"]
            s = v["output_shapes"][0]
            f["attn"] = max(f["attn"], s[2])
        elif t.startswith("ew_") or t in ("ew_add", "ew_mult", "ew_sub"):
            f["n_ew"] += 1
        elif "pool" in t:
            f["n_pool"] += 1
        elif t == "resize":
            f["n_resize"] += 1
    return f


if __name__ == "__main__":
    best = {}
    for js in glob.glob(f"{C}/*.json"):
        try:
            j = json.load(open(js))
        except ValueError:
            continue
        if not j.get("ok") or not j.get("fps"):
            continue
        key = js[:-5]
        har = key + ".har"
        if not os.path.exists(har):
            continue
        body = open(key + ".alls").read() if os.path.exists(key + ".alls") else ""
        try:
            hn = hn_of(har)
        except Exception:  # noqa: BLE001
            continue
        lk = hn["name"] + json.dumps(sorted(n for n, v in hn["layers"].items()
                                            if v["type"] not in ("input_layer", "output_layer")))
        rec = {"key": os.path.basename(key), "fps": j["fps"], "lat_ms": j["latency_ms"], "model": hn["name"],
               "variant": ("lat" if "optimize_for_batch=1" in body else
                                                                 "max" if "level=max" in body else "default"),
               **features(hn)}
        b = best.setdefault(lk, {})
        for k_, better in (("fps", lambda a, o: a["fps"] > o["fps"]), ("lat", lambda a, o: a["lat_ms"] < o["lat_ms"])):
            if k_ not in b or better(rec, b[k_]):
                b[k_] = rec
    with open(sys.argv[1], "w") as f:
        for lk, b in best.items():
            f.write(json.dumps({"best_fps": b["fps"], "best_lat": b["lat"]}) + "\n")
    print(len(best), "distinct contexts")

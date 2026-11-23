"""layer_cost.py : measured Hailo-8 resources of every layer of a quantized HAR, one single-context compile per layer.

The partitioner's (and part_gen.py's) per-node cost estimates are far off for 16-bit layers (16-bit defuses into more
pieces): PP-DocLayoutV3 mix16 started from an even split that overloaded every context and searched > 4 h without one
valid partition. Here each layer is cut out at its real shape and precision (turbo/subhar.py: cut edges become
input/output layers carrying the producer's quantization), compiled alone in one context, and the compiler's own
per-node resources are read back from the HEF (sweep.harvest). Packing contexts against these numbers replaces the
estimate.

  python layer_cost.py list Q.har                     -> one layer name per line (cut units)
  python layer_cost.py run  Q.har LAYER OUT_DIR       -> OUT_DIR/LAYER.json  {ok, error, res_sum, nodes, wall_s}
  python layer_cost.py table OUT_DIR                  -> per-layer resource table (tsv on stdout)
A cut unit = the layer + its const_input producers; a format_conversion / precision_change also takes its producer
(a sub-network may not start at one, RULES R10), and only the layer's own HW nodes are counted.
"""
import glob
import json
import os
import re
import sys
import tarfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "turbo"))

SKIP = {"input_layer", "output_layer", "const_input"}
LEAD = ("format_conversion", "precision_change")  # need their producer in the cut


def hn_of(har):
    t = tarfile.open(har)
    n = next(x for x in t.getnames() if x.endswith(".hn") and "native" not in x and ".fp." not in x)
    return json.loads(t.extractfile(n).read())


def units(har):
    L = hn_of(har)["layers"]; short = {k: k.split("/")[-1] for k in L}
    out = []
    for k, v in L.items():
        if v["type"] in SKIP:
            continue
        s = short[k]; cut = [s]
        for p in v.get("input", []):
            pv = L.get(p, {})
            if pv.get("type") == "const_input" or (s.startswith(LEAD) and pv.get("type") not in ("input_layer",)):
                cut.append(short[p])
                for pp in pv.get("input", []):  # the producer's own const inputs
                    if L.get(pp, {}).get("type") == "const_input":
                        cut.append(short[pp])
        out.append((s, sorted(set(cut))))
    return out


def own(node_name, layer):
    """HW node belongs to layer: exact name or a defuse piece (layer_d0, layer_sd3, layer_defuse_..., layer_fs)."""
    return node_name == layer or re.match(rf"^{re.escape(layer)}(_[a-z]*\d*|_defuse\w*|_fs|_dc)+$", node_name) is not None


def run(har, layer, out_dir):
    import subhar
    from sweep import harvest
    os.makedirs(out_dir, exist_ok=True)
    cut = dict(units(har))[layer]
    w = os.path.join(out_dir, layer + ".d"); os.makedirs(w, exist_ok=True); os.chdir(w)
    res = {"layer": layer, "cut": cut}; t0 = time.time()
    try:
        res.update(subhar.build(har, cut, "sub.har"))
        from hailo_sdk_client import ClientRunner
        r = ClientRunner(har="sub.har")
        if os.environ.get("LC_ALLS"):  # e.g. performance_param(fps=20): minimum footprint, not the spread-out one
            r.load_model_script(os.environ["LC_ALLS"] + "\n")
        hef = r.compile()
        open("sub.hef", "wb").write(hef)
        r.save_har("sub_compiled.har")
        with tarfile.open("sub_compiled.har") as t:
            auto = next((t.extractfile(x).read().decode() for x in t.getnames() if x.endswith(".auto.alls")), "")
        h = harvest("sub.hef", auto)
        mine = [n for n in h["nodes"] if own(n["name"], layer)]
        tot = {}
        for n in mine:
            tot["hw_nodes"] = tot.get("hw_nodes", 0) + 1  # one layer controller (LCU) per HW node
            for k, v in n["res"].items():
                try:
                    x = float(v) if not isinstance(v, bool) else None  # counts come as strings ("6")
                except (TypeError, ValueError):
                    x = None
                if x is not None:
                    tot[k] = tot.get(k, 0) + x
        res.update(ok=True, contexts=len({n["context"] for n in h["nodes"]}), n_nodes=len(mine), res_sum=tot,
                   nodes=[{"name": n["name"], "type": n["type"], "res": n["res"]} for n in mine],
                   defuse_lines=h["defuse_lines"], bottleneck_fps=h["bottleneck_fps"])
    except Exception as e:  # noqa: BLE001  a layer that does not compile alone is itself the key fact
        res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}", trace=traceback.format_exc()[-600:])
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(out_dir, layer + ".json"), "w"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("layer", "ok", "contexts", "n_nodes", "wall_s", "error")}))


def table(out_dir):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, "*.json")))]
    keys = sorted({k for r in rows if r.get("ok") for k in r["res_sum"]})
    print("\t".join(["layer", "ok", "ctx", "nodes"] + keys))
    for r in rows:
        s = r.get("res_sum", {})
        print("\t".join([r["layer"], str(r.get("ok")), str(r.get("contexts", "")), str(r.get("n_nodes", ""))]
                        + [f"{s.get(k, '')}" for k in keys]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        for s, _ in units(sys.argv[2]):
            print(s)
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "table":
        table(sys.argv[2])

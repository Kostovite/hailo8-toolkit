import sys, json, argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo")
import ctx_grow as C
G = "/mnt/storage/lib_project/models/Hailo_model/convert/runs/layout_g3"
a = argparse.Namespace(har=G + "/q.har", out=G + "/grow", timeout=2400, mem="20G")
L = C.hn_of(a.har)["layers"]
order = [n for n, v in L.items() if v["type"] not in C.SKIP and v["type"] != "const_input"]
starts = [int(x) for x in sys.argv[1].split(",")]
with ThreadPoolExecutor(len(starts)) as p:
    for s, (ok, r) in zip(starts, p.map(lambda s: C.test(a, L, order, s, len(order)), starts)):
        print(s, order[s].split("/")[-1], ok, r.get("fps_target"), r.get("s"), str(r.get("error"))[:120], flush=True)

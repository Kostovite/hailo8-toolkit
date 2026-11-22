"""Print every edge that crosses a context boundary of a partition: producer(type,ctx) -> consumer(type,ctx), shape.
Usage: python cross_edges.py FULL.har PART_DIR"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../turbo"))
from part_fix import hn_of
hn = hn_of(sys.argv[1]); L = hn["layers"]; full = lambda s: s if "/" in s else f"{hn['name']}/{s}"
ctx, i = {}, 0
while os.path.exists(f"{sys.argv[2]}/c{i}.layers.json"):
    for n in json.load(open(f"{sys.argv[2]}/c{i}.layers.json")): ctx[full(n)] = i
    i += 1
for n, v in L.items():
    for o in v["output"]:
        if n in ctx and o in ctx and ctx[o] != ctx[n]:
            print(f"  {n.split('/')[-1]}({v['type']},c{ctx[n]}) -> {o.split('/')[-1]}({L[o]['type']},c{ctx[o]}) {v['output_shapes'][0]}")

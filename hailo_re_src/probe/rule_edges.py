"""Per crossing-edge class (rule + producer type + consumer type), how many fuzz cuts that contain it were exact vs
failed in the ORIGINAL run (before R17-R22): which narrow form of each rule the data supports."""
import json, os, re, sys, collections
sys.path.insert(0, "turbo")
from part_fix import hn_of
P = "probe/runs"
ex_re = {(r["case"], r["tag"]) for r in map(json.loads, open(f"{P}/recheck_infra.jsonl")) if r.get("res") == "EXACT"}
grp = lambda n: (lambda m: m and m.group(0))(re.search(r"(layer_normalization|softmax)\d+$", n))
st = collections.defaultdict(lambda: [0, 0, []])
for f in ("fz1", "fz2"):
    for l in open(f"{P}/{f}/fuzz.jsonl"):
        r = json.loads(l)
        if r.get("stock") != "ok": continue
        hn = hn_of(f"{P}/{f}/{r['case']}/{r['case']}/q.har"); L = hn["layers"]
        for c in r["cuts"]:
            pd = f"{P}/{f}/{r['case']}/{c['tag']}/part"
            if not os.path.isdir(pd): continue
            ctx, i = {}, 0
            while os.path.exists(f"{pd}/c{i}.layers.json"):
                for n in json.load(open(f"{pd}/c{i}.layers.json")): ctx[n if "/" in n else f"{hn['name']}/{n}"] = i
                i += 1
            good = bool(c.get("exact")) or (r["case"], c["tag"]) in ex_re
            keys = set()
            for n, v in L.items():
                for o in v["output"]:
                    if n in ctx and o in ctx and ctx[o] > ctx[n]:
                        sh = v["output_shapes"][0]; on = o.split("/")[-1]; nn = n.split("/")[-1]
                        why = ("R17" if v["type"] == "format_conversion" else
                               "R18" if len(sh) == 4 and sh[1] == 1 and sh[2] > 1 else
                               "R19" if grp(n) and grp(n) == grp(o) else
                               "R20" if v["type"] == "feature_splitter" else
                               "R21" if v["type"] == "dense" else
                               "R16b" if L[o]["type"] == "resize" else
                               "R22" if "conv_slice" in o else None)
                        if why:
                            pn = re.sub(r"\d+", "", nn.split("_layer_norm")[0].split("_softmax")[0])
                            cn = re.sub(r"\d+", "", on.split("_layer_norm")[0].split("_softmax")[0])
                            k = f"{why} {pn}->{cn}" + (f" {L[o].get('params',{}).get('method','')} {sh[1]}x{sh[2]}" if why == "R16b" else "")
                            keys.add(k)
            for k in keys:
                st[k][0 if good else 1] += 1
                if not good: st[k][2].append(f"{r['case']}:{c['tag']}")
for k in sorted(st): print(f"{k:60s} exact {st[k][0]:3d} fail {st[k][1]:3d}", " ".join(st[k][2][:4]))

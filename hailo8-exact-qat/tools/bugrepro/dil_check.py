"""Guard for the dilated-conv silicon defect (RULES.md section 25): flags every compiled conv piece of a dilated conv
whose output row is 9 or 10 wide. Those pieces read stale memory on the chip (frame- and history-dependent garbage).
Usage: python dil_check.py M.HEF Q.HAR   -> exit 1 and one line per bad piece; fix: zero-pad W, conv, crop."""
import json, re, sys
from ighw import load
from hailo_sdk_client import ClientRunner
hef, har = sys.argv[1], sys.argv[2]
hn = ClientRunner(har=har).get_hn(); hn = json.loads(hn) if isinstance(hn, str) else hn
dil = {n.split("/")[-1]: l["params"]["dilations"] for n, l in hn["layers"].items()
       if l["type"] in ("conv", "dw") and max(l["params"].get("dilations", [1, 1, 1, 1])[1:3]) > 1}
bad = []
for cn in load(hef).contexts_graph.nodes:
    for n in cn.network_graph.nodes:
        base = re.sub(r"_(sd|d)\d+$", "", n.name)
        if base in dil and n.name != base + "_sdc" and n.output_shapes and n.output_shapes[0].width in (9, 10):
            bad.append((n.name, n.output_shapes[0].width, dil[base]))
for b in bad:
    print(f"BAD dilated conv piece {b[0]}: width {b[1]}, dilations {b[2]} -> zero-pad W to a multiple of 8 (or 16), conv, crop")
print(f"{len(dil)} dilated conv layer(s), {len(bad)} bad piece(s)")
sys.exit(1 if bad else 0)

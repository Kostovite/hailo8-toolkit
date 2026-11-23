"""kv8_alls.py PARSED.har : print the quantization_param line that makes every conv fed by a token-layout format_conversion
(the AIFI K and V projections after aifi_spatial.py) a8_w8_a8.

Why: the DFC parser places the conv that produces a matmul's second operand (K for Q.K, V for P.V) behind the
28x28 -> tokens format_conversion, which is 8-bit. Left at the mix16 default (a16_w8_a8) the quantizer puts an 8 -> 16 bit
precision_change on the 784-wide token row in front of it, and that layer cannot be placed on Hailo-8 ("needs
super-defuse" / "more than 1 subclusters are needed for 16bit L2 biases"). a8_w8_a8 reads the 8-bit tokens directly.
Measured 2026-11-22: AIFI context then compiles in 1 context, 339 FPS.
"""
import json
import sys
import tarfile

t = tarfile.open(sys.argv[1])
hn = json.loads(t.extractfile(next(x for x in t.getnames() if x.endswith(".hn") and "native" not in x and ".fp." not in x)).read())
L = hn["layers"]
pick = []
for k, v in L.items():
    if v["type"] != "conv":
        continue
    ins = v.get("input", [])
    if ins and all(L.get(i, {}).get("type") == "format_conversion" for i in ins) \
            and all(L[i]["output_shapes"][0][1] == 1 for i in ins):  # token layout [-1, 1, N, C]
        pick.append(k.split("/")[-1])
assert len(pick) == 2, f"expected the K and V projections, found {pick}"
print(f"quantization_param([{', '.join(sorted(pick))}], precision_mode=a8_w8_a8)")

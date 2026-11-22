"""Silent-wrong guard (RULES 22): the chip reads a dynamic-weights matmul's SECOND operand as signed int8; the
emulator as uint8. List every such operand whose quantized range exceeds 127 (unsigned, zp 0, max > 127 LSB) and
print the force_range_out lines that fix it (an operand the HAR's script already forces counts as handled). Usage: python dynmm_check.py Q.har"""
import io, json, re, sys, tarfile
import numpy as np
t = tarfile.open(sys.argv[1]); names = t.getnames()
hn = json.loads(t.extractfile([x for x in names if x.endswith(".hn") and "native" not in x and ".fp." not in x][0]).read())
L = hn["layers"]; z = np.load(io.BytesIO(t.extractfile([x for x in names if x.endswith(".q.npz")][0]).read()))
alls = next((t.extractfile(x).read().decode() for x in names if x.endswith(".alls") and "auto" not in x), "")
bad = 0
for k, v in L.items():
    if v["type"] != "matmul" or not v.get("params", {}).get("dynamic_weights"):
        continue
    s = v["input"][1]
    while L[s]["type"] == "format_conversion":
        s = L[s]["input"][0]
    lv, zp = z[f"{s}/limvals_out:0"], float(z[f"{s}/zero_point_out:0"])
    sc = float(z[f"{s}/qp_out:0"][1]) or float(z[f"{s}/layer_params/output_scales:0"].ravel()[0])  # qp = [zp, scale];
    signed = L[s].get("quantization_params", {}).get("signed_output", False)  # multi-output layers keep scale 0 there
    qmax, qmin = lv[1] / sc + zp, lv[0] / sc + zp
    if signed:  # the quantizer clips a signed output to int8 itself: the chip and the emulator agree on it
        qmax, qmin = min(qmax, 127), max(qmin, -128)
    forced = any(re.search(rf"[\[(,/ ]{s.split('/')[-1]}[\]),]", ln) and "force_range_out" in ln
                 for ln in alls.splitlines())
    if forced:  # limvals are then the forced range, not the data's: trust the recipe (data max = half of it)
        print(f"{k.split('/')[-1]}: operand {s.split('/')[-1]} range forced by the model script (ok)")
    elif qmax > 127.5 or qmin < -128.5:
        bad += 1
        print(f"{k.split('/')[-1]}: operand {s.split('/')[-1]} q-range [{qmin:.0f}, {qmax:.0f}] (limvals {lv}) -> "
              f"quantization_param({s.split('/')[-1]}, force_range_out=[{lv[0]:.6g}, {lv[0] + 2 * (lv[1] - lv[0]):.6g}])")
print(f"{bad} dynamic-matmul weight operand(s) outside int8" if bad else "all dynamic-matmul weight operands within int8")
sys.exit(1 if bad else 0)

"""Post-compile guard for every known Hailo-8 silent-wrong pattern (probe/RULES.md sections 22-26).

Usage: python hailo8_guard.py M.HEF Q_OR_COMPILED.HAR [AUTO.alls] [Q.har]   -> exit 1 if any S1 pattern is present
  (Q.har: the quantized input, whose model script holds the quantization commands; a compiled HAR's script is
  replaced by the compiler script)
  G1 dynamic-matmul weights operand: unsigned > 127, or SIGNED but non-negative (the chip runs that producer
     unsigned and reads int8; its scale is range/255, so values above HALF the range wrap where the emulator
     saturates: 2*max still left 4.3% wrong, 4*max is bit-exact)                 fix: force_range_out=[0, 4*max]
  G2 dilated conv (d <= 4) piece 9..8+d-1 wide (stale memory; d >= 5: warning, untested)  fix: compat dil_pad
  G3 ANY dynamic matmul INPUT_FEATURES-defused (P.V 1-6 dB, Q.K forced 96% wrong) fix: pin its SCs (compile_sub)
  G4 deconv kernel == stride lowered as DECONV / FEATURE_INTERLEAVE (~100% wrong) fix: SUPER_DECONV (R23)
  G5 GroupNorm(1) whole-tensor decomposition (~50% wrong)                         fix: compat rule gn1
  G6 softmax denominator accumulator can wrap (row sum > the calibrated maximum)  fix: smx_headroom (warning only)
"""
import io
import json
import re
import sys
import tarfile

import numpy as np


def har_parts(har):
    t = tarfile.open(har)
    names = t.getnames()
    hn = json.loads(t.extractfile([x for x in names if x.endswith(".hn") and "native" not in x and ".fp." not in x][0]).read())
    z = np.load(io.BytesIO(t.extractfile([x for x in names if x.endswith(".q.npz")][0]).read()))
    auto = next((t.extractfile(x).read().decode() for x in names if x.endswith(".auto.alls")), "")
    user = next((t.extractfile(x).read().decode() for x in names if x.endswith(".alls") and "auto" not in x), "")
    return hn["layers"], z, auto, user


def short(n):
    return n.split("/")[-1]


def main(hef, har, auto_path=None, qhar=None):
    L, z, auto, user = har_parts(har)
    if auto_path:
        auto = open(auto_path).read()
    if qhar:
        user += "\n" + har_parts(qhar)[3]
    s1, warn = [], []
    # G1
    for k, v in L.items():
        if v["type"] != "matmul" or not v.get("params", {}).get("dynamic_weights"):
            continue
        s = v["input"][1]
        while L[s]["type"] == "format_conversion":
            s = L[s]["input"][0]
        if f"{s}/limvals_out:0" not in z:
            continue
        lv, zp = z[f"{s}/limvals_out:0"], float(z[f"{s}/zero_point_out:0"])
        sc = float(z[f"{s}/qp_out:0"][1]) or float(z[f"{s}/layer_params/output_scales:0"].ravel()[0])
        signed = L[s].get("quantization_params", {}).get("signed_output", False)
        forced = any(re.search(rf"[\[(,/ ]{short(s)}[\]),]", ln) and "force_range_out" in ln for ln in user.splitlines())
        qmax = lv[1] / sc + zp
        fix = f"quantization_param({short(s)}, force_range_out=[{min(lv[0], 0):.6g}, {4 * lv[1]:.6g}])"
        if forced:
            continue
        if not signed and qmax > 127.5:
            s1.append(f"G1 {short(k)}: weights operand {short(s)} unsigned up to {qmax:.0f} LSB -> {fix}")
        elif signed and lv[0] >= 0:
            s1.append(f"G1 {short(k)}: weights operand {short(s)} signed but non-negative: values above the calibrated "
                      f"max wrap on the chip -> {fix}")
    # HEF nodes (G2, G5)
    sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe")
    from ighw import load
    dil = {short(n): l["params"]["dilations"][2] for n, l in L.items()
           if l["type"] in ("conv", "dw") and l["params"].get("dilations", [1, 1, 1, 1])[2] > 1}
    nodes = [n for cn in load(hef).contexts_graph.nodes for n in cn.network_graph.nodes]
    for n in nodes:
        base = re.sub(r"_(sd|d)\d+$", "", n.name)
        if base not in dil or n.name == base + "_sdc" or not n.output_shapes:
            continue
        w, d = n.output_shapes[0].width, dil[base]  # chip sweep: wrong iff 9 <= w <= 8 + d - 1 (d = 2..4)
        if 9 <= w <= 8 + d - 1:
            (s1 if d <= 4 else warn).append(f"G2 dilated conv piece {n.name}: width {w} at dilation {d}"
                                            f"{'' if d <= 4 else ' (d >= 5 untested)'} -> compat rule dil_pad")
    gn = sorted({re.sub(r".*_(layer_normalization\d+)$", r"\1", n.name) for n in nodes
                 if re.search(r"reduce_mean1_spatial_(variance_)?layer_normalization\d+$", n.name)})
    for g in gn:
        s1.append(f"G5 {g}: GroupNorm(1) whole-tensor decomposition -> compat rule gn1")
    # G3, G4 from the compiler's own script
    for ln in auto.splitlines():
        m = re.search(r"defuse\((\w+), \d+, defuse_type=INPUT_FEATURES\)", ln)
        if m:
            full = next((n for n in L if short(n) == m.group(1)), None)
            if full and L[full]["type"] == "matmul" and L[full].get("params", {}).get("dynamic_weights"):
                s1.append(f"G3 {m.group(1)}: dynamic matmul INPUT_FEATURES-defused -> pin manual_scs_selection")
        m = re.search(r"defuse\((\w+), \d+, defuse_type=DECONV\)", ln)
        if m:
            full = next((n for n in L if short(n) == m.group(1)), None)
            p = L[full]["params"] if full else {}
            if full and list(p.get("kernel_shape", [0, 0])[:2]) == list(p.get("strides", [1, 0, 0, 1])[1:3]):
                s1.append(f"G4 {m.group(1)}: deconv kernel == stride lowered as DECONV -> SUPER_DECONV (R23)")
    # G6
    for n, l in L.items():
        if l["type"] != "reduce_sum" or "softmax" not in n:
            continue
        k = np.abs(np.asarray(z.get(f"{n}/kernel:0", [2]), np.float64)).max()
        sh = float(np.asarray(z.get(f"{n}/output_stage/mult_shift:0", 1)))
        red = int(np.prod([l["input_shapes"][0][a] for a in l["params"].get("reduce_axes", [3])])) // int(l["params"].get("groups", 1))
        worst = red * 255 * k / 2 ** sh
        if worst > 32767:
            warn.append(f"G6 {short(n)}: denominator accumulator covers {100 * 32767 / worst:.0f}% of the worst-case "
                        f"row sum ({red} keys): rows above the calibrated maximum wrap -> smx_headroom")
    for x in s1:
        print("S1", x)
    for x in warn:
        print("WARN", x)
    print(f"hailo8_guard: {len(s1)} silent-wrong pattern(s), {len(warn)} warning(s)")
    return 1 if s1 else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))

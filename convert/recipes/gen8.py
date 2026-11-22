"""gen8.py PARSED_HAR OUT_ALLS: the rewritten upsample blocks (resize/slice/concat) -> a8_w8_a8, their dw3x3 -> a8_w8_a16."""
import sys
from hailo_sdk_client import ClientRunner
hn = ClientRunner(har=sys.argv[1]).get_hn_model(); a8, dw = [], []; flt = sys.argv[3] if len(sys.argv) > 3 else ""
for l in hn.stable_toposort():
    on = " ".join(getattr(l, "original_names", []) or [])
    if flt and flt not in on: continue
    if any(on.endswith(k) for k in ("_nn", "_pad", "_h", "_t", "_b", "_l", "_r", "_crop")) and "Resize" in on: a8.append(l.name.split("/")[-1])
    if on.endswith("_dwc"): dw.append(l.name.split("/")[-1])
open(sys.argv[2], "w").write(f"quantization_param([{', '.join(a8)}], precision_mode=a8_w8_a8)\nquantization_param([{', '.join(dw)}], precision_mode=a8_w8_a16)\n")
print(len(a8), "a8 layers;", len(dw), "dw a8_w8_a16:", dw)

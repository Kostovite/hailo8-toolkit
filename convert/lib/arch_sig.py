"""arch_sig.py Q_HAR -> prints a 16-hex signature of the compiled-relevant architecture: layer names, types, shapes,
precision modes and compile params (NOT weights / scales), so a retrain of the same model maps to the same cache."""
import sys, json, hashlib
from hailo_sdk_client import ClientRunner
h = ClientRunner(har=sys.argv[1]).get_hn(); h = json.loads(h) if isinstance(h, str) else h
rows = []
for n, v in sorted(h["layers"].items()):
    q = v.get("quantization_params", {})
    rows.append([n, v["type"], v.get("input_shapes"), v.get("output_shapes"), v.get("input"), q.get("precision_mode"),
                 {k: v.get("params", {}).get(k) for k in ("kernel_shape", "strides", "dilations", "groups", "padding")}])
print(hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()[:16])

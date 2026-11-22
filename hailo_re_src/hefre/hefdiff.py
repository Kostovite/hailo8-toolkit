"""hefdiff.py A.hef B.hef: field-level diff of two HEFs (protobuf leaves + CCW blob), to tell real differences
(placement, config bytes) from metadata noise (compile time, paths, embedded logs)."""
import hashlib, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hef_dump import load  # noqa: E402
from google.protobuf.descriptor import FieldDescriptor as FD  # noqa: E402


def leaves(m, p, out):
    for f, v in m.ListFields():
        q = p + "/" + f.name
        if f.type == FD.TYPE_MESSAGE:
            if f.label == FD.LABEL_REPEATED:
                for i, x in enumerate(v):
                    leaves(x, f"{q}[{i}]", out)
            else:
                leaves(v, q, out)
        else:
            if f.label == FD.LABEL_REPEATED:
                v = list(v)
            if isinstance(v, (bytes, bytearray)) or (isinstance(v, list) and len(v) > 8):
                v = "sha1:" + hashlib.sha1(bytes(v) if isinstance(v, (bytes, bytearray)) else repr(v).encode()).hexdigest()[:12] + f" n={len(v)}"
            out[q] = v
    return out


(ha, _, sa, oa), (hb, _, sb, ob) = load(sys.argv[1]), load(sys.argv[2])
A, B = leaves(ha, "", {}), leaves(hb, "", {})
d = sorted(k for k in set(A) | set(B) if A.get(k) != B.get(k))
ra, rb = open(sys.argv[1], "rb").read()[oa:], open(sys.argv[2], "rb").read()[ob:]
print(f"leaves {len(A)}/{len(B)} differ {len(d)}; ccw blob {len(ra)}/{len(rb)} {'same' if ra == rb else 'DIFF'}")
for k in d[:25]:
    print(" ", k, "|", str(A.get(k))[:90], "|", str(B.get(k))[:90])

# the `mapping` blob = integrated_hw_graph (compiler's per-node resources / profiler data): diff it too
try:
    from hailo_sdk_client.allocator import integrated_hw_graph_pb2 as IG
    ga, gb = IG.ProtoIntegratedHWGraph() if hasattr(IG, "ProtoIntegratedHWGraph") else None, None
    cls = [getattr(IG, n) for n in dir(IG) if n.endswith("Graph") and hasattr(getattr(IG, n), "ParseFromString")][0]
    ga, gb = cls(), cls()
    ga.ParseFromString(ha.mapping); gb.ParseFromString(hb.mapping)
    MA, MB = leaves(ga, "", {}), leaves(gb, "", {})
    md = sorted(k for k in set(MA) | set(MB) if MA.get(k) != MB.get(k))
    import collections, re
    kinds = collections.Counter(re.sub(r"\[\d+\]", "[]", k) for k in md)
    print(f"mapping ({cls.__name__}): leaves {len(MA)}/{len(MB)} differ {len(md)}")
    for k, n in kinds.most_common(12):
        ex = next(x for x in md if re.sub(r"\[\d+\]", "[]", x) == k)
        print(f"   {n:5d} x {k} | e.g. {str(MA.get(ex))[:50]} -> {str(MB.get(ex))[:50]}")
except Exception as e:  # noqa: BLE001
    print("mapping diff skipped:", e)

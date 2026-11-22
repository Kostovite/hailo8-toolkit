"""Every host-side ONNX op runs on the host: the official onnx node conformance tests (onnx/backend/test/data/node)
through onnxruntime CPU, falling back to onnx's ReferenceEvaluator where onnxruntime has no kernel.

Usage (general env): python host_ops_check.py OPS...   -> host_ops.json  {op: {"ort": k/n, "ref": k/n}}
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import onnx
from onnx import numpy_helper

warnings.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(onnx.__file__), "backend/test/data")
RANDOM = {"Bernoulli", "Multinomial", "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"}


def load_pb(path, vtype):
    """Test-data proto of the declared input/output type (a SequenceProto can parse as a TensorProto silently)."""
    raw = open(path, "rb").read()
    which = vtype.WhichOneof("value") if vtype is not None else "tensor_type"
    if which == "sequence_type":
        p = onnx.SequenceProto()
        p.ParseFromString(raw)
        return numpy_helper.to_list(p)
    if which == "optional_type":
        p = onnx.OptionalProto()
        p.ParseFromString(raw)
        return numpy_helper.to_optional(p)
    p = onnx.TensorProto()
    p.ParseFromString(raw)
    return numpy_helper.to_array(p)


def same(a, b, random=False):
    if a is None or b is None:
        return a is None and b is None
    if random:  # random generators: the run itself and the shape / dtype are what can be checked
        return np.asarray(a).shape == np.asarray(b).shape and np.asarray(a).dtype == np.asarray(b).dtype
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "OUS":
        return a.shape == b.shape and (a.astype(str) == b.astype(str)).all()
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f":
        return bool(np.allclose(a, b, rtol=1e-3, atol=1e-4, equal_nan=True))
    return bool((a == b).all())


def run_case(d, engine, random=False):
    m = onnx.load(f"{d}/model.onnx")
    ins = [i for i in m.graph.input if i.name not in {t.name for t in m.graph.initializer}]
    outs = list(m.graph.output)
    sets = sorted(glob.glob(f"{d}/test_data_set_*"))
    ok = 0
    for s in sets:
        feeds = {v.name: load_pb(f"{s}/input_{i}.pb", v.type) for i, v in enumerate(ins)
                 if os.path.exists(f"{s}/input_{i}.pb")}
        want = [load_pb(f"{s}/output_{i}.pb", v.type) for i, v in enumerate(outs)
                if os.path.exists(f"{s}/output_{i}.pb")]
        if engine == "ort":
            import onnxruntime as ort
            ort.set_default_logger_severity(4)
            got = ort.InferenceSession(f"{d}/model.onnx", providers=["CPUExecutionProvider"]).run(None, feeds)
        else:
            from onnx.reference import ReferenceEvaluator
            got = ReferenceEvaluator(m).run(None, feeds)
        ok += all(same(g, w, random) for g, w in zip(got, want))
    return ok == len(sets)


res = {}
for op in sys.argv[1:]:
    dirs = [d for d in sorted(glob.glob(f"{DATA}/*/test_*")) if os.path.exists(f"{d}/model.onnx") and
            any(n.op_type == op for n in onnx.load(f"{d}/model.onnx").graph.node)]
    r = {"tests": len(dirs), "ort": 0, "ref": 0, "either": 0}
    for d in dirs:
        a = b = False
        try:
            a = run_case(d, "ort", op in RANDOM)
        except Exception:  # noqa: BLE001
            pass
        if not a:
            try:
                b = run_case(d, "ref", op in RANDOM)
            except Exception:  # noqa: BLE001
                pass
        r["ort"] += a
        r["ref"] += b
        r["either"] += a or b
    res[op] = r
    print(op, r, flush=True)
json.dump(res, open("host_ops.json", "w"), indent=1)

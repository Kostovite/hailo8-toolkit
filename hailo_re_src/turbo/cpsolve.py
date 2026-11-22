"""Solve dumped Hailo CP-SAT models (text CpModelProto) with modern OR-tools: status, wall time, objective.
Usage: ortools_env/bin/python turbo/cpsolve.py [--workers N] [--time S] MODEL.pbtxt ..."""
import argparse
import time

from google.protobuf import text_format
from ortools.sat import cp_model_pb2
from ortools.sat.python import cp_model


def _expr(ref, key):
    """Old-schema variable reference -> new LinearExpressionProto (negative ref -r-1 means -var)."""
    v, c = (ref, 1) if ref >= 0 else (-ref - 1, -1)
    return f"{key} {{ vars: {v} coeffs: {c} }}"


def upgrade(text):
    """2020 CpModelProto text -> current schema: int_div/int_prod/all_diff took plain variable refs (target, vars),
    now they take linear expressions (target, exprs)."""
    out, block = [], None
    for line in text.split("\n"):
        s = line.strip()
        if s in ("int_div {", "int_prod {", "all_diff {"):
            block = s[:-2]
            out.append(line)
            continue
        if block:
            if s == "}":
                block = None
            elif s.startswith("target:"):
                line = "    " + _expr(int(s.split(":")[1]), "target")
            elif s.startswith("vars:"):
                line = "    " + _expr(int(s.split(":")[1]), "exprs")
        out.append(line)
    return "\n".join(out)


def load(path):
    m = cp_model_pb2.CpModelProto()
    text_format.Parse(upgrade(open(path).read()), m)
    return m


def solve(m, workers=8, limit=60.0):
    s = cp_model.CpSolver()
    s.parameters.num_workers = workers
    s.parameters.max_time_in_seconds = limit
    model = cp_model.CpModel()
    model.proto.parse_text_format(text_format.MessageToString(m))
    t = time.time()
    st = s.Solve(model)
    return s.StatusName(st), time.time() - t, (s.ObjectiveValue() if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    for p in a.models:
        m = load(p)
        st, dt, obj = solve(m, a.workers, a.time)
        print(f"{p.split('/')[-1]}: vars {len(m.variables)} cons {len(m.constraints)} -> {st} in {dt:.2f}s obj {obj}")

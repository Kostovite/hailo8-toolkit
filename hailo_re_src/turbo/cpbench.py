"""Offline benchmark of the CP-SAT problems the Hailo compiler solved (memo-cache corpus: KEY.req / KEY.resp / KEY.us).
Re-solves each with modern OR-tools CP-SAT in several configurations and compares with Hailo's own answer.
Usage: ortools_env/bin/python turbo/cpbench.py CACHE_DIR [--top N] [--configs w1,w8,...] [--limit-x F] [--out TSV]"""
import argparse, glob, os, time
from ortools.sat import cp_model_pb2, sat_parameters_pb2
from ortools.sat.python import cp_model

ST = {0: "UNKNOWN", 1: "MODEL_INVALID", 2: "FEASIBLE", 3: "INFEASIBLE", 4: "OPTIMAL"}


import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oldproto import old_classes  # noqa: E402
from cpsolve import upgrade  # noqa: E402
from google.protobuf import text_format  # noqa: E402
OldM, OldR = old_classes()


def load(key):
    om = OldM(); om.ParseFromString(open(key + ".req", "rb").read())
    m = cp_model_pb2.CpModelProto(); text_format.Parse(upgrade(text_format.MessageToString(om)), m)
    r = OldR(); r.ParseFromString(open(key + ".resp", "rb").read())
    return m, r, float(open(key + ".us").read()) / 1e6


def solve(m, workers, limit, extra=""):
    s = cp_model.CpSolver(); p = s.parameters
    p.num_workers = workers; p.max_time_in_seconds = limit
    if extra:
        p.merge_text_format(extra) if hasattr(p, 'merge_text_format') else None
    model = cp_model.CpModel(); model.proto.parse_text_format(text_format.MessageToString(m))
    t = time.time(); st = s.Solve(model); dt = time.time() - t
    obj = s.ObjectiveValue() if m.HasField("objective") and st in (2, 4) else None
    return int(st), dt, obj


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cache"); a.add_argument("--top", type=int, default=20)
    a.add_argument("--configs", default="w1,w8,w16"); a.add_argument("--limit-x", type=float, default=1.0)
    a.add_argument("--out", default=None); a.add_argument("--info", action="store_true")
    o = a.parse_args()
    keys = sorted((float(open(f).read()), f[:-3]) for f in glob.glob(o.cache + "/*.us"))[::-1][:o.top]
    rows = []
    for t_h, k in keys:
        try:
            m, r, _ = load(k)
        except Exception as e:  # noqa: BLE001
            print("PARSE_FAIL", os.path.basename(k), e); continue
        hobj = r.objective_value if m.HasField("objective") else None
        line = [os.path.basename(k)[:16], f"{t_h / 1e6:.1f}", ST[r.status], f"{hobj}", f"vars={len(m.variables)} cons={len(m.constraints)} obj={m.HasField('objective')}"]
        if o.info:
            print("\t".join(line), "| hailo det_time %.1f conflicts %d" % (r.deterministic_time, r.num_conflicts)); continue
        for c in o.configs.split(","):
            w = int(c[1:]); st, dt, ob = solve(m, w, max(1.0, o.limit_x * t_h / 1e6))
            line.append(f"{c}:{ST[st][:4]}/{dt:.1f}s/{ob}")
        print("\t".join(line), flush=True); rows.append(line)
    if o.out:
        open(o.out, "w").write("\n".join("\t".join(x) for x in rows) + "\n")

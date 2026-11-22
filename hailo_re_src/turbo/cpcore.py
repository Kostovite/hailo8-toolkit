"""Minimal infeasible core of a dumped (infeasible) Hailo CP-SAT model: which constraints contradict each other.

Every constraint whose type takes an enforcement literal gets a fresh assumption literal; CP-SAT's sufficient
assumptions give a first core, then deletion shrinks it to a minimal one (removing any member makes it feasible).
Types without enforcement support (int_div, int_prod, all_diff) stay hard.

Usage: ortools_env/bin/python turbo/cpcore.py MODEL.pbtxt [--out core.txt]
"""
import argparse
import sys

from google.protobuf import text_format
from ortools.sat.python import cp_model

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cpsolve import load  # noqa: E402

ENFORCEABLE = {"linear", "bool_or", "bool_and", "table", "exactly_one", "at_most_one"}


def kind(c):
    return c.WhichOneof("constraint")


def build(m, active):
    """Model with constraint i kept only if i in `active` (None = all), as a CpModel."""
    keep = type(m)()
    keep.CopyFrom(m)
    del keep.constraints[:]
    for i, c in enumerate(m.constraints):
        if active is None or i in active or kind(c) not in ENFORCEABLE:
            keep.constraints.add().CopyFrom(c)
    keep.ClearField("objective")
    model = cp_model.CpModel()
    model.proto.parse_text_format(text_format.MessageToString(keep))
    return model


def feasible(m, active, limit=30.0):
    s = cp_model.CpSolver()
    s.parameters.num_workers = 8
    s.parameters.max_time_in_seconds = limit
    st = s.Solve(build(m, active))
    if st == cp_model.UNKNOWN:
        raise RuntimeError("solver timed out: core undecided")
    return st in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def first_core(m):
    """Sufficient assumptions for infeasibility (one literal per enforceable constraint)."""
    p = type(m)()
    p.CopyFrom(m)
    p.ClearField("objective")
    n = len(p.variables)
    lits = {}
    for i, c in enumerate(p.constraints):
        if kind(c) in ENFORCEABLE:
            v = p.variables.add()
            v.domain.extend([0, 1])
            c.enforcement_literal.append(n)
            lits[n] = i
            n += 1
    p.assumptions.extend(lits)
    model = cp_model.CpModel()
    model.proto.parse_text_format(text_format.MessageToString(p))
    s = cp_model.CpSolver()
    s.parameters.num_workers = 1
    st = s.Solve(model)
    assert st == cp_model.INFEASIBLE, s.StatusName(st)
    return sorted(lits[v] for v in s.SufficientAssumptionsForInfeasibility())


def minimise(m, core):
    """Deletion-based: drop each member; keep the drop if still infeasible."""
    core = list(core)
    i = 0
    while i < len(core):
        trial = set(core[:i] + core[i + 1:])
        if not feasible(m, trial):
            core = core[:i] + core[i + 1:]
        else:
            i += 1
    return core


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out")
    a = ap.parse_args()
    m = load(a.model)
    assert not feasible(m, None), "model is feasible"
    hard = [i for i, c in enumerate(m.constraints) if kind(c) not in ENFORCEABLE]
    if not feasible(m, set()):
        print(f"infeasible from the {len(hard)} hard (non-enforceable) constraints alone")
        core = []
    else:
        core = first_core(m)
        print(f"first core: {len(core)} constraints")
        core = minimise(m, core)
    print(f"minimal core: {len(core)} constraints")
    text = "\n".join(f"#{i} {kind(m.constraints[i])}: " + text_format.MessageToString(m.constraints[i], as_one_line=True)
                     for i in core)
    print(text[:6000])
    if a.out:
        open(a.out, "w").write(text + "\n")

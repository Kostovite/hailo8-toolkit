"""phases.py SAMPLES.txt [SAMPLES2.txt ...] : inclusive time per NAMED function in gsample.sh stacks.

A stripped frame is named by the C++ __PRETTY_FUNCTION__ strings its function references (assertion messages such as
"SharedStatus<allocator::ContextSplit> allocator::Slotter::solve_and_validate_partition(...)"); functions that
reference none get the name of the nearest named caller + "/anon". Inclusive = share of samples whose stack contains
the function at least once. Also prints the leaf-function (self time) top list.
"""
import collections
import re
import sys

sys.argv, files = sys.argv[:1], sys.argv[1:]
import fnmap_lib as F  # noqa: E402

PRETTY = re.compile(r"((?:[A-Za-z_]\w*::)+~?[A-Za-z_]\w*)\(")
stacks = []
for f in files:
    for line in open(f):
        fr = []
        for x in line.split():
            try:
                fr.append(int(x, 16))
            except ValueError:
                pass
        if fr:
            stacks.append(fr)
names = {}


def name(off):
    fn = F.func(off)
    if not fn:
        return None
    if fn[0] in names:
        return names[fn[0]]
    best = None
    for s in F.strings_of(fn)[0]:
        m = PRETTY.search(s)
        if m and ("allocator" in m.group(1) or "network_graph" in m.group(1) or "hw_graph" in m.group(1)
                  or "operations_research" in m.group(1) or "compiler" in m.group(1) or "utils" in m.group(1)):
            best = m.group(1); break
    names[fn[0]] = best or f"fn_{fn[0]:x}"
    return names[fn[0]]


incl = collections.Counter(); leaf = collections.Counter()
for st in stacks:
    nm = []
    for off in st:
        n = name(off)
        if n:
            nm.append(n)
    if not nm:
        continue
    # anon frames inherit the nearest named caller (stack is leaf first)
    lab = []
    for i, n in enumerate(nm):
        if n.startswith("fn_"):
            caller = next((m for m in nm[i + 1:] if not m.startswith("fn_")), "?")
            n = caller + "/anon"
        lab.append(n)
    for n in set(lab):
        incl[n] += 1
    leaf[lab[0]] += 1
N = len(stacks)
print(f"{N} samples")
print("== inclusive (named, >= 5%)")
for n, c in incl.most_common():
    if c / N >= 0.05 and not n.endswith("/anon"):
        print(f"{100 * c / N:5.1f}%  {n}")
print("== self (leaf) top 15")
for n, c in leaf.most_common(15):
    print(f"{100 * c / N:5.1f}%  {n}")

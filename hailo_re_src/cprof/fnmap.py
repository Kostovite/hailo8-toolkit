"""Stripped-binary function naming: .eh_frame FDE ranges + the C strings each function references (rip-relative lea).

Usage: python fnmap.py BINARY fde_ranges.txt OFFSET... -> function start, size, referenced strings
"""
import bisect
import re
import subprocess
import sys

B, fde = sys.argv[1], sys.argv[2]
R = sorted((int(a, 16), int(b, 16)) for a, b in (l.split() for l in open(fde)))
starts = [r[0] for r in R]
data = open(B, "rb").read()


def func(off):
    i = bisect.bisect_right(starts, off) - 1
    return R[i] if i >= 0 and R[i][0] <= off < R[i][1] else None


def cstr(addr, n=90):
    if not (0 <= addr < 0xc96ccc):  # first LOAD (r--): rodata
        return None
    end = data.find(b"\0", addr, addr + 400)
    s = data[addr:end if end > 0 else addr + n]
    if len(s) < 4 or not all(32 <= c < 127 or c in (9, 10) for c in s):
        return None
    return s[:n].decode()


def strings_of(r):
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", f"--start-address={r[0]}", f"--stop-address={r[1]}", B],
                         capture_output=True, text=True).stdout
    found = []
    for m in re.finditer(r"lea\s+-?0x[0-9a-f]+\(%rip\),%\w+\s+# ([0-9a-f]+)", out):
        s = cstr(int(m.group(1), 16))
        if s and s not in found:
            found.append(s)
    calls = len(re.findall(r"\bcall", out))
    return found, calls


for o in sys.argv[3:]:
    off = int(o, 16)
    r = func(off)
    if not r:
        print(f"0x{off:x}: no FDE")
        continue
    s, calls = strings_of(r)
    print(f"0x{off:x} in fn 0x{r[0]:x} (+{r[1] - r[0]} B, {calls} calls): {s[:6]}")

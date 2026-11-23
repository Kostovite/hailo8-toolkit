"""fnmap.py helpers as a module (func, cstr, strings_of) for the compiler binary."""
import bisect
import re
import subprocess
import sys

B, fde = "/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/hailo_tools/build/compiler", "/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/fde_ranges.txt"
R = sorted((int(a, 16), int(b, 16)) for a, b in (l.split() for l in open(fde)))
starts = [r[0] for r in R]
data = open(B, "rb").read()


def func(off):
    i = bisect.bisect_right(starts, off) - 1
    return R[i] if i >= 0 and R[i][0] <= off < R[i][1] else None


def cstr(addr, n=400):
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



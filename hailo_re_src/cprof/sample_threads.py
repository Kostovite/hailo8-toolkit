"""Sample a running process tree every DT s: per-thread CPU (utime+stime delta), thread names, wchan -> TSV.

Usage: python sample_threads.py ROOT_PID OUT.tsv [DT]   (follows children: the Hailo `compiler` subprocess)
"""
import os
import sys
import time

root, out = int(sys.argv[1]), sys.argv[2]
dt = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
hz = os.sysconf("SC_CLK_TCK")


def tree(pid):
    pids = [pid]
    try:
        for c in open(f"/proc/{pid}/task/{pid}/children").read().split():
            pids += tree(int(c))
    except OSError:
        pass
    return pids


prev = {}
with open(out, "w") as f:
    f.write("t\tpid\tcomm\tnthreads\tbusy_threads\tcpu_cores\ttop_threads\n")
    t0 = time.time()
    while os.path.exists(f"/proc/{root}"):
        now = time.time()
        for pid in tree(root):
            try:
                comm = open(f"/proc/{pid}/comm").read().strip()
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                continue
            tot, busy, per = 0.0, 0, []
            for tid in tids:
                try:
                    st = open(f"/proc/{pid}/task/{tid}/stat").read()
                    name = open(f"/proc/{pid}/task/{tid}/comm").read().strip()
                except OSError:
                    continue
                fl = st[st.rindex(")") + 2:].split()
                ticks = int(fl[11]) + int(fl[12])
                key = (pid, tid)
                if key in prev:
                    d = (ticks - prev[key][0]) / hz / (now - prev[key][1])
                    tot += d
                    busy += d > 0.5
                    if d > 0.05:
                        per.append((d, f"{name}:{tid}"))
                prev[key] = (ticks, now)
            per.sort(reverse=True)
            f.write(f"{now - t0:.0f}\t{pid}\t{comm}\t{len(tids)}\t{busy}\t{tot:.2f}\t"
                    f"{' '.join(f'{n}={d:.2f}' for d, n in per[:6])}\n")
        f.flush()
        time.sleep(dt)

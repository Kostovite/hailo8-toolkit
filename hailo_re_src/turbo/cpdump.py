"""Keep every CP-SAT model the Hailo compiler solves. SolveCpModel() overwrites FLAGS_cp_model_dump_file with each model
(text CpModelProto); this polls that file and copies each new, size-stable version to DIR/NNNN.pbtxt.

Usage: python cpdump.py DUMP_FILE DIR [PID]     (stops when PID exits; run the compile with
       LD_PRELOAD=turbo/libhailo_turbo.so HAILO_GFLAGS="cp_model_dump_file=DUMP_FILE")
"""
import os
import shutil
import sys
import time


def main(dump, out, pid=None):
    os.makedirs(out, exist_ok=True)
    last, n = None, 0
    while True:
        alive = pid is None or os.path.exists(f"/proc/{pid}")
        try:
            st = os.stat(dump)
            sig = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            sig = None
        if sig and sig != last:
            time.sleep(0.05)
            st2 = os.stat(dump)
            if (st2.st_mtime_ns, st2.st_size) == sig:  # writer finished
                shutil.copyfile(dump, os.path.join(out, f"{n:04d}.pbtxt"))
                n += 1
                last = sig
                continue
        if not alive:
            break
        time.sleep(0.01)
    print(f"kept {n} models in {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)

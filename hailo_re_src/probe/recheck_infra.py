"""Re-run the bit-exact check (chip lock held) for fuzz cuts whose check died in the device-leak window (fps 0)."""
import fcntl
import json
import os
import subprocess
import sys

R = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"
out = open(sys.argv[1], "a")
for f in sys.argv[2:]:
    run = os.path.dirname(f)
    for l in open(f):
        r = json.loads(l)
        for c in r.get("cuts", []):
            if not (c.get("ok") and c.get("fps", {}).get("1", 1) == 0):
                continue
            hef = f"{run}/{r['case']}/{c['tag']}/work/best/full.hef"
            ref = f"{run}/{r['case']}/{r['case']}/m.hef"
            lk = open(f"{R}/turbo/.chip_" + os.environ.get("DEVICE", "0000:03:00.0").replace(":", "_") + ".lock", "w")
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                o = subprocess.run([sys.executable, f"{R}/cprof/port/exact_check.py", ref, hef], capture_output=True,
                                   text=True, env=dict(os.environ, DEVICE=os.environ.get("DEVICE", "0000:03:00.0")))
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
            txt = o.stdout + o.stderr[-300:]
            res = "EXACT" if "BIT-EXACT" in txt and "DIFF" not in txt.upper() else ("DIFF" if "DIFF" in txt.upper() else "ERR")
            rec = {"case": r["case"], "tag": c["tag"], "at": c["at"], "res": res, "tail": txt.strip()[-160:]}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(rec["case"], rec["tag"], rec["at"], res, flush=True)

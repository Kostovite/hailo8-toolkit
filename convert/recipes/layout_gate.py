"""layout_gate.py CKPT OUT_DIR PREFIX|ref : box AP via preproc_hailo/layout_gate.py score (host decoder on chip-part features).
'ref' scores the torch features (FP baseline, saved to host/fp_ap.json). PASS: box AP drop <= MAX_DROP (.005) on every set."""
import os, sys, json, subprocess, ast
ck, out, pre = sys.argv[1:4]
r = subprocess.run([sys.executable, "-u", "/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/layout_gate.py", "score", ck, f"{out}/{pre}"], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "")
line = [l for l in r.stdout.splitlines() if "(AP, box AP, box AP75):" in l][-1]; ap = ast.literal_eval(line.split(":", 1)[1].strip())
if pre == "ref": json.dump(ap, open(f"{out}/host/fp_ap.json", "w")); print("FP baseline", ap); sys.exit(0)
fp = json.load(open(f"{out}/host/fp_ap.json")); lim = float(os.environ.get("MAX_DROP", ".005")); ok = True
for k, v in ap.items():
    d = fp[k][1] - v[1]; ok &= d <= lim; print(f"RESULT {k} box AP fp {fp[k][1]:.4f} {pre} {v[1]:.4f} drop {d:+.4f}  AP75 {fp[k][2]:.4f} -> {v[2]:.4f}")
print("GATE", "PASS" if ok else "FAIL", "max drop", lim); sys.exit(0 if ok else 2)

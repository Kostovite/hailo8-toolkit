"""router_gate.py OUT_DIR PREFIX : decision agreement per head vs torch FP; PASS needs >= MIN_AGREE (default .99) on every head"""
import sys, os, numpy as np
out, pre = sys.argv[1:3]; th = float(os.environ.get("MIN_AGREE", ".99")); ok = True
for i, k in enumerate(("lang", "kind", "body")):
    a = np.load(f"{out}/ref_{i}.npy").reshape(-1, {"lang": 21, "kind": 6, "body": 3}[k]); b = np.load(f"{out}/{pre}_{i}.npy").reshape(len(a), -1)
    ag = float((a.argmax(1) == b.argmax(1)).mean()); ok &= ag >= th
    print(f"RESULT {k} n {len(a)} agreement {ag:.4f} logit_mae {np.abs(a - b).mean():.4f}")
print("GATE", "PASS" if ok else "FAIL", "min", th); sys.exit(0 if ok else 2)

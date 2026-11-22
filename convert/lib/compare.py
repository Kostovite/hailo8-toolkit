"""compare.py REF_PREFIX TEST_PREFIX TOL : per output max|a-b| / max|a| ; exit 2 if any > TOL. Shapes compared after squeeze."""
import sys, glob, numpy as np
a_p, b_p, tol = sys.argv[1], sys.argv[2], float(sys.argv[3]); bad = 0; i = 0
while glob.glob(f"{a_p}_{i}.npy"):
    a = np.load(f"{a_p}_{i}.npy", mmap_mode="r"); b = np.load(f"{b_p}_{i}.npy", mmap_mode="r")
    n = min(len(a), len(b)); a2, b2 = a[:n].reshape(n, -1), b[:n].reshape(n, -1); assert a2.shape == b2.shape, (i, a.shape, b.shape)
    num = den = mae = 0.0
    for j in range(0, len(a2), 64):
        x, y = np.asarray(a2[j:j+64], np.float64), np.asarray(b2[j:j+64], np.float64)
        num = max(num, np.abs(x - y).max()); den = max(den, np.abs(x).max()); mae += np.abs(x - y).sum()
    r = num / max(den, 1e-12); print(f"output {i} {a.shape}: rel_max {r:.2e}  mae {mae / a2.size:.3e}"); bad += r > tol; i += 1
assert i, "no outputs"; print("GATE_FP", "FAIL" if bad else "PASS", "tol", tol); sys.exit(2 if bad else 0)

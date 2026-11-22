"""capture_probes.py ROOT_DIR... -> for every ROOT/CASE/q.har: CASE.hn, cap_in.npy (NHWC), gcap/ (ALL_OPS HSim capture +
numpy op pickle). DFC env, CPU. One case at a time; skips cases already captured. Prints one status line per case."""
import enum, json, os, pickle, subprocess, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import probe  # noqa: E402
from hailo_sdk_client import ClientRunner  # noqa: E402
def conv(v):
    if isinstance(v, dict): return {str(k): conv(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return type(v)(conv(x) for x in v)
    if isinstance(v, enum.Enum): return conv(v.value)
    if isinstance(v, (bool, int, float, str)) or v is None: return v
    if isinstance(v, np.ndarray): return np.array([conv(x) for x in v.ravel()], dtype=object).reshape(v.shape) if v.dtype == object else np.array(v)
    if isinstance(v, np.generic): return v.item()
    try: return np.asarray(v)
    except Exception: return str(v)
for root in sys.argv[1:]:
    for har in sorted(glob.glob(f"{root}/*/q.har")):
        d = os.path.dirname(har); n = os.path.basename(root) + "__" + os.path.basename(d)
        if os.path.exists(f"{d}/gcap/_all_ops_np.pkl"): print(n, "cached"); continue
        try:
            h = ClientRunner(har=har).get_hn(); h = json.loads(h) if isinstance(h, str) else h; json.dump(h, open(f"{d}/case.hn", "w"))
            ishape = [v for v in h["layers"].values() if v["type"] == "input_layer"][0]["input_shapes"][0][1:]
            dom = open(f"{d}/dom.txt").read().strip() if os.path.exists(f"{d}/dom.txt") else "normal"
            x = np.asarray(probe.domain(dom, 1, np.random.default_rng(5)), np.float32)
            x = x.reshape([1] + list(ishape)) if x.size == int(np.prod(ishape)) else x
            np.save(f"{d}/cap_in.npy", x.astype(np.float64)); np.save(f"{d}/cap_in_chw.npy", x[0].transpose(2, 0, 1))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="99", ALL_OPS="1", AUTO_ATTRS="1")
            r = subprocess.run([sys.executable, "capture.py", har, f"{d}/cap_in_chw.npy", f"{d}/gcap"], cwd=f"{Q}/hailoqat", env=env, capture_output=True, text=True)
            if r.returncode: print(n, "CAPTURE_FAIL", r.stderr.strip().splitlines()[-1][:150] if r.stderr.strip() else ""); continue
            dd = pickle.load(open(f"{d}/gcap/_all_ops.pkl", "rb")); pickle.dump({str(k): conv(v) for k, v in dd.items()}, open(f"{d}/gcap/_all_ops_np.pkl", "wb"))
            os.remove(f"{d}/gcap/_all_ops.pkl"); print(n, "ok", ishape, dom, flush=True)
        except Exception as e:  # noqa: BLE001
            print(n, "FAIL", type(e).__name__, str(e)[:150], flush=True)

"""GPU replacement for the DFC quantized emulator: the hailo8-exact-qat executor (QATNet forward = the chip's integer
arithmetic), fed from a one-input HSim capture of the same HAR.
usage (DFC env): gpu_emu.py capture HAR EVAL_X CAP_DIR            -> CAP_DIR (+ CAP_DIR.hn), runs bias_lattice for 16-bit
                 gpu_emu.py run    CAP_DIR EVAL_X OUT_PREFIX [BS]  -> OUT_PREFIX_<i>.npy float, outputs ordered by size
Exactness: QATNet raises on its own exactness flags; validate once per model against the real chip (lib/chip.py)."""
import os, sys, json, subprocess, numpy as np
QAT = "/mnt/storage/lib_project/hailo8-exact-qat"; sys.path.insert(0, QAT + "/hailoqat")
cmd = sys.argv[1]
if cmd == "capture":
    har, ev, cap = sys.argv[2:5]
    from hailo_sdk_client import ClientRunner
    h = ClientRunner(har=har).get_hn(); h = json.loads(h) if isinstance(h, str) else h; json.dump(h, open(cap + ".hn", "w"))
    x = np.load(ev, mmap_mode="r")[0]; x = np.asarray(x)
    inp = cap + ".input.npy"; np.save(inp, x.transpose(2, 0, 1).astype(np.float32))   # capture.py takes CHW .npy
    np.save(cap + '.input_nhwc.npy', x[None].astype(np.float64))   # graph26 takes the NHWC batch capture.py fed
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="99", ALL_OPS="1", AUTO_ATTRS="1")
    r = subprocess.run([sys.executable, "-u", "capture.py", har, inp, cap], cwd=QAT + "/hailoqat", env=env)
    if r.returncode: sys.exit("capture.py failed")
    import enum, pickle
    def conv(v):  # hailo enums subclass str/int and need the SDK to unpickle (tools/har_pipeline.sh step 3)
        if isinstance(v, dict): return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, list): return [conv(x) for x in v]
        if isinstance(v, tuple): return tuple(conv(x) for x in v)
        if isinstance(v, enum.Enum): return conv(v.value)
        if isinstance(v, bool): return bool(v)
        if isinstance(v, int): return int(v)
        if isinstance(v, float): return float(v)
        if isinstance(v, str): return str(v)
        if v is None: return None
        if isinstance(v, np.ndarray): return np.array([conv(q) for q in v.ravel()], dtype=object).reshape(v.shape) if v.dtype == object else np.array(v)
        if isinstance(v, np.generic): return v.item()
        try: return np.asarray(v)
        except Exception: return str(v)
    d = pickle.load(open(f"{cap}/_all_ops.pkl", "rb")); pickle.dump({str(k): conv(v) for k, v in d.items()}, open(f"{cap}/_all_ops_np.pkl", "wb"))
    r = subprocess.run([sys.executable, "-u", "hailoqat/bias_lattice.py", har, cap], cwd=QAT, env=env)
    if r.returncode: sys.exit("bias_lattice failed")
    print("CAPTURED", cap, "ops", len(d)); sys.exit()
import torch, graph26, qatfast
cap, ev, pre = sys.argv[2:5]; bs = int(sys.argv[5]) if len(sys.argv) > 5 else 8
dev = torch.device("cuda"); hn = json.load(open(cap + ".hn")); P = graph26.Params(cap, dev)
L = hn["layers"]
try:
    net = qatfast.QATNet(hn, P, dev).to(dev).eval(); impl = "qatfast"
except (AssertionError, NotImplementedError) as e:  # e.g. a16_w16 conv with fused add: exact float64 executor instead
    print(f"[gpu_emu] qatfast cannot build ({str(e)[:80]}) -> graph26 batched float64 executor", flush=True); net = None; impl = "graph26"
def forward(xb):
    if net is not None:
        o = net(torch.from_numpy(xb).to(dev))
        if net.flag_events: raise SystemExit(f"exactness flags {net.flag_events[:5]}")
        return o
    return graph26.run(hn, P, xb.astype(np.float64), dev, check=False)[0]
outs = [L[n]["input"][0] for n in L if L[n]["type"] == "output_layer"]
def deq(name, q):
    r = P.conv.get(name)
    if r is None or "out_scale" not in r: raise SystemExit(f"no output scale recorded for {name}")
    sc = torch.as_tensor(np.asarray(r["out_scale"], np.float64), device=dev).reshape(1, -1, 1, 1)
    zp = torch.as_tensor(np.asarray(r["out_zp"], np.float64), device=dev).reshape(1, -1, 1, 1)
    return ((q.double() - zp) * sc).permute(0, 2, 3, 1).float().cpu().numpy()
x = np.load(ev, mmap_mode="r"); N = len(x); files = None; f16 = os.environ.get("EMU_F16") == "1"
with torch.no_grad():
    for i in range(0, N, bs):
        xb = np.asarray(x[i:i + bs], np.float32)
        o = forward(xb)
        if i == 0 and len(xb) > 1:  # batch self-test: batched integers must equal one-at-a-time integers
            for j in range(min(len(xb), 3)):
                oj = forward(xb[j:j + 1])
                for n in outs:
                    if not torch.equal(oj[n][0].double(), o[n][j].double()):
                        raise SystemExit(f"BATCH SELF-TEST FAILED at {n} row {j}: batched != single")
            print("batch self-test passed", flush=True)
        ys = [deq(n, o[n]) for n in outs]
        if files is None:
            sz = [y[0].size for y in ys]; assert len(set(sz)) == len(sz), sz
            order = sorted(range(len(ys)), key=lambda k: -sz[k])
            files = [np.lib.format.open_memmap(f"{pre}_{j}.npy", "w+", np.float16 if f16 else np.float32, (N,) + ys[k].shape[1:]) for j, k in enumerate(order)]
            print("impl", impl, "outputs", [outs[k] for k in order], flush=True)
        for j, k in enumerate(order): files[j][i:i + len(ys[k])] = ys[k]
        if i % (bs * 25) == 0: print(f"{i + len(xb)}/{N}", flush=True)
[f.flush() for f in files]; print("saved", pre, [f.shape for f in files])

"""Frame contamination / batch-dependence scan: every per-frame output must be identical whatever the frame order,
the frames around it and the configured batch size. Usage: python contam.py HEF_LIST OUT.jsonl (DEVICE env)"""
import fcntl, json, os, sys
import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams, InputVStreamParams,
                            OutputVStreamParams, VDevice)
dev = os.environ.get("DEVICE", "0000:02:00.0")
lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w")
done = set()
if os.path.exists(sys.argv[2]):
    done = {json.loads(l)["hef"] for l in open(sys.argv[2])}
out = open(sys.argv[2], "a")


def frames(hef, n, seed):
    rng = np.random.default_rng(seed); X = {}
    for i in hef.get_input_vstream_infos():
        u16 = "16" in str(i.format.type); hi = 65535 if u16 else 255; q = i.quant_info
        X[i.name] = np.clip(np.round(rng.normal(0, 1, (n, *i.shape)) / q.qp_scale + q.qp_zp), 0, hi).astype(np.uint16 if u16 else np.uint8)
    return X


def infer(vd, hef, X, batch):
    cp = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    for k in cp: cp[k].batch_size = batch
    ng = vd.configure(hef, cp)[0]
    u16 = any(x.dtype == np.uint16 for x in X.values())
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT16 if u16 else FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.AUTO)
    try:
        with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
            return {k: np.asarray(v).copy() for k, v in pipe.infer(X).items()}
    finally:
        vd.release_configured_network_groups() if hasattr(vd, "release_configured_network_groups") else None


for h in [l.strip() for l in open(sys.argv[1]) if l.strip()]:
    if h in done: continue
    rec = {"hef": h}
    try:
        hef = HEF(h); X = frames(hef, 16, 0); Rn = frames(hef, 16, 1)
        fcntl.flock(lk, fcntl.LOCK_EX); p = VDevice.create_params(); p.device_ids = [dev]
        res = {}
        try:
            for b in (1, 8, 16):
                vd = VDevice(p)
                try:
                    base = infer(vd, hef, X, b)
                    rev = {k: v[::-1] for k, v in infer(vd, hef, {k: v[::-1] for k, v in X.items()}, b).items()}
                    mix = {k: v[0::2] for k, v in infer(vd, hef, {k: np.stack([X[k], Rn[k]], 1).reshape(32, *X[k].shape[1:]) for k in X}, b).items()}
                finally:
                    vd.release()
                res[b] = (base, rev, mix)
            vd = VDevice(p)
            try:
                single = [infer(vd, hef, {k: v[j:j + 1] for k, v in X.items()}, 1) for j in range(4)]
            finally:
                vd.release()
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
        ref = res[1][0]; bad = {}
        for b, (base, rev, mix) in res.items():
            for nm, o in (("base", base), ("reversed", rev), ("interleaved", mix)):
                n = sum(int((o[k] != ref[k]).sum()) for k in ref)
                if n: bad[f"b{b}_{nm}"] = n
        for j, s in enumerate(single):
            n = sum(int((s[k][0] != ref[k][j]).sum()) for k in ref)
            if n: bad[f"single{j}"] = n
        rec.update(ok=True, bad=bad, contexts=None)
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:200]}")
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(os.path.relpath(h, "/mnt/storage/lib_project/models/Hailo_model/hailo_re"), "OK" if rec.get("ok") and not rec["bad"] else rec.get("bad") or rec.get("err"), flush=True)

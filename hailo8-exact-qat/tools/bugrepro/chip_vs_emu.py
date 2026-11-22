"""Silent-wrong sweep: each stock probe HEF on the chip vs the SDK_BIT_EXACT emulator of its own quantized HAR, on
the same raw uint8 frames (outputs requantized to the HEF's raw format). Large n_diff / low corr = the chip computes
something other than what every simulator says.

Usage (DFC env): python chip_vs_emu.py emu PAR DIRS.txt     (CPU, parallel: DIR/emu_ref.npz)
                 python chip_vs_emu.py chip OUT.jsonl DIRS.txt (one chip, serial)
DIR holds q.har + m.hef (a sweep.py / zoo/ops case dir).
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/port")
import exact_check as ec  # noqa: E402
from hailo_platform import HEF, VDevice  # noqa: E402
from hailo_sdk_client import ClientRunner, InferenceContext  # noqa: E402

N = 8
DIST = os.environ.get("DIST", "uniform")  # uniform raw bytes, or "normal": quantized N(0,1) = the probes' calibration
REF = "emu_ref.npz" if DIST == "uniform" else "emu_ref_n.npz"


def frames(hef):  # raw frames in each input's own format (UINT8 or UINT16)
    rng = np.random.default_rng(0)
    out = {}
    for i in hef.get_input_vstream_infos():
        u16 = "16" in str(i.format.type)
        hi = 65535 if u16 else 255
        if DIST == "normal":
            q = np.round(rng.normal(0, 1, (N, *i.shape)) / i.quant_info.qp_scale + i.quant_info.qp_zp)
            out[i.name] = np.clip(q, 0, hi).astype(np.uint16 if u16 else np.uint8)
        else:
            out[i.name] = rng.integers(0, hi + 1, (N, *i.shape), dtype=np.uint16 if u16 else np.uint8)
    return out


def run16(vd, hef, X):
    """exact_check.run with the input format given explicitly: FormatType.AUTO on a UINT16 input rejects uint16
    arrays ("numpy array item size is 2, not 1"; HailoRT 4.23)."""
    from hailo_platform import (ConfigureParams, FormatType, HailoStreamInterface, InferVStreams, InputVStreamParams,
                                OutputVStreamParams)
    ng = vd.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    u16 = any(x.dtype == np.uint16 for x in X.values())
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT16 if u16 else FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.AUTO)
    with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
        return pipe.infer(X)


def emu(d):
    try:
        hef = HEF(f"{d}/m.hef")
        ins = hef.get_input_vstream_infos()
        X = frames(hef)
        runner = ClientRunner(har=f"{d}/q.har")
        feed = [(X[i.name].astype(np.float32) - i.quant_info.qp_zp) * i.quant_info.qp_scale for i in ins]
        with runner.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
            e = runner.infer(ctx, feed[0] if len(feed) == 1 else feed)
        e = e if isinstance(e, list) else [e]
        np.savez(f"{d}/{REF}", *[np.asarray(x) for x in e])
        return d, "ok"
    except Exception as ex:  # noqa: BLE001
        return d, f"{type(ex).__name__}: {str(ex)[:150]}"


def one(vd, d):
    hef = HEF(f"{d}/m.hef")
    outs = hef.get_output_vstream_infos()
    z = np.load(f"{d}/{REF}")
    e = [z[f"arr_{i}"] for i in range(len(z.files))]
    X = frames(hef)
    chip = run16(vd, hef, X)
    rec = {"dir": d, "outs": []}
    for o, ef in zip(outs, e):
        c = np.asarray(chip[o.name]).astype(np.int64).reshape(N, -1)
        hi = 65535 if "16" in str(o.format.type) else 255
        q = np.clip(np.round(np.asarray(ef) / o.quant_info.qp_scale + o.quant_info.qp_zp), 0, hi).astype(np.int64)
        q = q.reshape(N, -1)
        if q.shape != c.shape:
            rec["outs"].append({"name": o.name, "err": f"shape {q.shape} vs {c.shape}"})
            continue
        dd = np.abs(c - q)
        cc = np.corrcoef(c.ravel(), q.ravel())[0, 1] if c.std() and q.std() else float("nan")
        rec["outs"].append({"name": o.name, "frac": round(float((dd > 0).mean()), 4), "max": int(dd.max()),
                            "p99": int(np.percentile(dd, 99)), "corr": round(float(cc), 4),
                            "f0_only": bool(dd[1:].max() == 0 and dd[0].max() > 0)})
    return rec


if __name__ == "__main__":
    dirs = [x.strip() for x in open(sys.argv[3]) if x.strip()] if sys.argv[1] != "emu1" else []
    if sys.argv[1] == "emu1":  # one dir, own process (the SDK forks; a multiprocessing pool can't host it)
        print(*emu(sys.argv[2])[::-1])
        sys.exit()
    if sys.argv[1] == "emu":
        import subprocess
        from concurrent.futures import ThreadPoolExecutor

        def sub(d):
            p_ = subprocess.run([sys.executable, __file__, "emu1", d, "-"], capture_output=True, text=True)
            return (p_.stdout.strip().splitlines() or [p_.stderr.strip()[-150:]])[-1]
        with ThreadPoolExecutor(int(sys.argv[2])) as ex:
            for st in ex.map(sub, [d for d in dirs if not os.path.exists(f"{d}/{REF}")]):
                print(st[:200], flush=True)
        sys.exit()
    import fcntl
    p = VDevice.create_params()
    p.device_ids = [os.environ.get("DEVICE", "0000:02:00.0")]
    with open(sys.argv[2], "a") as f:
        for d in [d for d in dirs if os.path.exists(f"{d}/{REF}")]:
            lk = open("/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_" + os.environ.get("DEVICE", "0000:02:00.0").replace(":", "_") + ".lock", "w")
            fcntl.flock(lk, fcntl.LOCK_EX)  # the same lock hcc takes: never share a measurement device
            vd = None
            try:
                vd = VDevice(p)
                r = one(vd, d)
            except Exception as ex:  # noqa: BLE001
                r = {"dir": d, "err": f"{type(ex).__name__}: {str(ex)[:200]}"}
            finally:  # release BEFORE unlocking: a leaked VDevice blocks every later user (HAILO_DEVICE_IN_USE)
                if vd is not None:
                    vd.release()
                fcntl.flock(lk, fcntl.LOCK_UN)
                lk.close()
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(json.dumps(r)[:300], flush=True)

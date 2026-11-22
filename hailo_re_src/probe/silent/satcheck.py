"""Uniform-input chip vs SDK_BIT_EXACT: are the mismatches at saturated emulator values (range clipping modelled
differently) or in-range (different arithmetic)? Usage: python satcheck.py DIR [DIR...]  (DEVICE env, chip lock)"""
import fcntl, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DIST", "uniform")
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
dev = os.environ.get("DEVICE", "0000:03:00.0")
lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w")
for d in sys.argv[1:]:
    hef = HEF(f"{d}/m.hef"); o = hef.get_output_vstream_infos()[0]; q = o.quant_info
    z = np.load(f"{d}/emu_ref.npz")["arr_0"]
    X = cv.frames(hef)
    fcntl.flock(lk, fcntl.LOCK_EX)
    p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
    try: c = np.asarray(next(iter(cv.run16(vd, hef, X).values()))).astype(np.int64)
    finally: vd.release(); fcntl.flock(lk, fcntl.LOCK_UN)
    hi = 65535 if c.dtype == np.uint16 or c.max() > 255 else 255
    E = np.round(z / q.qp_scale + q.qp_zp).reshape(c.shape)
    Ec = np.clip(E, 0, hi).astype(np.int64)
    bad = c != Ec
    sat = (E <= 0) | (E >= hi)
    csat = (c == 0) | (c == hi)
    print(f"{d.split('/ops/')[-1]}: out max {hi} mismatch {bad.mean():.4f} | emu saturated {sat.mean():.3f} -> "
          f"mismatch among sat {bad[sat].mean() if sat.any() else 0:.3f} / among in-range {bad[~sat].mean():.3f} | "
          f"chip saturated where emu in-range {(csat & ~sat).mean():.3f} | median |d| in-range "
          f"{np.median(np.abs(c - Ec)[bad & ~sat]) if (bad & ~sat).any() else 0}")

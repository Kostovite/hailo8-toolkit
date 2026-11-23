"""var.py NAME SCRIPT_LINES OUT: compile the real dw31 single-layer sub-HAR with extra model-script lines (';'-separated)."""
import json, os, sys, time, shutil
name, extra, out = sys.argv[1], sys.argv[2].replace(";", "\n"), sys.argv[3]
HAR = "/mnt/storage/lib_project/models/Hailo_model/convert/runs/layout_g3/grow_t2/spans/s214_215.har"
d = os.path.join(out, name); os.makedirs(d, exist_ok=True); os.chdir(d)
res = {"name": name, "extra": extra}; t0 = time.time()
try:
    from hailo_sdk_client import ClientRunner
    r = ClientRunner(har=HAR)
    r.load_model_script("context_switch_param(mode=disabled)\nperformance_param(fps=20)\n" + extra + "\n")
    hef = r.compile(); open("m.hef", "wb").write(hef); r.save_har("c.har"); res["ok"] = True
except Exception as e:
    res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
res["s"] = round(time.time() - t0, 1); json.dump(res, open("res.json", "w"))
print(json.dumps({k: res.get(k) for k in ("name", "ok", "s", "error")}), flush=True)

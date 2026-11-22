"""One context = one process: cut LAYERS out of HAR, compile it as a single context, profile -> OUT.json.

Usage (DFC env): python compile_sub.py FULL.har LAYERS.json OUT_PREFIX [EXTRA_ALLS]
  -> OUT_PREFIX.har (sub), OUT_PREFIX_compiled.har, OUT_PREFIX.json {ok, fps, latency_ms, compile_s, error}
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subhar  # noqa: E402



def silent_wrong(har):
    """The compiler's own choices that are known silent-wrong on the chip (RULES 26), read from the compiled HAR's
    auto script, as the lines that avoid them: an INPUT_FEATURES defuse of ANY dynamic matmul (P.V 1-6 dB, Q.K
    forced 96% wrong) -> pin its SCs; a DECONV lowering of a kernel == stride deconv (~100% wrong) -> SUPER_DECONV"""
    import re
    import tarfile
    with tarfile.open(har) as t:
        names = t.getnames()
        hn = json.loads(t.extractfile(next(x for x in names if x.endswith(".hn") and "native" not in x
                                           and ".fp." not in x)).read())
        auto = next((t.extractfile(x).read().decode() for x in names if x.endswith(".auto.alls")), "")
    L = {n.split("/")[-1]: v for n, v in hn["layers"].items()}
    fix = []
    for m in re.finditer(r"defuse\(([\w-]+), \d+, defuse_type=(INPUT_FEATURES|DECONV)\)", auto):
        v = L.get(m.group(1), {})
        if m.group(2) == "INPUT_FEATURES" and v.get("type") == "matmul" and v.get("params", {}).get("dynamic_weights"):
            fix.append(f"compilation_param({m.group(1)}, resources_allocation_strategy=manual_scs_selection, "
                       f"number_of_subclusters={8 if v['output_shapes'][0][2] > 400 else 2})")
        p = v.get("params", {})
        if m.group(2) == "DECONV" and list(p.get("kernel_shape", [0, 0])[:2]) == list(p.get("strides", [1, 0, 0, 1])[1:3]):
            fix.append(f"{m.group(1)}_defuse_conv, {m.group(1)}_defuse_d2s = defuse({m.group(1)}, 1, "
                       f"defuse_type=SUPER_DECONV)")
    return fix


full, layers, pre = (os.path.abspath(x) for x in sys.argv[1:4])
extra = open(sys.argv[4]).read() if len(sys.argv) > 4 else ""
os.makedirs(pre + ".d", exist_ok=True)  # private cwd: parallel compiles in one cwd clobber errors.pb / logs
os.chdir(pre + ".d")
res = {"layers": json.load(open(layers))}
t0 = time.time()
try:
    res.update(subhar.build(full, res["layers"], pre + ".har"))
    from hailo_sdk_client import ClientRunner
    r = ClientRunner(har=pre + ".har")
    if extra:
        r.load_model_script(extra)  # quantized HAR: append mode is refused; compiler commands only
    t = time.time()
    hef = r.compile()
    r.save_har(pre + "_compiled.har")
    fix = silent_wrong(pre + "_compiled.har")
    if fix:  # recompile once with the compiler's silent-wrong choices blocked; failing that the variant is dropped
        res["repair"] = fix
        os.remove(pre + "_compiled.har")
        r = ClientRunner(har=pre + ".har")
        r.load_model_script(extra + "\n".join(fix) + "\n")
        hef = r.compile()
        r.save_har(pre + "_compiled.har")
        left = silent_wrong(pre + "_compiled.har")
        if left:
            raise RuntimeError(f"silent-wrong lowering survived the repair: {left}")
    res["compile_s"] = round(time.time() - t, 1)
    try:
        prof = r.profile()
        pd = (prof[0] if isinstance(prof, tuple) else prof)["stats"]["performance_details"][0]
        res.update(ok=True, fps=pd["fps"], latency_ms=pd["latency"], contexts=pd["number_of_contexts"])
    except Exception as e:  # noqa: BLE001  the PROFILER can reject a graph the compiler accepted (SmolLM2:
        # a layer with 2 outputs feeding rotate_half + RoPE mult -> "Shapes ... [8,16,192] vs [8,8,192]");
        # measure the single-context HEF on the chip instead (per-chip lock, first device of HCC_DEVS)
        import re
        import subprocess
        open(pre + ".hef", "wb").write(hef)
        dev = os.environ.get("HCC_DEVS", "0000:02:00.0").split(",")[0]
        lock = f"{os.path.dirname(os.path.abspath(__file__))}/.chip_{dev.replace(':', '_')}.lock"
        out = subprocess.run(["flock", lock, "hailortcli", "run2", "-t", "2", "-s", dev, "set-net", pre + ".hef"],
                             capture_output=True, text=True).stdout
        fps = [float(x) for x in re.findall(r"fps:\s*([\d.]+)", out)]
        if not fps:
            raise RuntimeError(f"profile failed ({type(e).__name__}: {str(e)[:120]}) and chip run gave no fps")
        res.update(ok=True, fps=max(fps), latency_ms=1000 / max(fps), contexts=1, fps_from="chip",
                   profile_error=str(e)[:200])
except Exception as e:  # noqa: BLE001
    res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}", trace=traceback.format_exc()[-800:])
res["wall_s"] = round(time.time() - t0, 1)
json.dump(res, open(pre + ".json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("layers", "trace")}))

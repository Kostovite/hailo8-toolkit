"""subhar the attention block out of the shipped yolo26m q3a HAR, compile single-context with stock's recipe, save HEF."""
import os, sys
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo")
import json, subhar
tag = sys.argv[1]; d = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/runs/det/att"
os.makedirs(f"{d}/{tag}.d", exist_ok=True); os.chdir(f"{d}/{tag}.d")
lay = json.load(open(f"{d}/layers{tag}.json"))
print(subhar.build("/mnt/storage/lib_project/models/Hailo_model/yolo26m_v14sec_q3a_s2000.har", lay, f"{d}/{tag}.har"))
from hailo_sdk_client import ClientRunner
r = ClientRunner(har=f"{d}/{tag}.har")
rec = [l for l in open("/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/t/m_part_stock15/c1.extra.alls") if l.split("(")[1].split(",")[0] in lay]
r.load_model_script("".join(rec) + os.environ.get("EXTRA_ALLS", ""))
open(f"{d}/{tag}.hef", "wb").write(r.compile()); r.save_har(f"{d}/{tag}_compiled.har"); print("HEF ok", tag)

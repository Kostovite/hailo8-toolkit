"""Per-layer int8 sensitivity with the DFC's layer noise analysis.
Usage: noise_analysis.py QUANT_HAR OUT_JSON H W [N]  (TRAIN_GLOB selects pages; each layer quantized alone vs FP)"""
import glob, json, os, random, sys, time
import cv2
import numpy as np
from hailo_sdk_client import ClientRunner
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from letterbox import letterbox  # noqa: E402

har, out, H, W = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
n = int(sys.argv[5]) if len(sys.argv) > 5 else 32
files = sorted(glob.glob(os.environ["TRAIN_GLOB"]))
random.Random(1).shuffle(files)
x = np.stack([letterbox(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (H, W))[0] for f in files[:n]]).astype(np.float32)
runner = ClientRunner(har=har)
t = time.time()
runner.analyze_noise(x, batch_size=8, data_count=n, analyze_mode=os.environ.get("MODE", "simple"))
print(f"analyze_noise {time.time() - t:.0f} s")
runner.save_har(out.replace(".json", ".har"))

"""Exact-executor intermediates of yolo26m's first attention block for the chip-mismatch pages and good pages."""
import json, sys
import cv2, numpy as np, torch
import graph26, qatfast
R = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"
per = json.load(open(f"{R}/runs/det/chip0_val.npz.pages.json"))
files = list(np.load(f"{R}/runs/det/chip0_val.npz")["files"])
bad = list(per); good = [f for f in files if f not in per][::100][:8]
dev = torch.device("cuda"); hn = json.load(open(f"{R}/q3a_s2000_har.hn"))
net = qatfast.QATNet(hn, graph26.Params(f"{R}/capture_q3a_s2000_har", dev), dev).to(dev).eval()
keep = ["best/ew_mult_softmax1", "best/activation1", "best/reduce_max_softmax1", "best/ew_sub_softmax1", "best/reduce_sum_softmax1", "best/ew_add9", "best/conv_feature_splitter5_2", "best/conv_feature_splitter6_1", "best/conv_feature_splitter6_2", "best/matmul1",
        "best/matmul2", "best/conv104"]
res = {k: [] for k in keep}
for f in bad + good:
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32)[None]
    with torch.no_grad(): out = net(torch.from_numpy(img).to(dev))
    for k in keep:
        y = out[k]; y = y[0] if isinstance(y, (list, tuple)) else y
        res[k].append(y.permute(0, 2, 3, 1).cpu().numpy().astype(np.int32)[0])
np.savez(f"{R}/runs/det/att/ints2.npz", files=np.array(bad + good), nbad=len(bad), **{k.replace("/", "__"): np.stack(v) for k, v in res.items()})
print({k: np.stack(v).shape for k, v in res.items()}, "bad", len(bad))

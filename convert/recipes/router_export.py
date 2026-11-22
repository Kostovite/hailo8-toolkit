"""router_export.py CKPT TRAIN_FEATS OUT_DIR
Host contract: chip input = pooled_encoder / scale (per-feature, scale = 1.05 * max|x| over the training features).
The 1/scale is folded back into the first Linear, so chip(x/scale) == router(x). Writes model.onnx, host/router_scale.npy."""
import sys, numpy as np, torch, torch.nn as nn
sys.path[:0] = ["/mnt/storage/lib_project/models/Dolphin", "/mnt/storage/lib_project/models/Dolphin/wordlid"]
from region_router import RegionRouter
ck, feats, out = sys.argv[1:4]
net = RegionRouter(); net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["router"]); net.eval()
x = np.load(feats).astype(np.float32); s = (1.05 * np.abs(x).max(0)).clip(1e-3).astype(np.float32)
np.save(f"{out}/host/router_scale.npy", s)
class Chip(nn.Module):
    def __init__(s_, n, sc): super().__init__(); s_.n = n; s_.register_buffer("sc", torch.from_numpy(sc))
    def forward(s_, xn): r = s_.n(xn * s_.sc); return r["kind"], r["body"], r["lang"]
chip = Chip(net, s).eval(); xt = torch.from_numpy(x[:64])
with torch.no_grad():
    a = net(xt); b = chip(xt / chip.sc)
print("fold parity", [float((a[k] - v).abs().max()) for k, v in zip(("kind", "body", "lang"), b)])
torch.onnx.export(chip, xt[:1] / chip.sc, f"{out}/model.onnx", opset_version=13, input_names=["pooled_scaled"], output_names=["kind", "body", "lang"], dynamo=False)
print("exported", f"{out}/model.onnx")

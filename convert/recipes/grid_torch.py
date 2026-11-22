"""grid_torch.py export|data CKPT OUT_DIR [N_BENCH]
export: GridNet trunk (p2, p4) with the ProjBlock broadcast rewrite (parity asserted) -> model.onnx. Host keeps every head + TableMergerV2.
data:   eval_x.npy = the exact 896 canvases TableStructure feeds for N_BENCH (default 400) VN thesis-bench tables (seed 1121),
        ref_<i>.npy = torch trunk p2/p4 NHWC for the first 32 rows (FP parse check), torch end-to-end result -> torch_e2e.json."""
import sys, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
W = "/mnt/storage/lib_project/models/Dolphin/wordlid"; sys.path[:0] = [W, W + "/third_party"]
import gridnet as G
cmd, ck, out = sys.argv[1:4]
def compatible(self, x):
    r = F.interpolate(x.mean(3, keepdim=True), size=x.shape[-2:], mode="nearest"); c = F.interpolate(x.mean(2, keepdim=True), size=x.shape[-2:], mode="nearest")
    return self.fuse(torch.cat([x, r, c], 1))
if cmd == "export":
    c = torch.load(ck, map_location="cpu", weights_only=False); ar = c["net_arch"]
    net = G.GridNet(head_stride=ar["head_stride"], width=ar["width"]); net.load_state_dict(c["net"]); net = net.cuda().eval()
    class Chip(nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, x): return s.n.trunk(x)
    chip = Chip(net).eval(); x = torch.rand(1, 1, 896, 896, device="cuda")
    with torch.no_grad(): ref = chip(x)
    G.ProjBlock.forward = compatible
    with torch.no_grad(): y = chip(x)
    e = [float((a - b).abs().max()) for a, b in zip(y, ref)]; print("broadcast rewrite parity", e); assert max(e) < 1e-5
    torch.onnx.export(chip, x, f"{out}/model.onnx", opset_version=13, input_names=["canvas"], output_names=["p2", "p4"], dynamo=False); sys.exit()
from gridnet_full import TableStructure
n = int(sys.argv[4]) if len(sys.argv) > 4 else 400; BD = W + "/data/vn_bench"; B = json.load(open(BD + "/bench.json"))
from PIL import Image
m = TableStructure.load(ck); ix = np.sort(np.random.default_rng(1121).choice(len(B), n, replace=False))
cv = np.lib.format.open_memmap(f"{out}/eval_x.npy", "w+", np.uint8, (n, 896, 896, 1))
for j, i in enumerate(ix): cv[j, ..., 0] = m._canvas(np.asarray(Image.open(f"{BD}/img/{B[i]['id']}.png").convert("L")), m.norm_text)[0]
cv.flush(); np.save(f"{out}/host/bench_ix.npy", ix)
with torch.no_grad():
    x = torch.from_numpy(np.asarray(cv[:32, ..., 0])).float().div(255)[:, None].cuda(); r = [m.net.trunk(x[i:i + 4]) for i in range(0, 32, 4)]
for k in (0, 1): np.save(f"{out}/ref_{k}.npy", torch.cat([q[k] for q in r]).permute(0, 2, 3, 1).float().cpu().numpy())
print("eval", cv.shape, "ref rows 32")

"""layout_torch.py export|data CKPT OUT_DIR
PP-DocLayoutV3 chip part (backbone + encoder_input_proj + hybrid encoder + decoder_input_proj) at 896, input RGB/255.
export: model.onnx (parity vs the full model printed).  data: calib/eval links + torch reference ref_<i>.npy (f16 NHWC, size order
s8 112x112x256, mask 224x224x32, s16, s32) for ALL eval rows, so the FP AP baseline is scored from the same files as the emulator."""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/mnt/storage/lib_project/models/layout_teacher/train"); import train_ppdl3 as TP
cmd, ck, out = sys.argv[1:4]; PP = "/mnt/storage/lib_project/models/Hailo_model/preproc_hailo"
m = TP.build_model(); m.config._attn_implementation = "eager"
for mod in m.modules():
    if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"): mod.config._attn_implementation = "eager"
m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval(); core = m.model
class Chip(nn.Module):
    def __init__(s, c): super().__init__(); s.c = c
    def forward(s, x):
        c = s.c; f = c.backbone(x, torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device)); x4 = f.pop(0)
        e = c.encoder([c.encoder_input_proj[l](src) for l, (src, _) in enumerate(f)], x4)
        return tuple(c.decoder_input_proj[l](s_) for l, s_ in enumerate(e.last_hidden_state)) + (e.mask_feat,)
class ConstPos(nn.Module):
    def __init__(s, pe): super().__init__(); s.register_buffer("pe", pe)
    def forward(s, width, height, device, dtype): return s.pe
for a in core.encoder.aifi:
    with torch.no_grad(): pe = a.position_embedding(28, 28, torch.device("cpu"), torch.float32)
    a.position_embedding = ConstPos(pe.clone())
m = m.cuda(); chip = Chip(core).eval().cuda()
if cmd == "export":
    x = torch.rand(1, 3, 896, 896, device="cuda"); cap = {}
    h = core.encoder.register_forward_hook(lambda mod, i, o: cap.__setitem__("enc", o))
    with torch.no_grad(): y = chip(x); m(pixel_values=x)
    h.remove(); ref = [core.decoder_input_proj[l](s_) for l, s_ in enumerate(cap["enc"].last_hidden_state)] + [cap["enc"].mask_feat]
    with torch.no_grad(): e = [float((a - b).abs().max()) for a, b in zip(y, ref)]
    print("parity vs full model", e); assert max(e) < 1e-3
    torch.onnx.export(chip, x, f"{out}/model.onnx", opset_version=13, input_names=["image"], output_names=["src0", "src1", "src2", "mask_feat"], dynamo=False); sys.exit()
for a, b in ((f"{PP}/calib/layout_1024.npy", "calib.npy"), (f"{PP}/eval/layout_x.npy", "eval_x.npy")):
    if not os.path.exists(f"{out}/{b}"): os.symlink(a, f"{out}/{b}")
x = np.load(f"{out}/eval_x.npy", mmap_mode="r"); refs = None
with torch.no_grad():
    for i in range(0, len(x), 8):
        y = [t.permute(0, 2, 3, 1).half().cpu().numpy() for t in chip(torch.from_numpy(np.asarray(x[i:i + 8])).cuda().permute(0, 3, 1, 2).float() / 255)]
        if refs is None:
            order = sorted(range(4), key=lambda k: -y[k][0].size)
            refs = [np.lib.format.open_memmap(f"{out}/ref_{j}.npy", "w+", np.float16, (len(x),) + y[k].shape[1:]) for j, k in enumerate(order)]
        for j, k in enumerate(order): refs[j][i:i + len(y[k])] = y[k]
[r.flush() for r in refs]; print("ref", [r.shape for r in refs])

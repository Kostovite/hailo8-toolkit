"""preproc_torch.py export|data KIND(orient|flat) CKPT OUT_DIR
export: chip part -> OUT/model.onnx.   data: calib.npy / eval_x.npy links + torch reference OUT/ref_0.npy (NHWC, float32).
orient chip part = stem+body on x/255 (the -.5 is inside); host = 7x7 mean, fc, softmax, decide_ll.
flat chip part   = UVDocnet(...)[0] mesh on RGB/255 712x488; host = remap."""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path[:0] = ["/mnt/storage/lib_project/models/layout_teacher/train", "/mnt/storage/lib_project/models/Hailo_model/uvdoc/src"]
cmd, kind, ck, out = sys.argv[1:5]; PP = "/mnt/storage/lib_project/models/Hailo_model/preproc_hailo"
if kind == "orient":
    from train_orient import OriNet
    n = OriNet(); n.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    class W(nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, x): return s.n.body(s.n.stem(x - .5))
    shape, names, cal, ev = (1, 3, 896, 896), ["feat"], "orient_1024", "orient_canvas"
else:
    from model import UVDocnet
    n = UVDocnet(num_filter=32, kernel_size=5); sd = torch.load(ck, map_location="cpu", weights_only=False); n.load_state_dict(sd.get("model_state", sd))
    class W(nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, x): return s.n(x)[0]
    shape, names, cal, ev = (1, 3, 712, 488), ["mesh"], "flat_1024", "flat_frames"
m = W(n).eval().cuda()
if cmd == "export":
    x = torch.rand(shape, device="cuda")
    torch.onnx.export(m, x, f"{out}/model.onnx", opset_version=13, input_names=["image"], output_names=names, dynamo=False); print("exported", names)
else:
    for a, b in ((f"{PP}/calib/{cal}.npy", "calib.npy"), (f"{PP}/eval/{ev}.npy", "eval_x.npy")):
        if not os.path.exists(f"{out}/{b}"): os.symlink(a, f"{out}/{b}")
    x = np.load(f"{out}/eval_x.npy", mmap_mode="r"); ref = None
    with torch.no_grad():
        for i in range(0, len(x), 16):
            y = m(torch.from_numpy(np.asarray(x[i:i + 16])).cuda().permute(0, 3, 1, 2).float() / 255).permute(0, 2, 3, 1).float().cpu().numpy()
            if ref is None: ref = np.lib.format.open_memmap(f"{out}/ref_0.npy", "w+", np.float32, (len(x),) + y.shape[1:])
            ref[i:i + len(y)] = y
    ref.flush(); print("ref", ref.shape)

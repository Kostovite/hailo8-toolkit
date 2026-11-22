"""PP-DocLayoutV3 (r3b / r5) chip part -> ONNX at 896x896: backbone + encoder_input_proj + hybrid encoder (AIFI + FPN/PAN
+ mask feature head) + decoder_input_proj.  Outputs: src0 (s8), src1 (s16), src2 (s32), mask_feat.  Host: enc_output,
score/box heads, top-k queries, deformable decoder, mask head, order head.  Input RGB 0..1 (black letterbox, /255).
usage: export_layout.py EMA_CKPT OUT.onnx"""
import sys, torch, torch.nn as nn
sys.path.insert(0, "/mnt/storage/lib_project/models/layout_teacher/train")
import train_ppdl3 as TP
ck, out = sys.argv[1], sys.argv[2]
m = TP.build_model(); m.config._attn_implementation = "eager"
for mod in m.modules():
    if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"): mod.config._attn_implementation = "eager"
m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval(); core = m.model

class Chip(nn.Module):
    def __init__(s, c): super().__init__(); s.c = c
    def forward(s, x):
        c = s.c; f = c.backbone(x, torch.ones(x.shape[0], x.shape[2], x.shape[3])); x4 = f.pop(0)
        e = c.encoder([c.encoder_input_proj[l](src) for l, (src, _) in enumerate(f)], x4)
        return tuple(c.decoder_input_proj[l](s_) for l, s_ in enumerate(e.last_hidden_state)) + (e.mask_feat,)

class ConstPos(nn.Module):                                    # fixed 896 input: the AIFI sine embedding is a constant
    def __init__(s, pe): super().__init__(); s.register_buffer("pe", pe)
    def forward(s, width, height, device, dtype): return s.pe
for a in core.encoder.aifi:
    with torch.no_grad(): pe = a.position_embedding(28, 28, torch.device("cpu"), torch.float32)
    a.position_embedding = ConstPos(pe.clone())
chip = Chip(core).eval(); x = torch.rand(1, 3, 896, 896)
cap = {}
h = core.encoder.register_forward_hook(lambda mod, i, o: cap.__setitem__("enc", o))
with torch.no_grad():
    y = chip(x); m(pixel_values=x)
h.remove()
ref = [core.decoder_input_proj[l](s_) for l, s_ in enumerate(cap["enc"].last_hidden_state)] + [cap["enc"].mask_feat]
with torch.no_grad():
    print("parity vs full model:", [float((a - b).abs().max()) for a, b in zip(y, ref)], [tuple(a.shape) for a in y])
torch.onnx.export(chip, x, out, opset_version=13, input_names=["image"], output_names=["src0", "src1", "src2", "mask_feat"], dynamo=False)
import onnx, collections; g = onnx.load(out).graph; print(collections.Counter(n.op_type for n in g.node))

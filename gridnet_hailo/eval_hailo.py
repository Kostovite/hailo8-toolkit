"""Score a quantized HAR with the trainer's own validate(): the CHIP part runs in the DFC emulator, the HOST part
(profiles, 1-D heads, band/strip pooling) is the unchanged torch module fed the emulator's p2.
usage: python eval_hailo.py HAR CONTEXT(native|quant|exact) VAL_TAG LIMIT"""
import json, sys, time
import common as C
import numpy as np, torch
from hailo_sdk_client import ClientRunner, InferenceContext

har, ctxname, tag, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
CTX = {"native": InferenceContext.SDK_NATIVE, "quant": InferenceContext.SDK_QUANTIZED,
       "exact": InferenceContext.SDK_BIT_EXACT}[ctxname]
torch.set_num_threads(16)
net = C.load_net()
r = ClientRunner(har=har)


class Emu(torch.nn.Module):
    def __init__(self, net, ctx):
        super().__init__(); self.net, self.ctx = net, ctx

    def forward(self, x, bands=None, strips=None, units=None, unit_bi=None):
        # SDK_NATIVE of an optimized HAR runs WITHOUT the normalization layer (measured): feed it 0..1
        outs = r.infer(self.ctx, (x.numpy() * (1 if ctxname == "native" else 255)).transpose(0, 2, 3, 1).astype(np.float32), batch_size=len(x))
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        p2 = next(o for o in outs if o.ndim == 4 and o.shape[1] == 448)
        kind = next(o for o in outs if o.size // len(x) == len(C.S.KINDS)).reshape(len(x), -1)
        p2 = torch.from_numpy(np.ascontiguousarray(p2.transpose(0, 3, 1, 2))).float()
        self.net.trunk.forward = lambda _x: (p2, torch.zeros(len(x), p2.shape[1] * 4 // 3, 1, 1))
        o = self.net(x, bands, strips)
        o["kind"] = torch.from_numpy(kind).float()
        return o


t = time.time()
with r.infer_context(CTX) as ctx:
    m = C.T.validate(Emu(net, ctx), C.loader(tag), "cpu", False, limit)
print(har, ctxname, tag, f"{time.time() - t:.0f}s", json.dumps({k: round(v, 4) for k, v in m.items()
      if k not in ("utype", "ucls", "body", "lang", "word_f1_pred")}), flush=True)

import common as C, numpy as np, torch
from hailo_sdk_client import ClientRunner, InferenceContext
net = C.load_net(); b = next(iter(C.loader("vtval", bs=2, workers=0)))
x = b["x"].float()
with torch.no_grad():
    p2, p4 = net.trunk(x / 255); k = net.kind_head(net.region(p4.mean((2, 3))))
r = ClientRunner(har="L0_int8.har")
with r.infer_context(InferenceContext.SDK_NATIVE) as ctx:
    o = r.infer(ctx, x.numpy().transpose(0, 2, 3, 1), batch_size=2)
for t in (o if isinstance(o, (list, tuple)) else [o]): print(type(t), t.shape, float(t.mean()))
print("torch p2", p2.shape, float(p2.mean()), "kind", k)
e = next(t for t in o if t.ndim == 4 and t.shape[1] == 448)
print("max abs diff", float(np.abs(e.transpose(0, 3, 1, 2) - p2.numpy()).max()), float(np.abs(p2.numpy()).max()))

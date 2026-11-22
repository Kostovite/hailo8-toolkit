import common as C, numpy as np, torch, onnxruntime as ort
from hailo_sdk_client import ClientRunner, InferenceContext
net = C.load_net(); b = next(iter(C.loader("vtval", bs=1, workers=0)))
x = b["x"].float() / 255
with torch.no_grad():
    p2, p4 = net.trunk(x)
s = ort.InferenceSession("gridnet_vert1_chip.onnx", providers=["CPUExecutionProvider"])
oo = s.run(None, {"canvas": x.numpy()})
print("onnx vs torch p2", float(np.abs(oo[0] - p2.numpy()).max()))
for har, scale in (("gridnet_vert1_parsed.har", 1), ("L0_int8.har", 255), ("L0_int8.har", 1)):
    r = ClientRunner(har=har)
    with r.infer_context(InferenceContext.SDK_NATIVE) as ctx:
        o = r.infer(ctx, (x.numpy() * scale).transpose(0, 2, 3, 1), batch_size=1)
    e = next(t for t in o if t.ndim == 4 and t.shape[1] == 448)
    print(har, scale, "diff", float(np.abs(e.transpose(0, 3, 1, 2) - p2.numpy()).max()))

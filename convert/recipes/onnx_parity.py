import sys, numpy as np, onnxruntime as ort
a, b = sys.argv[1:3]; x = np.load("/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/eval/layout_x.npy", mmap_mode="r")
sa, sb = ort.InferenceSession(a, providers=["CPUExecutionProvider"]), ort.InferenceSession(b, providers=["CPUExecutionProvider"])
inp = sa.get_inputs()[0]; print(inp.name, inp.shape)
for i in (0, 700, 1400):
    xi = np.asarray(x[i:i+1], np.float32)
    if inp.shape[1] == 3: xi = xi.transpose(0, 3, 1, 2)
    xi = xi / 255.0 if "norm" not in sys.argv else xi
    ya, yb = sa.run(None, {inp.name: xi}), sb.run(None, {sb.get_inputs()[0].name: xi})
    r = [float(np.abs(p-q).max()/(np.abs(p).max()+1e-9)) for p, q in zip(ya, yb)]; print(i, [f"{v:.2e}" for v in r])
    assert max(r) < 1e-5, "rewrite changed the float model"
print("ONNX_PARITY PASS")

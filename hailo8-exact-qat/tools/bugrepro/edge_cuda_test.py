"""hq_ext APU (fused) vs intops.apu on the softmax reciprocal table, inputs spanning and exceeding the table ends."""
import pickle, sys, numpy as np, torch
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat")
import intops as I, atomic as A, qatfast as Q
P = pickle.load(open("capture_mTPC/_all_ops.pkl", "rb")); ra = dict(P["best/reduce_sum_softmax1/act_op"])
ra["act_in"] = np.zeros((1, 1, 1, 4))
apu = Q.APU(ra, "cuda")
x = torch.arange(-32768, 32768, dtype=torch.float32, device="cuda").reshape(1, 1, 256, 256).repeat(1, 4, 1, 1)
y_cuda = apu.fused(x).double().cpu()
y_ref = apu(x.double()).cpu() if False else I.apu(x.double().cpu(), A._t(ra["thr"], x.double().cpu()), A._t(ra["mant"], x.double().cpu()), A._t(ra["shift"], x.double().cpu()), A._t(ra["offs"], x.double().cpu()), int(ra["final_shift"]), bool(ra["mant_signed"]), A.lossy(ra, "clip1"), A.lossy(ra, "clip2"), A.lossy(ra, "out"))
d = (y_cuda != y_ref)
print("inputs", x.numel(), "cuda vs intops mismatches", int(d.sum()), "at", x.cpu()[d][:8].tolist())
print("y at -32768, -32767, 32766, 32767:", [int(y_cuda.reshape(4, -1)[0, i]) for i in (0, 1, 65534, 65535)])

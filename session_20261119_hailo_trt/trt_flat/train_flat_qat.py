import os, time, copy, random, numpy as np, torch
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
import train_flat as F
from torch.utils.data import DataLoader

torch.set_num_threads(3); random.seed(42); np.random.seed(42); torch.manual_seed(42)
D=os.path.expanduser('~/codex_trt_qat_20261119'); out=D+'/artifacts/flat3d_qat'; os.makedirs(out,exist_ok=True)
dev='cuda:0'; ck=D+'/source/layout_teacher/staged_v2_20261114/flat3d_2500.pt'; uv=D+'/source/layout_teacher/data_ext/uvdoc/UVDoc_final'
tr,va=F.split(); uvt,uvv=F.uv_list(uv); print('split', {k:len(v) for k,v in tr.items()},'val',sum(map(len,va.values())),'uv',len(uvt),len(uvv),flush=True)
assert tr and va and uvt and uvv
# Current FP model is the teacher; preserve its photo and scan behavior rather than inventing a new target.
teacher=F.load(ck,dev).eval(); [p.requires_grad_(False) for p in teacher.parameters()]
net=F.load(ck,dev).eval(); ds=F.DS(tr,seed=4921,p_flat=.35,uv=(uv,uvt),p_uv=.3)
it=iter(DataLoader(ds,batch_size=4,num_workers=4,prefetch_factor=2,persistent_workers=True))
def xbatch():
 x,g,k=next(it); return x.to(dev,non_blocking=True).float().div_(255),g.to(dev),k
print('calibrating on training data',flush=True)
def calib(m):
 with torch.no_grad():
  for i in range(16):
   x,_,_=xbatch(); m(x); print('calibration',i+1,flush=True)
mtq.quantize(net,mtq.INT8_DEFAULT_CFG,calib)
# Frozen teacher distillation, conservative learning rate; evaluate held-out samples with common inputs.
net.train(); opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=0); start=time.time()
for step in range(1,1001):
 x,g,k=xbatch()
 with torch.no_grad(): target=teacher(x)[0].float()
 pred=net(x)[0].float(); loss=(pred-target).abs().mean()
 if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
 opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
 if step%25==0: print('step',step,'distill_mae',float(loss),'sec/step',(time.time()-start)/step,flush=True)
 if step in (250,500,1000):
  net.eval(); vals=iter(DataLoader(F.DS(va,seed=812,p_flat=.35,uv=(uv,uvv),p_uv=.3),batch_size=4,num_workers=2))
  rows=[]
  with torch.no_grad():
   for j in range(10):
    vx,vg,vk=next(vals); vx=vx.to(dev).float()/255; vg=vg.to(dev); fp=teacher(vx)[0].float(); q=net(vx)[0].float(); rows.append(((fp-vg).abs().mean().item()*50,(q-vg).abs().mean().item()*50,(q-fp).abs().mean().item()*50))
  print('heldout step',step,'fp_gt q_gt q_fp percent_frame',np.mean(rows,axis=0).tolist(),flush=True)
  mto.save(net,f'{out}/flat3d_qat_{step}.pth'); net.train()
print('FLAT_QAT_DONE',flush=True)

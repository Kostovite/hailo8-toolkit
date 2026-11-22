import os,sys,time,copy,hashlib,random
import numpy as np,torch,torch.nn as nn
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
import photo_data as P
import train_ppdl3 as TP
from torch.utils.data import DataLoader
D=os.path.expanduser('~/codex_trt_qat_20261119'); TP.HF=D+'/source/layout_teacher/weights/ppdl3_hf'
ck=D+'/source/layout_teacher/staged_v2_20261114/r5_ord5.pt'; out=D+'/artifacts/layout_qat'; os.makedirs(out,exist_ok=True)
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(3)
# Restrict calibration and training to non-held-out pages.
manifest=P.manifest(); filt=lambda v,held:[it for it in v if (int(hashlib.md5(it[0].encode()).hexdigest()[:6],16)%50==0)==held]
training={k:filt(v,False) for k,v in manifest.items()}; val={k:filt(v,True) for k,v in manifest.items()}; training={k:v for k,v in training.items() if v};val={k:v for k,v in val.items() if v}
print('train/val',sum(map(len,training.values())),sum(map(len,val.values())),flush=True);assert training and val
class Chip(nn.Module):
 def __init__(self,core): super().__init__(); self.c=core
 def forward(self,x):
  c=self.c; f=c.backbone(x,torch.ones(x.shape[0],x.shape[2],x.shape[3],device=x.device)); x4=f.pop(0)
  e=c.encoder([c.encoder_input_proj[i](s) for i,(s,_) in enumerate(f)],x4)
  return tuple(c.decoder_input_proj[i](s) for i,s in enumerate(e.last_hidden_state))+(e.mask_feat,)
def make():
 m=TP.build_model();m.config._attn_implementation='eager'
 for mod in m.modules():
  if hasattr(mod,'config') and hasattr(mod.config,'_attn_implementation'): mod.config._attn_implementation='eager'
 m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True));return Chip(m.model).cuda().eval()
teacher=make(); [p.requires_grad_(False) for p in teacher.parameters()]; net=copy.deepcopy(teacher)
P.manifest=lambda:training
weights={k:1 for k in training}; ds=TP.Mix(10000,.5,weights,seed=42)
it=iter(DataLoader(ds,batch_size=1,num_workers=3,persistent_workers=True,collate_fn=TP.collate))
def get(): return next(it)[0].cuda(non_blocking=True)
def calib(m):
 with torch.no_grad():
  for j in range(16): m(get()); print('calibration',j+1,flush=True)
print('calibrating',flush=True);mtq.quantize(net,mtq.INT8_DEFAULT_CFG,calib)
net.train(); opt=torch.optim.AdamW(net.parameters(),lr=2e-6,weight_decay=0);start=time.time()
for step in range(1,501):
 x=get()
 with torch.no_grad(): target=teacher(x)
 pred=net(x); losses=[(a.float()-b.float()).abs().mean()/(b.float().abs().mean().detach()+1e-4) for a,b in zip(pred,target)];loss=sum(losses)/len(losses)
 if not torch.isfinite(loss): raise RuntimeError('nonfinite')
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
 if step%10==0:print('step',step,'relative_mae',float(loss),'s/step',(time.time()-start)/step,flush=True)
 if step in (100,250,500):mto.save(net,f'{out}/layout_chip_qat_{step}.pth')
print('LAYOUT_CHIP_QAT_DONE; validation and TensorRT engine required',flush=True)

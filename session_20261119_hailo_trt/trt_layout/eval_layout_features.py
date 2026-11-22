import os,sys,hashlib,random, numpy as np,torch,torch.nn as nn
import modelopt.torch.opt as mto
import photo_data as P,train_ppdl3 as TP
from torch.utils.data import DataLoader
D=os.path.expanduser('~/codex_trt_qat_20261119');TP.HF=D+'/source/layout_teacher/weights/ppdl3_hf'; ck=D+'/source/layout_teacher/staged_v2_20261114/r5_ord5.pt'
random.seed(73);np.random.seed(73);torch.manual_seed(73);torch.set_num_threads(3)
class Chip(nn.Module):
 def __init__(self,c):super().__init__();self.c=c
 def forward(self,x):
  c=self.c;f=c.backbone(x,torch.ones(x.shape[0],x.shape[2],x.shape[3],device=x.device));x4=f.pop(0);e=c.encoder([c.encoder_input_proj[i](s) for i,(s,_) in enumerate(f)],x4);return tuple(c.decoder_input_proj[i](s) for i,s in enumerate(e.last_hidden_state))+(e.mask_feat,)
def make():
 m=TP.build_model();m.config._attn_implementation='eager'
 for mod in m.modules():
  if hasattr(mod,'config') and hasattr(mod.config,'_attn_implementation'):mod.config._attn_implementation='eager'
 m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True));return Chip(m.model).cuda().eval()
manifest=P.manifest();v={k:[x for x in vs if int(hashlib.md5(x[0].encode()).hexdigest()[:6],16)%50==0] for k,vs in manifest.items()};v={k:x for k,x in v.items() if x};P.manifest=lambda:v
models=[make() for _ in range(4)];
for n,s in zip(models[1:],(100,250,500)):mto.restore(n,D+f'/artifacts/layout_qat/layout_chip_qat_{s}.pth');n.eval()
ds=TP.Mix(100,.5,{k:1 for k in v},seed=73);loader=DataLoader(ds,batch_size=1,num_workers=0,collate_fn=TP.collate);total=np.zeros((3,4));n=0
with torch.no_grad():
 for x,tg in loader:
  x=x.cuda();fp=models[0](x)
  for i,m in enumerate(models[1:]):
   y=m(x)
   for j,(a,b) in enumerate(zip(y,fp)):total[i,j]+=float((a-b).abs().mean()/(b.abs().mean()+1e-4))
  n+=1
  if n%20==0:print('samples',n,'relative_feature_mae', (total/n).tolist(),flush=True)
print('RESULT samples',n,'rows=qat100,250,500 columns=src0,src1,src2,mask relative_feature_mae',(total/n).tolist(),flush=True)

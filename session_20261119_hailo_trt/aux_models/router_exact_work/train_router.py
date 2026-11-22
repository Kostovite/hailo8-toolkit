import os,sys,json,numpy as np,torch
A=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path[:0]=['/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat','/mnt/storage/lib_project/models/Dolphin','/mnt/storage/lib_project/models/Dolphin/wordlid']
import graph26 as G
from qatfast_router import QATNet
from region_router import RegionRouter
torch.set_num_threads(2);torch.manual_seed(42);device='cuda'
hn=json.load(open(A+'/router_exact.hn'));P=G.Params(A+'/router_exact_capture',device);net=QATNet(hn,P,device)
ck=torch.load('/mnt/storage/lib_project/models/Dolphin/wordlid/data/grid/region_router.pt',map_location='cpu',weights_only=False);teacher=RegionRouter().to(device).eval();teacher.load_state_dict(ck['router'])
tr=torch.from_numpy(np.load(A+'/router_more16_train.npy')).to(device);va=torch.from_numpy(np.load(A+'/router_more16_val.npy')).to(device)
heads={'body':'routercur/conv3','kind':'routercur/conv4','lang':'routercur/conv5'}
def forward(x):
 out=net(x[:,None,None,:]);return {k:(out[n].flatten(1)-torch.as_tensor(np.asarray(P.conv[n]['out_zp']),device=device).reshape(1,-1))*torch.as_tensor(np.asarray(P.conv[n]['out_scale']),device=device).reshape(1,-1) for k,n in heads.items()}
def evaluate():
 totals={k:0 for k in heads};num=0
 with torch.no_grad():
  for x in va.split(16):
   t=teacher(x);y=forward(x);num+=len(x)
   for k in heads:totals[k]+=int((t[k].argmax(1)==y[k].argmax(1)).sum())
 return {k:round(v/num,5) for k,v in totals.items()}
print('baseline heldout agreement',evaluate(),flush=True)
optim=torch.optim.Adam([{'params':[m.kernel for m in net.mac.values()],'lr':0.001},{'params':[m.bias for m in net.mac.values()],'lr':0.02}])
for step in range(1,1001):
 x=tr[torch.randint(len(tr),(16,),device=device)]
 with torch.no_grad():t=teacher(x)
 y=forward(x);loss=sum((y[k]-t[k]).square().mean() for k in heads)
 if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
 optim.zero_grad(set_to_none=True);loss.backward();optim.step()
 if step%100==0:
  match=evaluate();print('step',step,'loss',float(loss),'heldout_agreement',match,flush=True)
  torch.save({'step':step,'state_dict':net.state_dict(),'agreement':match},A+'/router_exact_work/router_exact_step%04d.pt'%step)
print('QAT experiment only: no export, roundtrip or chip validation yet',flush=True)

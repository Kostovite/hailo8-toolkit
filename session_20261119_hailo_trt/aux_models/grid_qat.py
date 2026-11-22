import os,sys,copy,numpy as np,torch,torch.nn as nn
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
D=os.path.expanduser('~/codex_trt_qat_20261119');A=D+'/aux_models';sys.path.insert(0,A+'/wordlid')
import gridnet as G
ck=torch.load(A+'/wordlid/data/grid/table_structure_final.pt',map_location='cpu',weights_only=False);ar=ck['net_arch'];base=G.GridNet(head_stride=ar['head_stride'],width=ar['width']);base.load_state_dict(ck['net']);base.eval()
class Chip(nn.Module):
 def __init__(self,n):super().__init__();self.n=n
 def forward(self,x):
  p2,p4=self.n.trunk(x);h=self.n.region(p4.mean((2,3)));return p2,self.n.kind_head(h)
ref=Chip(base).cuda().eval();[p.requires_grad_(False) for p in ref.parameters()];net=copy.deepcopy(ref);[p.requires_grad_(True) for p in net.parameters()]
torch.set_num_threads(3);torch.manual_seed(42);train=np.load(A+'/grid_g5train_320.npz')['images'];val=np.load(A+'/grid_g5val_320.npz')['images'];os.makedirs(A+'/grid_qat',exist_ok=True)
def get(a,i):return torch.from_numpy(np.ascontiguousarray(a[i]))[None,None].cuda().float().div_(255)
def calib(m):
 with torch.no_grad():
  for j in range(64):m(get(train,j));print('calib',j+1,flush=True) if j%8==7 else None
cfg=copy.deepcopy(mtq.INT8_DEFAULT_CFG);cfg['quant_cfg'].update({'n.region.*':{'enable':False},'n.kind_head.*':{'enable':False}})
mtq.quantize(net,cfg,calib);net.eval();opt=torch.optim.AdamW(net.parameters(),lr=2e-6,weight_decay=0)
for step in range(1,301):
 x=get(train,(step*101)%len(train))
 with torch.no_grad():a=ref(x)
 b=net(x);loss=sum((u-v).abs().mean()/(v.abs().mean().detach()+1e-5) for u,v in zip(b,a));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
 if step%25==0:print('step',step,'loss',float(loss),flush=True)
 if step in (100,200,300):
  acc=[]
  with torch.no_grad():
   for i in range(min(20,len(val))):
    x=get(val,i);a=ref(x);b=net(x);acc.append([(float((u-v).abs().mean()/(v.abs().mean()+1e-5))) for u,v in zip(b,a)])
  print('heldout feature relative MAE',step,np.mean(acc,axis=0).tolist(),flush=True);mto.save(net,A+f'/grid_qat/grid_chip_qat_{step}.pth')
print('GRID_CHIP_QAT_COMPLETE; host and merger gates remain',flush=True)

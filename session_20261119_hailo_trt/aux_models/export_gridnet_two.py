import sys,torch,torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0,'/mnt/storage/lib_project/models/Dolphin/wordlid')
import gridnet as G
ck=torch.load('/mnt/storage/lib_project/models/Dolphin/wordlid/data/grid/table_structure_final.pt',map_location='cpu',weights_only=False)
ar=ck['net_arch'];net=G.GridNet(head_stride=ar['head_stride'],width=ar['width']);net.load_state_dict(ck['net']);net=net.cuda().eval()
class Chip(nn.Module):
 def __init__(s,n):super().__init__();s.n=n
 def forward(s,x):
  p2,p4=s.n.trunk(x);h=s.n.region(p4.mean((2,3)));return p2,s.n.kind_head(h)
chip=Chip(net).eval();x=torch.rand(1,1,896,896,device='cuda')
with torch.no_grad():ref=chip(x)
def compatible(self,x):
 r=F.interpolate(x.mean(3,keepdim=True),size=x.shape[-2:],mode='nearest');c=F.interpolate(x.mean(2,keepdim=True),size=x.shape[-2:],mode='nearest');return self.fuse(torch.cat([x,r,c],1))
G.ProjBlock.forward=compatible
with torch.no_grad():
 y=chip(x);err=[float((a-b).abs().max()) for a,b in zip(y,ref)];print('broadcast rewrite parity maxabs',err,'outputs',[tuple(a.shape) for a in y],flush=True);assert max(err)<1e-5
 torch.onnx.export(chip,x,sys.argv[1],opset_version=13,input_names=['canvas'],output_names=['p2','kind'],dynamo=False)
print('export complete',flush=True)

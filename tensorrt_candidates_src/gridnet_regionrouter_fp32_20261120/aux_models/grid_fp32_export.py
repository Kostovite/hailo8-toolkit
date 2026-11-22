import os,sys,torch,torch.nn as nn
D=os.path.expanduser('~/codex_trt_qat_20261119');A=D+'/aux_models';sys.path.insert(0,A+'/wordlid');import gridnet as G
ck=torch.load(A+'/wordlid/data/grid/table_structure_final.pt',map_location='cpu',weights_only=False);ar=ck['net_arch'];net=G.GridNet(head_stride=ar['head_stride'],width=ar['width']);net.load_state_dict(ck['net']);net.eval();torch.set_num_threads(3)
class Chip(nn.Module):
 def __init__(self,n):super().__init__();self.n=n
 def forward(self,x):
  p2,p4=self.n.trunk(x);h=self.n.region(p4.mean((2,3)));return p2,self.n.kind_head(h)
w=Chip(net).eval();x=torch.rand(1,1,896,896);out=A+'/grid_fp32.onnx'
with torch.no_grad():
 print('outputs',[tuple(a.shape) for a in w(x)],flush=True)
 torch.onnx.export(w,x,out,opset_version=17,input_names=['canvas'],output_names=['p2','kind'],dynamo=False,do_constant_folding=True)
print('exported',out,flush=True)

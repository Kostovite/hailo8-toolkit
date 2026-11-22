import os,sys,torch,torch.nn as nn
base=sys.argv[1];out=sys.argv[2]
sys.path.insert(0,base);sys.path.insert(0,base+'/wordlid')
from region_router import RegionRouter
ck=torch.load(base+'/wordlid/data/grid/region_router.pt',map_location='cpu',weights_only=False);net=RegionRouter();net.load_state_dict(ck['router']);net.eval()
class Pooled(nn.Module):
 def __init__(self,n):super().__init__();self.n=n
 def forward(self,x):
  r=self.n(x);return r['kind'],r['body'],r['lang']
x=torch.randn(1,1024);w=Pooled(net).eval();torch.onnx.export(w,x,out,opset_version=13,input_names=['pooled_encoder'],output_names=['kind','body','lang'],dynamic_axes={'pooled_encoder':{0:'batch'},'kind':{0:'batch'},'body':{0:'batch'},'lang':{0:'batch'}},dynamo=False)
print('exported pooled router',out,'recorded validation',ck.get('val'),flush=True)

import os,sys,torch,torch.nn as nn
sys.path.insert(0,'/mnt/storage/lib_project/models/layout_teacher/train')
sys.path.insert(0,'/mnt/storage/lib_project/models/Hailo_model/uvdoc/src')
from train_orient import OriNet
from model import UVDocnet
kind,out=sys.argv[1:]; device='cuda:0'; base='/mnt/storage/lib_project/models/layout_teacher/staged_v2_20261114/'
if kind=='orient3':
 n=OriNet();n.load_state_dict(torch.load(base+'orient3_10000.pt',map_location='cpu',weights_only=True))
 class Wrapper(nn.Module):
  def __init__(self,n): super().__init__();self.n=n
  def forward(self,x): return self.n.body(self.n.stem(x-.5))
 m=Wrapper(n); shape=(1,3,896,896); names=['feat']
else:
 n=UVDocnet(num_filter=32,kernel_size=5); sd=torch.load(base+'flat3d_2500.pt',map_location='cpu',weights_only=False); n.load_state_dict(sd.get('model_state',sd))
 class Wrapper(nn.Module):
  def __init__(self,n): super().__init__();self.n=n
  def forward(self,x): return self.n(x)[0]
 m=Wrapper(n);shape=(1,3,712,488);names=['mesh']
m=m.eval().to(device);x=torch.rand(shape,device=device)
with torch.no_grad(): print('float output',tuple(m(x).shape),flush=True)
torch.onnx.export(m,x,out,opset_version=13,input_names=['image'],output_names=names,dynamo=False)
print('EXPORT_OK',out,flush=True)

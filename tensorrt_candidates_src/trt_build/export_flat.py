import os,sys,torch
import modelopt.torch.opt as mto
D=os.path.expanduser('~/codex_trt_qat_20261119');sys.path.insert(0,D+'/flat_qat')
from model import UVDocnet
class Wrapper(torch.nn.Module):
 def __init__(self,net):super().__init__();self.net=net
 def forward(self,x):return self.net(x)[0]
net=UVDocnet(num_filter=32,kernel_size=5);mto.restore(net,D+'/artifacts/flat3d_qat/flat3d_qat_500.pth');net=Wrapper(net.cpu().eval()).eval()
x=torch.rand(1,3,712,488);out=D+'/artifacts/trt/flat3d_qat500.onnx';torch.set_num_threads(3)
with torch.no_grad():
 print('torch output',net(x).shape,flush=True)
 torch.onnx.export(net,x,out,opset_version=17,input_names=['image'],output_names=['mesh'],dynamo=False,do_constant_folding=True)
print('exported',out,flush=True)

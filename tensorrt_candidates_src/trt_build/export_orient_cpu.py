import os,sys,torch
import modelopt.torch.opt as mto
D=os.path.expanduser('~/codex_trt_qat_20261119');sys.path.insert(0,D+'/orient_full')
from train_orient_qat import OriNet
class Feature(torch.nn.Module):
 def __init__(self,n):super().__init__();self.stem=n.stem;self.body=n.body
 def forward(self,x):return self.body(self.stem(x))
net=OriNet();mto.restore(net,D+'/artifacts/orient_full/ori_qat_500.pth');net.cpu().eval();net=Feature(net).eval().cpu()
x=torch.rand(1,3,224,224,device='cpu')-.5;out=D+'/artifacts/trt/orient3_qat500_features.onnx';os.makedirs(os.path.dirname(out),exist_ok=True)
with torch.no_grad():
 y=net(x);print('torch output',tuple(y.shape),flush=True)
 torch.onnx.export(net,x,out,opset_version=17,input_names=['image'],output_names=['features'],dynamic_axes={'image':{0:'batch',2:'height',3:'width'},'features':{0:'batch',2:'feature_height',3:'feature_width'}},dynamo=False,do_constant_folding=True)
print('exported',out,flush=True)

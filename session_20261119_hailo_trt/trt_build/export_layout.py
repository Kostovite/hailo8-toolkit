import os,sys,torch,torch.nn as nn
import modelopt.torch.opt as mto
D=os.path.expanduser('~/codex_trt_qat_20261119');sys.path.insert(0,D+'/layout_qat')
import train_ppdl3 as TP
TP.HF=D+'/source/layout_teacher/weights/ppdl3_hf'
class Chip(nn.Module):
 def __init__(self,c):super().__init__();self.c=c
 def forward(self,x):
  c=self.c;f=c.backbone(x,torch.ones(x.shape[0],x.shape[2],x.shape[3],device=x.device));x4=f.pop(0)
  e=c.encoder([c.encoder_input_proj[i](s) for i,(s,_) in enumerate(f)],x4)
  return tuple(c.decoder_input_proj[i](s) for i,s in enumerate(e.last_hidden_state))+(e.mask_feat,)
m=TP.build_model();m.config._attn_implementation='eager'
for mod in m.modules():
 if hasattr(mod,'config') and hasattr(mod.config,'_attn_implementation'):mod.config._attn_implementation='eager'
m.load_state_dict(torch.load(D+'/source/layout_teacher/staged_v2_20261114/r5_ord5.pt',map_location='cpu',weights_only=True));net=Chip(m.model).eval()
mto.restore(net,D+'/artifacts/layout_mixed_qat/layout_chip_qat_250.pth');net=net.cpu().eval();torch.set_num_threads(4)
x=torch.rand(1,3,896,896);out=D+'/artifacts/trt/layout_r5_mixed_qat250.onnx'
with torch.no_grad():
 y=net(x);print('output',[tuple(i.shape) for i in y],flush=True)
 torch.onnx.export(net,x,out,opset_version=17,input_names=['image'],output_names=['src0','src1','src2','mask_feat'],dynamo=False,do_constant_folding=True)
print('exported',out,flush=True)

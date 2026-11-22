import os,sys,torch,numpy as np,tensorrt as trt
import modelopt.torch.opt as mto
D=os.path.expanduser('~/codex_trt_qat_20261119');sys.path.insert(0,D+'/flat_qat')
from model import UVDocnet
import train_flat as F
net=UVDocnet(num_filter=32,kernel_size=5);mto.restore(net,D+'/artifacts/flat3d_qat/flat3d_qat_500.pth');net=net.cuda().eval();torch.set_num_threads(2)
logger=trt.Logger(trt.Logger.WARNING);runtime=trt.Runtime(logger)
with open(D+'/artifacts/trt/flat3d_qat500_headfp32.engine','rb') as f:engine=runtime.deserialize_cuda_engine(f.read())
assert engine is not None
ctx=engine.create_execution_context();tr,va=F.split();uv=D+'/source/layout_teacher/data_ext/uvdoc/UVDoc_final';_,uvv=F.uv_list(uv)
ds=F.DS(va,seed=812,p_flat=.35,uv=(uv,uvv),p_uv=.3); it=iter(ds); dif=[]; fpgt=[]; trtgt=[]
with torch.no_grad():
 for j in range(80):
  x,g,k=next(it);x=x[None].cuda().float().div_(255).contiguous();fp=net(x)[0];shape=tuple(ctx.get_tensor_shape('mesh'));assert min(shape)>0,shape;y=torch.empty(shape,device='cuda',dtype=torch.float32)
  assert ctx.set_tensor_address('image',x.data_ptr());assert ctx.set_tensor_address('mesh',y.data_ptr());assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize()
  d=(y-fp).abs();dif.append([float(d.mean()),float(d.max())]);fpgt.append(float((fp.cpu()[0]-g).abs().mean()*50));trtgt.append(float((y.cpu()[0]-g).abs().mean()*50))
  if (j+1)%20==0: print('pages',j+1,'mean_abs',np.mean(dif,axis=0),'fp_gt',np.mean(fpgt),'trt_gt',np.mean(trtgt),flush=True)
print('RESULT n',len(dif),'mean_abs',np.mean(dif,axis=0),'fp_gt',np.mean(fpgt),'trt_gt',np.mean(trtgt),flush=True)

assert float(np.mean(dif,axis=0)[0])*50 < .10, "engine differs from QAT by >=0.10% frame"
assert np.mean(trtgt) <= np.mean(fpgt)+.10, "held-out mesh error worsened by >0.10% frame"

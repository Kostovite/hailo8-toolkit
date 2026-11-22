import os,sys,numpy as np,torch,torch.nn as nn,tensorrt as trt
D=os.path.expanduser('~/codex_trt_qat_20261119');A=D+'/aux_models';sys.path.insert(0,A+'/wordlid');import gridnet as G
ck=torch.load(A+'/wordlid/data/grid/table_structure_final.pt',map_location='cpu',weights_only=False);ar=ck['net_arch'];n=G.GridNet(head_stride=ar['head_stride'],width=ar['width']);n.load_state_dict(ck['net']);n=n.cuda().eval();torch.set_num_threads(3)
rt=trt.Runtime(trt.Logger(trt.Logger.WARNING));engine=rt.deserialize_cuda_engine(open(A+'/grid_fp32.engine','rb').read());assert engine is not None;ctx=engine.create_execution_context();data=np.load(A+'/grid_g5val_320.npz')['images'];errs=[];agree=[]
with torch.no_grad():
 for j in range(30):
  x=torch.from_numpy(np.ascontiguousarray(data[j]))[None,None].float().cuda().div_(255).contiguous();p2,p4=n.trunk(x);kind=n.kind_head(n.region(p4.mean((2,3))));ctx.set_tensor_address('canvas',x.data_ptr());out=[]
  for key in ('p2','kind'):
   y=torch.empty(tuple(ctx.get_tensor_shape(key)),device='cuda',dtype=torch.float32);ctx.set_tensor_address(key,y.data_ptr());out.append(y)
  assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize();errs.append(float((p2-out[0]).abs().mean()/(p2.abs().mean()+1e-5)));agree.append(int(kind.argmax()==out[1].argmax()))
  if (j+1)%10==0:print('pages',j+1,'relative_feature_mae',np.mean(errs),'kind_agreement',np.mean(agree),flush=True)
print('RESULT n',len(errs),'relative_feature_mae',np.mean(errs),'kind_agreement',np.mean(agree),flush=True)
assert np.mean(agree)==1 and np.mean(errs)<.01

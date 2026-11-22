import os,sys,json,cv2,numpy as np,torch,tensorrt as trt
import torch.nn.functional as F
import modelopt.torch.opt as mto
D=os.path.expanduser('~/codex_trt_qat_20261119');sys.path.insert(0,D+'/orient_full')
import photo_data as P
from train_orient_qat import OriNet,ValPages,decide_ll
net=OriNet();mto.restore(net,D+'/artifacts/orient_full/ori_qat_500.pth');net=net.cuda().eval();torch.set_num_threads(2)
logger=trt.Logger(trt.Logger.WARNING);runtime=trt.Runtime(logger)
with open(D+'/artifacts/trt/orient3_qat500_features.engine','rb') as f:engine=runtime.deserialize_cuda_engine(f.read())
assert engine is not None
ctx=engine.create_execution_context()
def trt_forward(x):
 assert ctx.set_input_shape('image',tuple(x.shape)); shape=tuple(ctx.get_tensor_shape('features')); assert min(shape)>0,shape
 y=torch.empty(shape,device='cuda',dtype=torch.float32);ctx.set_tensor_address('image',x.data_ptr());ctx.set_tensor_address('features',y.data_ptr());assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream);return y
def decide(f):
 p=F.conv2d(F.avg_pool2d(f,7,1),net.fc.weight[:,:,None,None],net.fc.bias).softmax(1)[0];return decide_ll(p.flatten(1).t().cpu().numpy())[0]
v=json.load(open(P.M+'/layout_teacher/train/runs/orient/val_items.json'));items=[tuple(y.replace('/mnt/storage/lib_project/models/',P.M+'/') if isinstance(y,str) else y for y in x) for g in v.values() for x in g[:3]]
ds=ValPages(items);count=0;different=0;mean=0.;maximum=0.;shape_set=set()
with torch.no_grad():
 for j in range(min(len(ds),120)):
  img,k,photo=ds[j];h,w=img.shape[:2];s=896/max(h,w);nw=max(224,int(round(w*s/32))*32);nh=max(224,int(round(h*s/32))*32)
  x=cv2.resize(np.ascontiguousarray(img),(nw,nh),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR);x=torch.from_numpy(x).permute(2,0,1)[None].float().cuda().div_(255).sub_(.5).contiguous()
  fp=net.body(net.stem(x));q=trt_forward(x);torch.cuda.synchronize();d=(fp-q).abs();mean+=float(d.mean());maximum=max(maximum,float(d.max()));different+=decide(fp)!=decide(q);count+=1;shape_set.add((nh,nw))
  if count%20==0:print('pages',count,'decision_diff',different,'mean_abs',mean/count,'max',maximum,flush=True)
print('RESULT n',count,'decision_diff',different,'mean_abs',mean/count,'max_abs',maximum,'shapes',len(shape_set),flush=True)
assert different==0,'decision mismatch; do not promote engine'

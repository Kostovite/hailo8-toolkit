import os,sys,hashlib,random,torch,torch.nn as nn,numpy as np,tensorrt as trt
from torch.utils.data import DataLoader
D=os.path.expanduser('~/codex_trt_qat_20261119');sys.path.insert(0,D+'/layout_qat')
import photo_data as P, train_ppdl3 as TP
TP.HF=D+'/source/layout_teacher/weights/ppdl3_hf';ck=D+'/source/layout_teacher/staged_v2_20261114/r5_ord5.pt';torch.set_num_threads(3);torch.manual_seed(73);np.random.seed(73);random.seed(73)
manifest=P.manifest();val={k:[it for it in v if int(hashlib.md5(it[0].encode()).hexdigest()[:6],16)%50==0] for k,v in manifest.items()};val={k:v for k,v in val.items() if v};P.manifest=lambda:val
loader=DataLoader(TP.Mix(100,.5,{k:1 for k in val},seed=73),batch_size=1,num_workers=0,collate_fn=TP.collate)
def model():
 m=TP.build_model();m.config._attn_implementation='eager'
 for mod in m.modules():
  if hasattr(mod,'config') and hasattr(mod.config,'_attn_implementation'):mod.config._attn_implementation='eager'
 m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True));return m.eval().cuda()
fp=model();dec=model()
class Stub(nn.Module):
 def forward(self,x,mask):return [None,(x,None)]
core=dec.model;core.backbone=Stub();core.encoder_input_proj=nn.ModuleList([nn.Identity()]);core.decoder_input_proj=nn.ModuleList([nn.Identity()]*3)
logger=trt.Logger(trt.Logger.WARNING);rt=trt.Runtime(logger)
with open(D+'/artifacts/trt/layout_r5_mixed_qat250.engine','rb') as f:eng=rt.deserialize_cuda_engine(f.read())
assert eng is not None;ctx=eng.create_execution_context();print('TRT tensors',[(eng.get_tensor_name(i),tuple(eng.get_tensor_shape(eng.get_tensor_name(i)))) for i in range(eng.num_io_tensors)],flush=True)
G=[];pred=[[],[]]
with torch.no_grad():
 for idx,(x,tg) in enumerate(loader):
  x=x.cuda().contiguous();t=tg[0];bc=t['boxes'].numpy();xy=np.stack([bc[:,0]-bc[:,2]/2,bc[:,1]-bc[:,3]/2,bc[:,0]+bc[:,2]/2,bc[:,1]+bc[:,3]/2],axis=1);G.append((xy,t['class_labels'].numpy()))
  a=TP.predict(fp,x);outs=[];ctx.set_tensor_address('image',x.data_ptr())
  for name in ['src0','src1','src2','mask_feat']:
   sh=tuple(ctx.get_tensor_shape(name));assert min(sh)>0,(name,sh);v=torch.empty(sh,device='cuda',dtype=torch.float32);ctx.set_tensor_address(name,v.data_ptr());outs.append(v)
  assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
  o=dec(pixel_values=x,encoder_outputs=(outs[:3],outs[3]));prob=o.logits.sigmoid();_,Q,C=prob.shape;s,i=prob.flatten(1).topk(300,1);q=i//C;c=i%C;b=o.pred_boxes.gather(1,q[...,None].expand(-1,-1,4));b=torch.stack([b[...,0]-b[...,2]/2,b[...,1]-b[...,3]/2,b[...,0]+b[...,2]/2,b[...,1]+b[...,3]/2],-1)
  for dest,(scores,classes,boxes) in zip(pred,[a,(s,c,b)]):
   for sc,cl,box in zip(scores[0].cpu().tolist(),classes[0].cpu().tolist(),boxes[0].cpu().tolist()):
    if sc>.01:dest.append({'image_id':idx+1,'category_id':cl+1,'bbox':[box[0]*1000,box[1]*1000,(box[2]-box[0])*1000,(box[3]-box[1])*1000],'score':sc})
  if (idx+1)%20==0:print('pages',idx+1,flush=True)
for name,det in zip(['FP','TRT_QAT'],pred):
 a,p=TP.coco_ap(G,det);print('RESULT',name,'n',len(G),'AP',a,'boxAP',p['_box'],'boxAP75',p['_box75'],flush=True)

import os,sys,random,hashlib,copy,numpy as np,torch
from PIL import Image
from transformers import VisionEncoderDecoderModel,AutoImageProcessor
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
D=os.path.expanduser('~/codex_trt_qat_20261119');A=D+'/aux_models';sys.path[:0]=[A,A+'/wordlid',D+'/orient_full']
import photo_data as P
from region_router import RegionRouter
os.makedirs(A+'/router_more16',exist_ok=True);torch.set_num_threads(3);random.seed(17);np.random.seed(17);torch.manual_seed(17)
model=os.path.expanduser('~/codex_docparse_20261119/moe_models/u1/U1');processor=AutoImageProcessor.from_pretrained(model);ved=VisionEncoderDecoderModel.from_pretrained(model,torch_dtype=torch.float32);enc=ved.encoder.cuda().eval();del ved.decoder
manifest=P.manifest(); sets={False:[],True:[]}
for items in manifest.values():
 for p,l,t in items:
  held=(int(hashlib.md5(p.encode()).hexdigest()[:6],16)%50==0);sets[held].append((p,l,t))
for held,count in [(False,2048),(True,300)]:
 candidates=sets[held];random.Random(17+held).shuffle(candidates);rows=[]
 with torch.no_grad():
  for p,l,t in candidates:
   try:
    img,b,c=P.load_sample(p,l,t)
    if len(b)==0:continue
    x0,y0,x1,y1=map(int,b[len(rows)%len(b)]);crop=np.ascontiguousarray(img[max(0,y0):y1,max(0,x0):x1])
    if crop.size==0:continue
    px=processor(images=Image.fromarray(crop),return_tensors='pt').pixel_values.cuda()
    rows.append(enc(px).last_hidden_state.float().mean(1).cpu().numpy()[0])
    if len(rows)%50==0:print('heldout' if held else 'train','vectors',len(rows),flush=True)
    if len(rows)>=count:break
   except Exception as e:print('skip',type(e).__name__,flush=True)
 assert len(rows)>=(100 if held else 321),(held,len(rows));np.save(A+('/router_more16/val.npy' if held else '/router_more16/train.npy'),np.stack(rows)); print('saved', 'val' if held else 'train', len(rows), 'target', count, flush=True)

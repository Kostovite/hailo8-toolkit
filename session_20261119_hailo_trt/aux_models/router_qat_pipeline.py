import os,sys,random,hashlib,copy,numpy as np,torch
from PIL import Image
from transformers import VisionEncoderDecoderModel,AutoImageProcessor
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
D=os.path.expanduser('~/codex_trt_qat_20261119');A=D+'/aux_models';sys.path[:0]=[A,A+'/wordlid',D+'/orient_full']
import photo_data as P
from region_router import RegionRouter
os.makedirs(A+'/router',exist_ok=True);torch.set_num_threads(3);random.seed(17);np.random.seed(17);torch.manual_seed(17)
model=os.path.expanduser('~/codex_docparse_20261119/moe_models/u1/U1');processor=AutoImageProcessor.from_pretrained(model);ved=VisionEncoderDecoderModel.from_pretrained(model,torch_dtype=torch.float32);enc=ved.encoder.cuda().eval();del ved.decoder
manifest=P.manifest(); sets={False:[],True:[]}
for items in manifest.values():
 for p,l,t in items:
  held=(int(hashlib.md5(p.encode()).hexdigest()[:6],16)%50==0);sets[held].append((p,l,t))
for held,count in [(False,320),(True,100)]:
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
    if len(rows)%25==0:print('heldout' if held else 'train','vectors',len(rows),flush=True)
    if len(rows)>=count:break
   except Exception as e:print('skip',type(e).__name__,flush=True)
 assert len(rows)>=count,(held,len(rows));np.save(A+('/router/val.npy' if held else '/router/train.npy'),np.stack(rows))
del enc,ved;torch.cuda.empty_cache()
ck=torch.load(A+'/wordlid/data/grid/region_router.pt',map_location='cpu',weights_only=False);teacher=RegionRouter();teacher.load_state_dict(ck['router']);teacher.cuda().eval()
class Wrap(torch.nn.Module):
 def __init__(self,n):super().__init__();self.n=n
 def forward(self,x):
  z=self.n(x);return z['kind'],z['body'],z['lang']
ref=Wrap(teacher).cuda().eval();net=copy.deepcopy(ref);train=torch.from_numpy(np.load(A+'/router/train.npy')).cuda();val=torch.from_numpy(np.load(A+'/router/val.npy')).cuda()
def calib(m):
 with torch.no_grad():
  for x in train.split(32):m(x)
mtq.quantize(net,mtq.INT8_DEFAULT_CFG,calib);net.eval();opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=0)
for step in range(1,501):
 ix=torch.randint(len(train),(32,),device='cuda');x=train[ix]
 with torch.no_grad():y=ref(x)
 z=net(x);loss=sum((a-b).square().mean() for a,b in zip(z,y));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 if step%100==0:
  with torch.no_grad():
   a=ref(val);b=net(val);match=[float((i.argmax(1)==j.argmax(1)).float().mean()) for i,j in zip(a,b)];print('step',step,'loss',float(loss),'heldout argmax agreement',match,flush=True)
  mto.save(net,A+f'/router/router_qat_{step}.pth')
print('ROUTER_QAT_COMPLETE; no promotion without routing accuracy gate',flush=True)

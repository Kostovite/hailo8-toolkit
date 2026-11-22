import os,sys,copy,numpy as np,torch
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
D=os.path.expanduser('~/codex_trt_qat_20261119');A=D+'/aux_models';sys.path[:0]=[A,A+'/wordlid'];from region_router import RegionRouter
torch.set_num_threads(3);torch.manual_seed(17);ck=torch.load(A+'/wordlid/data/grid/region_router.pt',map_location='cpu',weights_only=False);ref=RegionRouter();ref.load_state_dict(ck['router']);ref.cuda().eval();net=copy.deepcopy(ref)
train=torch.from_numpy(np.load(A+'/router/train.npy')).cuda();val=torch.from_numpy(np.load(A+'/router/val.npy')).cuda()
cfg=copy.deepcopy(mtq.INT8_DEFAULT_CFG);cfg['quant_cfg']['*input_quantizer']={'enable':False}
def calib(m):
 with torch.no_grad():
  for x in train.split(32):m(x)
mtq.quantize(net,cfg,calib);net.eval();opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=0)
for step in range(1,501):
 x=train[torch.randint(len(train),(32,),device='cuda')]
 with torch.no_grad():a=ref(x)
 b=net(x);loss=sum((a[k]-b[k]).square().mean() for k in a);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 if step%100==0:
  with torch.no_grad():
   a=ref(val);b=net(val);match={k:float((a[k].argmax(1)==b[k].argmax(1)).float().mean()) for k in a};print('step',step,'mse',float(loss),'agreement',match,flush=True)
  mto.save(net,A+f'/router/router_mixed_qat_{step}.pth')
print('ROUTER_MIXED_QAT_DONE',flush=True)

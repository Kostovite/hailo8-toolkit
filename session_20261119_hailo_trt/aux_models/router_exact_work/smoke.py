import sys, json, os, numpy as np, torch
A=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,'/mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat')
import graph26 as G
from qatfast_router import QATNet
hn=json.load(open(A+'/router_exact.hn'));P=G.Params(A+'/router_exact_capture','cuda');net=QATNet(hn,P,'cuda');print('trainable',len(net.mac),list(net.mac),flush=True)
x=torch.from_numpy(np.load(A+'/router_exact_input_nhwc.npy')).cuda()
with torch.no_grad():out=net(x)
for n in sorted(P.conv):
 if n not in out: continue
 r=P.conv[n]
 if 'act_out' not in r:continue
 actual=out[n].detach().cpu().numpy().transpose(0,2,3,1);ref=np.asarray(r['act_out']);print('PARITY',n,actual.shape,ref.shape,'mismatch',np.count_nonzero(actual!=ref),'maxerr',np.max(np.abs(actual-ref)),flush=True)

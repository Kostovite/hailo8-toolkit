import os,numpy as np
from hailo_sdk_client import ClientRunner
D=os.path.dirname(os.path.abspath(__file__));train=np.load(D+'/router_more16_train.npy').astype(np.float32).reshape(-1,1,1,1024)
r=ClientRunner(har=D+'/router_parsed.har');h=r.get_hn_model();layers=[x.name.split('/')[-1] for x in h.stable_toposort() if 'dense' in str(x.op).lower() or 'conv' in str(x.op).lower()];assert layers
alls='model_optimization_flavor(optimization_level=2, compression_level=0)\nmodel_optimization_config(calibration, batch_size=8, calibset_size=%d)\nquantization_param([%s], precision_mode=a16_w16)\n'%(len(train),', '.join(layers))
print('training-only pooled encoder vectors',train.shape,'script',alls,flush=True);r.load_model_script(alls);r.optimize(train);r.save_har(D+'/router_more16_qat.har');print('router optimized; inspect log to verify QAT actually ran',flush=True)

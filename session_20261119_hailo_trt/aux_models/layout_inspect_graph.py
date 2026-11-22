import json,sys
from hailo_sdk_client import ClientRunner
h=ClientRunner(har=sys.argv[1]).get_hn(); h=json.loads(h) if isinstance(h,str) else h
l=h['layers'];
for n,v in l.items():
 if 'resize3' in n:
  print('\nNODE',n,json.dumps(v,default=str)[:7000],flush=True)
  for x in v.get('input',[]):print('PRED',x,json.dumps(l.get(x),default=str)[:4500],flush=True)
  for x,w in l.items():
   if n in w.get('input',[]): print('SUCC',x,json.dumps(w,default=str)[:4500],flush=True)

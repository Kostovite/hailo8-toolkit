import sys, json
from hailo_sdk_client import ClientRunner
har, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
hn = ClientRunner(har=har).get_hn_model()
L = [l.name.split("/")[-1] for l in hn.stable_toposort() if "input_layer" not in str(l.op).lower() and "output_layer" not in str(l.op).lower()]
for i in range(k):
    c = L[len(L) * i // k: len(L) * (i + 1) // k]; json.dump(c, open(f"{out}_{i}.json", "w")); print(i, len(c), c[0], c[-1])

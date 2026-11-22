#!/bin/bash
# AOT build of the fused executor kernels (hq.cu) -> hq_ext/prebuilt/hq_ext*.so, loaded by qatfast.py before any JIT.
# Exactness: NO --use_fast_math (fused integer emulation relies on IEEE rounding). Targets: sm_52 (Tesla M40),
# sm_86 (RTX 3090) + compute_86 PTX (JIT-forward for newer GPUs). Run in the torch env (conda general).
set -e; cd "$(dirname "$0")"; G=${TORCH_PY:-/mnt/storage/anaconda3/envs/general/bin/python}
rm -rf prebuilt/build; mkdir -p prebuilt/build
env -u LD_PRELOAD MAX_JOBS=8 $G - <<'P'
import os, shutil, torch
from torch.utils.cpp_extension import load
here = os.getcwd()
m = load("hq_ext", [os.path.join(here, "hq.cu")], build_directory=os.path.join(here, "prebuilt/build"), verbose=True,
         extra_cflags=["-O3", "-march=native", "-fno-plt"],
         extra_cuda_cflags=["-O3", "-Xptxas=-O3", "--extra-device-vectorization", "-lineinfo",
                            "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_86,code=sm_86",
                            "-gencode=arch=compute_86,code=compute_86"])
so = os.path.join(here, "prebuilt/build/hq_ext.so"); shutil.copy(so, os.path.join(here, "prebuilt/hq_ext.so"))
open(os.path.join(here, "prebuilt/BUILD_INFO.txt"), "w").write(
    f"torch {torch.__version__} cuda {torch.version.cuda}\nflags: -O3 -Xptxas=-O3 --extra-device-vectorization, sm_52 sm_86 compute_86, no fast-math\n")
print("built", so)
P

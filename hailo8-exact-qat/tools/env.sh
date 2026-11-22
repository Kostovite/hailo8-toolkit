# Source before running DFC python: writable SDK home, MKL for the source-built TF, CPU unless overridden.
HRE=/mnt/storage/lib_project/models/Hailo_model/hailo_re
export VIRTUAL_ENV=$HRE/sdk_home CONDA_PREFIX=$HRE/sdk_home
export LD_LIBRARY_PATH=/mnt/storage/anaconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib:/mnt/storage/anaconda3/envs/hailo_env/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-99}
HPY=/mnt/storage/anaconda3/envs/hailo_env/bin/python
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# GPU TF: system CUDA 12.6; cuDNN must be the system 9.5.1 in /usr/lib/x86_64-linux-gnu (the env's pip
# nvidia-cudnn 9.1 wheel makes TF fail with "No DNN in stream executor", so it is deliberately not added)
for _d in /usr/local/cuda-12.6/targets/x86_64-linux/lib /usr/local/cuda-12.6/extras/CUPTI/lib64; do
  [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:$LD_LIBRARY_PATH"
done
# TF's RPATH would load the env's pip cuDNN 9.1 (older than the 9.4 TF was built with -> "No DNN in stream
# executor"). Preloading the system 9.5.1 sonames makes every later dlopen reuse them.
export LD_PRELOAD="$(ls /usr/lib/x86_64-linux-gnu/libcudnn*.so.9 | tr '\n' ':')${LD_PRELOAD:-}"
# Hailo HSim.so (bit-exact sim) on Maxwell: sm_52 port of its PTX + stream-0 race fence (hsim_sm52/, build.sh).
# Must come before the real libcudart so HSim binds to the shim; verified 882/882 tensors == CPU HSim.
_HS=/mnt/storage/lib_project/models/Hailo_model/hailo_re/hsim_sm52
if [ -f $_HS/libhsim_sm52_shim.so ]; then
  export HSIM_SM52_FATBIN=$_HS/hsim_sm52.fatbin
  export LD_PRELOAD="$_HS/libhsim_sm52_shim.so:/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart.so.12:$LD_PRELOAD"
fi

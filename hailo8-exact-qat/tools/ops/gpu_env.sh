# Source (after env.sh) to run host-side ops on a CUDA GPU from the hailo env:
#   onnxruntime-gpu 1.19.2 (CUDA 12 build; sm_52 M40 OK) installed OUTSIDE the env in ortgpu/ so the DFC keeps its
#   own CPU onnxruntime; the env's pip CUDA 12 / cuDNN 9 libs; the env's NCCL preloaded (the system loader path
#   resolves libnccl to an older Maxwell build that lacks ncclCommRegister and breaks `import torch`).
# Never use GPUs 2/3 (production llama-server): CUDA_VISIBLE_DEVICES defaults to 0.
_SP=/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages
export PYTHONPATH=/mnt/storage/lib_project/models/Hailo_model/hailo_re/ortgpu${PYTHONPATH:+:$PYTHONPATH}
for _l in cudnn cublas cuda_runtime cufft curand nvjitlink cuda_nvrtc; do
  export LD_LIBRARY_PATH=$_SP/nvidia/$_l/lib:$LD_LIBRARY_PATH
done
export LD_PRELOAD=$_SP/nvidia/nccl/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}
export CUDA_VISIBLE_DEVICES=${HOST_GPU:-0}

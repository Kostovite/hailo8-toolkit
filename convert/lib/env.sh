# Shared environment helpers. Every stage runs in its own subshell; call ONE of these at its start.
CV=/mnt/storage/lib_project/models/Hailo_model/convert
PP=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
HRE=/mnt/storage/lib_project/models/Hailo_model/hailo_re
QAT=/mnt/storage/lib_project/hailo8-exact-qat
NCCL=/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
CHIP_LOCK=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119/hailo_chip.lock
VENV_HF=/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python
# Hailo DFC env. env.sh PRELOADS the system cuDNN + the sm52 shim: always PREPEND to LD_PRELOAD, never replace it
# (replacing it gives TF "No DNN in stream executor").
henv(){ source $QAT/tools/env.sh >/dev/null; export TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
        export LD_PRELOAD="$NCCL:${LD_PRELOAD:-}"; }
# torch side (conda general; same env the original exports used). No DFC preloads.
tenv(){ unset LD_PRELOAD; source /mnt/storage/anaconda3/etc/profile.d/conda.sh; conda activate general
        export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}; }
# layout_teacher venv (PP-DocLayoutV3, orient, router export); clean of DFC libs
venv(){ unset LD_PRELOAD LD_LIBRARY_PATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}; }

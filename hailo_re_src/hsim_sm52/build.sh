#!/bin/bash
# Rebuild the sm_52 port of Hailo's HSim.so kernels and the LD_PRELOAD shim.
set -e
cd "$(dirname "$0")"
B=/usr/local/cuda-12.6/bin
H=/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/hailo_sdk_client/emulator/emulator/lib/HSim.so
mkdir -p ptx && (cd ptx && $B/cuobjdump -all -xptx all $H >/dev/null)
for f in ptx/HSim.*.sm_61.ptx; do
  i=$(basename $f | cut -d. -f2)
  sed 's/^\.target sm_61/.target sm_52/' $f > ptx/m$i.sm52.ptx
  $B/ptxas -arch=sm_52 -c -O3 ptx/m$i.sm52.ptx -o ptx/m$i.o
done
$B/nvlink -arch=sm_52 ptx/m*.o -o hsim_sm52.cubin
$B/fatbinary --create=hsim_sm52.fatbin --64 --image3=kind=elf,sm=52,file=hsim_sm52.cubin
gcc -O2 -shared -fPIC hsim_sm52_shim.c -o libhsim_sm52_shim.so -ldl
echo "built $(pwd)/hsim_sm52.fatbin and libhsim_sm52_shim.so"

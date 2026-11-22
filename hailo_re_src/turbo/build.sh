#!/bin/bash
# build.sh: optimized build of the DFC compiler hook -> bin/libhailo_turbo.so (x86-64-v3 = any AVX2 CPU; no fast-math)
set -e; cd "$(dirname "$0")"; mkdir -p bin
g++ -O3 -march=x86-64-v3 -mtune=generic -DNDEBUG -std=c++17 -shared -fPIC -Wl,-O1 -Wl,--as-needed \
    -o bin/libhailo_turbo.so.tmp turbo2.cpp -ldl -lpthread
mv bin/libhailo_turbo.so.tmp bin/libhailo_turbo.so   # mv: processes that already mapped the old file keep it
strip --strip-unneeded bin/libhailo_turbo.so
echo "built $(pwd)/bin/libhailo_turbo.so"

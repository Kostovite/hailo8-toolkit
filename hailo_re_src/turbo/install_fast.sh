#!/bin/bash
# install_fast.sh [--uninstall] [--root]
# Makes EVERY Hailo DFC compile (ClientRunner.compile, hailo CLI, our pipeline) load the hook (bin/libhailo_turbo.so):
# CP-SAT memo cache (bit-exact) + HAILO_UTIL_ONLY level-max control.
#   default (no root): a .pth startup file in the DFC python's USER site-packages. Inside the DFC env only, it prepends
#     the hook to LD_PRELOAD and sets HAILO_CPCACHE (shared cache dir); the SDK passes os.environ to the compiler child.
#     HAILO_FAST=0 disables it for one run.
#   --root: also replaces hailo_tools/build/compiler with a wrapper (original kept as compiler.real); covers any user.
# Env: HPY = the DFC python (default: hailo_env), HAILO_FAST_CACHE = cache dir.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); LIB=$T/bin/libhailo_turbo.so
HPY=${HPY:-/mnt/storage/anaconda3/envs/hailo_env/bin/python}
CACHE=${HAILO_FAST_CACHE:-/mnt/storage/lib_project/models/Hailo_model/convert/compile_cache/cpsat}
USITE=$("$HPY" -c "import site;print(site.getusersitepackages())"); PREFIX=$("$HPY" -c "import sys;print(sys.prefix)")
PTH=$USITE/hailo_fast_compiler.pth
BIN=$PREFIX/lib/python3.10/site-packages/hailo_tools/build
if [ "${1:-}" = --uninstall ]; then
  rm -f "$PTH"; echo "removed $PTH"
  if [ -f "$BIN/compiler.real" ]; then sudo mv -f "$BIN/compiler.real" "$BIN/compiler"; echo "restored stock $BIN/compiler"; fi
  exit 0
fi
[ -s "$LIB" ] || "$T/build.sh"
mkdir -p "$USITE" "$CACHE"
cat > "$PTH.tmp" <<PY
import os, sys; (sys.prefix == "$PREFIX" and os.environ.get("HAILO_FAST", "1") != "0") and (os.environ.__setitem__("LD_PRELOAD", "$LIB" + (":" + os.environ["LD_PRELOAD"] if os.environ.get("LD_PRELOAD") and "$LIB" not in os.environ["LD_PRELOAD"] else "") if "$LIB" not in os.environ.get("LD_PRELOAD", "") else os.environ["LD_PRELOAD"]), os.environ.setdefault("HAILO_CPCACHE", "$CACHE"))
PY
mv "$PTH.tmp" "$PTH"; echo "installed $PTH (hook $LIB, cache $CACHE)"
if [ "${1:-}" = --root ]; then
  [ -f "$BIN/compiler.real" ] || sudo mv "$BIN/compiler" "$BIN/compiler.real"
  printf '#!/bin/sh\n[ "${HAILO_FAST:-1}" = 0 ] || export LD_PRELOAD="%s${LD_PRELOAD:+:$LD_PRELOAD}" HAILO_CPCACHE="${HAILO_CPCACHE:-%s}"\nexec "$(dirname "$0")/compiler.real" "$@"\n' "$LIB" "$CACHE" | sudo tee "$BIN/compiler" >/dev/null
  sudo chmod 755 "$BIN/compiler"; echo "wrapped $BIN/compiler (stock kept as compiler.real)"
fi

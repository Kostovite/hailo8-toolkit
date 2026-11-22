#!/bin/bash
# bench_seq.sh HEF...: b1 and b4 hw FPS, ONE chip at a time (yolo26m is PCIe-heavy: concurrent chips interfere)
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
exec 9>turbo/.chips.lock; flock 9
for h in "$@"; do for B in 1 4; do
  echo -e "$h\tb$B\t$(hailortcli benchmark $h -t 5 --batch-size $B --device-id 0000:02:00.0 2>&1 | awk -F= '/hw_only/{print $2+0; exit}')"
done; done

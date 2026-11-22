#!/bin/bash
# resume after the session restart: (1) in-distribution judge of the key cases, (2) the silent-wrong chip rerun
source /mnt/storage/lib_project/models/Hailo_model/hailo_re/env.sh
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/probe
DIST=normal $HPY -u silent/chip_vs_emu.py emu 8 silent/key.txt > silent/key_emu.log 2>&1
for d in $(cat silent/key.txt); do echo "== ${d##*/}"; DIST=normal flock ../turbo/.chips.lock $HPY silent/judge.py "$d" 2>&1 | grep -v "Warn\|\[info\]" | grep "SNR\|equal"; done > silent/key_judge.txt
done_dirs=$(python3 -c "import json;print('\n'.join(json.loads(l)['dir'] for l in open('silent/res.jsonl')))" 2>/dev/null)
grep -vxF "$done_dirs" silent/redo.txt > silent/redo2.txt
$HPY -u silent/chip_vs_emu.py emu 16 silent/redo2.txt > silent/emu3.log 2>&1
DEVICE=0000:03:00.0 $HPY -u silent/chip_vs_emu.py chip silent/res.jsonl silent/redo2.txt > silent/chip3.log 2>&1
echo DONE > silent/resume.done

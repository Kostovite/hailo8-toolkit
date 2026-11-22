#!/bin/bash
# chip 0 queue (per-chip lock): in-distribution judge -> silent-wrong rerun -> fuzz infra re-check
source /mnt/storage/lib_project/models/Hailo_model/hailo_re/env.sh
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/probe
export DEVICE=0000:02:00.0
LK=../turbo/.chip_0000_02_00.0.lock
DIST=normal $HPY -u silent/chip_vs_emu.py emu 8 silent/key.txt > silent/key_emu.log 2>&1
for d in $(cat silent/key.txt); do echo "== ${d##*/}"; DIST=normal flock $LK $HPY silent/judge.py "$d" 2>&1 | grep -v "Warn\|\[info\]" | grep "SNR\|equal"; done > silent/key_judge.txt
python3 -c "import json;print('\n'.join(json.loads(l)['dir'] for l in open('silent/res.jsonl')))" > silent/done_dirs.txt
grep -vxF -f silent/done_dirs.txt silent/redo.txt > silent/redo2.txt
$HPY -u silent/chip_vs_emu.py emu 16 silent/redo2.txt > silent/emu3.log 2>&1
$HPY -u silent/chip_vs_emu.py chip silent/res.jsonl silent/redo2.txt > silent/chip3.log 2>&1
$HPY -u recheck_infra.py runs/recheck_infra.jsonl runs/fz1/fuzz.jsonl runs/fz2/fuzz.jsonl > runs/recheck_infra.out 2>&1
echo DONE > silent/resume2.done

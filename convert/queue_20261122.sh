#!/bin/bash
# Sequential queue (owner rule: one model at a time). Run inside: systemd-run --user --scope -p MemoryMax=120G -p CPUQuota=4800%
set -u; C=/mnt/storage/lib_project/models/Hailo_model/convert; cd $C; S=0,1,2,3,0,1,2,3   # 8 emulator shards
B=/mnt/storage/lib_project/models/layout_teacher/staged_v2_20261114
log(){ echo "[$(date '+%F %T')] $*" | tee -a $C/runs/queue.log; }
resume_q(){ R=$C/runs/$1; rm -f $R/emu_q.s* $R/emu_q_*.npy
  OMP_NUM_THREADS=2 bash lib/emu_shard.sh $R/q.har $R/eval_x.npy $R/emu_q q $S > $R/logs/emu_q_resume.log 2>&1 && touch $R/.done_emu; }
log "flat3d: quantized emulator (8 shards)"; resume_q flat3d_t1 && bash convert.sh flat3 $B/flat3d_2500.pt $C/runs/flat3d_t1 3 >> runs/flat3d_t1.out 2>&1; log "flat3d done: $(tail -n 1 runs/flat3d_t1/logs/run.log)"
log "orient3: quantized emulator (8 shards)"; resume_q orient3_t1 && bash convert.sh orient3 $B/orient3_10000.pt $C/runs/orient3_t1 2 >> runs/orient3_t1.out 2>&1; log "orient3 done: $(tail -n 1 runs/orient3_t1/logs/run.log)"
log "gridnet: resume at gate_q"; bash convert.sh gridnet /mnt/storage/lib_project/models/Dolphin/wordlid/data/grid/table_structure_final.pt $C/runs/gridnet_t1 0,1 gate_q >> runs/gridnet_t1.out 2>&1; log "gridnet done: $(tail -n 1 runs/gridnet_t1/logs/run.log)"
log "QUEUE FINISHED"

#!/bin/bash
S=/mnt/storage/lib_project/models/Hailo_model/hailo_re/small
until [ -f $S/y26n_v14sec/weights/best.pt ] && ! pgrep -f "[y]olo detect train" >/dev/null; do sleep 30; done
rm -f $S/L0_ready
echo "SMALL train done $(date)"; tr '\r' '\n' < $S/train_y26n.log | grep -E "^ +all " | tail -1
$S/lane_b.sh > $S/lane_b.log 2>&1 &
$S/lane_a.sh > $S/lane_a.log 2>&1
wait
echo "SMALL DONE $(date)"

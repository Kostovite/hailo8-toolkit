#!/bin/bash
# usage: run_pair.sh HAR CTX GPU LIMIT -- vtval then g5val
cd /mnt/storage/lib_project/models/Hailo_model/gridnet_hailo
./run_e.sh $1 $2 vtval $4 $3; ./run_e.sh $1 $2 g5val $4 $3

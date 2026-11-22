#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
REMOTE=user@gpu-box
R=codex_trt_qat_20261119
FILES="$D/scheduler/trt_candidates.files"
SSH=(ssh -o BatchMode=no -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)
export SSHPASS
SSHPASS=$(awk -F'`' '/\| password /{print $2;exit}' /mnt/storage/lib_project/models/TRAINING_SERVER.md)
[ -n "$SSHPASS" ] || { echo 'STOP missing SSH password configuration';exit 1; }
remote() { sshpass -e "${SSH[@]}" "$REMOTE" "$@"; }
echo 'Waiting for Flat3D build and validation; no artifacts will be copied on failure.'
while true; do
 if result=$(remote "test ! -f ~/$R/logs/flat_auto.exit || cat ~/$R/logs/flat_auto.exit" 2>/dev/null);then
  [ -z "$result" ] || break
 else echo '3090 unreachable; will retry';fi
 sleep 30
done
[ "$result" = 0 ] || { echo "STOP: Flat3D automatic build/validation failed (exit $result); see 3090 logs/flat_auto.log";exit 1; }
remote "test -f ~/$R/artifacts/trt/flat3d_headfp32_VALIDATED && test \$(cat ~/$R/logs/orient_trt_test.exit) = 0 && test \$(cat ~/$R/logs/layout_trt_ap.exit) = 0" || { echo 'STOP: a required parity/AP gate is missing';exit 1; }
# Guard the observed preliminary layout score against materially degrading relative to FP.
scores=$(remote "grep '^RESULT ' ~/$R/logs/layout_trt_ap.log")
fp=$(awk '$2=="FP"{print $6}' <<< "$scores")
qat=$(awk '$2=="TRT_QAT"{print $6}' <<< "$scores")
awk -v a="$fp" -v b="$qat" 'BEGIN {if (a=="" || b=="" || b+0 < a-.01) exit 1}' || { echo 'STOP: layout detection gate missing or AP drop > 0.01';exit 1; }

dest=/mnt/backup/Dolphin_training/codex_docparse_trt_candidates_20261120
[ ! -e "$dest" ] || { echo "STOP: destination exists, not overwriting $dest";exit 1; }
staging=$(mktemp -d /mnt/backup/Dolphin_training/.codex_trt_staging_XXXXXXXX)
echo "Copying to staging $staging"
for n in 1 2 3; do
 if sshpass -e rsync -a --partial --files-from="$FILES" -e 'ssh -o ConnectTimeout=10' "$REMOTE:$R/" "$staging/";then break;fi
 [ "$n" -lt 3 ] || { echo "STOP transfer failed; staging retained at $staging";exit 1; }; sleep 20
done
remote "cd ~/$R && xargs -d '\n' sha256sum" < "$FILES" > "$staging/SHA256SUMS"
(cd "$staging" && sha256sum -c SHA256SUMS)
cat > "$staging/README.txt" <<'DOC'
PRELIMINARY TENSORRT CANDIDATES, not a complete Docparse package.
TensorRT 10.13.3.9; built on RTX 3090. Plan files can be GPU, platform and runtime version specific.
Orient3 engine covers feature extractor only; host decision head required.
Layout engine covers backbone/encoder only; host decoder and reading-order path required.
Flat3D output head is FP32; remaining graph contains QAT Q/DQ.
See included logs for limited-sample parity/accuracy checks and scripts; full benchmarks, integration and deployment approval remain.
Not evidence of maximum possible accuracy. QAT checkpoints and original staged checkpoints included.
DOC
mv "$staging" "$dest"
echo "COPIED AND SHA256 VERIFIED: $dest (preliminary candidates only)"

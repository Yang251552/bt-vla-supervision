#!/usr/bin/env bash
# GR00T N1.5 RoboCasa365 checkpoint (HF robocasa/robocasa365_checkpoints, tagged apache-2.0, but fine-tuned from
# nvidia/GR00T-N1.5-3B, whose NVIDIA licence limits it and derivatives to non-commercial research/evaluation), only
#   gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000, without optimizer.pt.
# huggingface_hub.snapshot_download resumes and skips finished files, so a rerun after a stop continues.
# Public repo (assumed); no token is read or stored. Log ~/setup_logs/fetch_checkpoint.log.
source "$(dirname "$0")/lib.sh"
start_log fetch_checkpoint
on_root "$CKPT_ROOT" "$HOME/.cache/huggingface"
t0=$(date +%s)
"$G_PY" - "$CKPT_ROOT" "$CKPT_SUB" <<'PY'
import sys
from huggingface_hub import snapshot_download
root, sub = sys.argv[1:3]
snapshot_download(repo_id="robocasa/robocasa365_checkpoints", revision="c484448aba1a9b60a04c9b0ca117241518ea69f3",
                  allow_patterns=[f"{sub}/*"],
                  ignore_patterns=["*optimizer.pt"], local_dir=root)
PY
ls "$CKPT/config.json" >/dev/null && ls "$CKPT"/*.safetensors >/dev/null \
  || { log "FATAL: $CKPT lacks config.json or *.safetensors"; exit 1; }
find "$CKPT" -maxdepth 2 -type f -printf '%10s %p\n' | sort -k2
log "=== fetch_checkpoint ok: $(du -sh "$CKPT" | cut -f1) in $(( $(date +%s) - t0 )) s; $(df -h --output=avail / | tail -1) free on /"

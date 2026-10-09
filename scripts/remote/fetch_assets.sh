#!/usr/bin/env bash
# RoboCasa kitchen assets into the robocasa source tree (root EBS volume), light subset first:
#   tex fixtures_lw objs_objaverse objs_lw = what the atomic tasks sample from (Kitchen's default
#   obj_registries=("objaverse", "lightwheel") [V kitchen.py]; generative textures off by default
#   [V env_utils.create_env generative_textures=None]); not fetched: objs_aigen, tex_generative.
# Then sim_server.py --check-assets builds the task and resets 10 seeds (target split: layouts 1-10); if that
# fails, the rest is fetched and the check repeated. Instances are enumerated from the folders on disk
# [V kitchen_object_utils.py], so a missing registry would silently change the object set: the folders
# above must be non-empty. Idempotent (per-type markers), log ~/setup_logs/fetch_assets.log.
#   bash scripts/remote/fetch_assets.sh [TASK]
source "$(dirname "$0")/lib.sh"
start_log fetch_assets
TASK="${1:-PickPlaceCounterToStove}"
ASSETS="$SRC_DIR/robocasa/robocasa/models/assets"
on_root "$ASSETS"

fetch() {   # same URL, folder and unzip location as robocasa's download_and_extract_zip, but fails loudly
  "$RC_PY" - "$1" <<'PY'
import os, sys
from zipfile import ZipFile
from robocasa.scripts.download_kitchen_assets import DOWNLOAD_ASSET_REGISTRY, download_url
cfg = DOWNLOAD_ASSET_REGISTRY[sys.argv[1]]
parent = os.path.abspath(os.path.join(cfg["folder"], os.pardir))
os.makedirs(parent, exist_ok=True)
zpath = os.path.join(parent, os.path.basename(cfg["folder"]) + ".zip")
download_url(cfg["url"], parent, fname=os.path.basename(zpath), check_overwrite=False)
with ZipFile(zpath) as z:
    z.extractall(parent)
os.remove(zpath)
print(sys.argv[1], "->", cfg["folder"], sum(len(f) for _, _, f in os.walk(cfg["folder"])), "files")
PY
}
check() {
  "$RC_PY" "$BTVLA/sim/sim_server.py" --check-assets --task "$TASK" --seeds 0-9 \
    --out "$LOG_DIR/check_assets_$TASK.json"
}
t0=$(date +%s)
for t in tex fixtures_lw objs_objaverse objs_lw; do step "asset_$t" fetch "$t"; done
for d in objects/objaverse objects/lightwheel fixtures textures; do
  [[ -n "$(find "$ASSETS/$d" -mindepth 1 -maxdepth 2 -print -quit 2>/dev/null)" ]] || { log "FATAL: $ASSETS/$d empty"; exit 1; }
done
if ! check; then
  log "light subset failed the $TASK check: fetching the remaining types"
  for t in objs_aigen tex_generative; do step "asset_$t" fetch "$t"; done
  check || { log "FATAL: $TASK still fails with all assets (not an asset problem?)"; exit 1; }
fi
du -sh "$ASSETS"/* 2>/dev/null | tee -a "$LOG_DIR/versions.txt"
log "=== fetch_assets ok in $(( $(date +%s) - t0 )) s; $(df -h --output=avail / | tail -1) free on /"

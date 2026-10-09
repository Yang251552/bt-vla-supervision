# Sourced by scripts/remote/*.sh on the instance. Steps are idempotent: each finished step leaves
# ~/setup_logs/<step>.done, so a rerun after a stop mid-install resumes at the first unfinished step
# (delete a marker to force that step). Logs: ~/setup_logs/<script>.log, versions: ~/setup_logs/versions.txt.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
mkdir -p "$LOG_DIR"

log() { echo "[$(date -u +%FT%TZ)] $*"; }
start_log() {
  exec > >(tee -a "$LOG_DIR/$1.log") 2>&1
  log "=== $1 start on $(hostname); df /: $(df -h --output=avail / | tail -1) free"
}
step() {   # step NAME CMD...
  local name="$1"; shift
  if [[ -f "$LOG_DIR/$name.done" ]]; then log "skip $name (done)"; return 0; fi
  log "run  $name"
  "$@"
  date -u +%FT%TZ > "$LOG_DIR/$name.done"
  log "done $name"
}
on_root() {   # every path must be on the root EBS volume: the NVMe instance store is wiped on stop
  local root_src d
  root_src="$(df --output=source / | tail -1)"
  for d in "$@"; do
    mkdir -p "$d"
    [[ "$(df --output=source "$d" | tail -1)" == "$root_src" ]] || { log "FATAL: $d is not on the root volume"; exit 1; }
  done
}
apt_install() {   # waits for unattended-upgrades instead of failing on the dpkg lock
  sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 update
  sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=900 install -y --no-install-recommends "$@"
}
clone() {   # clone URL DIR SHA: exact commit, detached (shallow fetch by sha); atomic (tmp dir + mv), records it.
  local url="$1" dir="$2" sha="$3"   # an existing DIR must be a repository of its own (not a plain dir in another one)
  if [[ ! -e "$dir" ]]; then
    rm -rf "$dir.tmp"
    git init -q "$dir.tmp"
    git -C "$dir.tmp" fetch --depth 1 "$url" "$sha"
    git -C "$dir.tmp" checkout -q --detach FETCH_HEAD
    mv "$dir.tmp" "$dir"
  fi
  [[ -d "$dir" && "$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null)" == "$(cd "$dir" && pwd -P)" ]] \
    || { log "FATAL: $dir exists but is not a git repository of its own"; exit 1; }
  [[ "$(git -C "$dir" rev-parse HEAD)" == "$sha" ]] || { log "FATAL: $dir is not at $sha"; exit 1; }
  echo "$(date -u +%F) $(basename "$dir") $(git -C "$dir" rev-parse HEAD)" >> "$LOG_DIR/versions.txt"
}

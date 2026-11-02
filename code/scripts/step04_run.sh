#!/usr/bin/env bash
# Host orchestration only. Python and encoding stay in the existing uv image.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode=${1:?mode}; run=${2:?unique run ID}; trial=${3:-0}
[[ "$run" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$trial" =~ ^[0-4]$ ]] || exit 2
case "$mode" in smoke|record|validate|replay) ;; *) exit 2;; esac
base="artifacts/step04/$run"
mkdir -p "$base"
log="$base/${mode}_${trial}"
test ! -e "$log.log" || { echo 'Existing invocation: no overwrite' >&2; exit 2; }
compose=(docker compose -f compose.yaml)
limit=300
if [[ "$mode" == smoke || "$mode" == record ]]; then
  : "${PAI_GPU_DEVICE:=0}"
  export PAI_GPU_DEVICE
  [[ "$PAI_GPU_DEVICE" =~ ^[0-9]+$ ]] || exit 2
  # Multiple observations, stable VRAM, low sustained load, and substantial headroom.
  # This conservative gate is for this bounded rendering job, not a shared GPU policy.
  for sample in 1 2 3; do
    date -Iseconds >> "$log.preflight.txt"
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits >> "$log.preflight.txt"
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{n++; m+=$1} END {print "compute_processes="n+0", memory_mib="m+0}' >> "$log.preflight.txt"
    [[ "$sample" == 3 ]] || sleep 2
  done
  awk -F, '/^[0-9]+,/ {if ($2>30 || $4<8192) bad=1; if(n && ($3-prev>1024 || prev-$3>1024)) bad=1; prev=$3; n++} END {exit(bad || n!=3)}' "$log.preflight.txt" || { echo 'GPU gate deferred: see preflight' >&2; exit 3; }
  printf '%s\n' 'Start under current user conditional preauthorization, no new individual approval. Three low-load stable observations; >=8GiB free; 5-frame smoke or one <=2750-frame episode, 640x480 at 50Hz; one GPU; 180s recording wall cap plus <=90s CPU encoding; no GPU encoding. Stop for resource competition, reserve loss, warnings, timeout or >6GiB run output.' >> "$log.preflight.txt"
else
  compose+=(-f compose.step02-cpu.yaml)
  limit=600
fi
"${compose[@]}" config --format json | jq -e '.services.pai | (.cpus == 2) and (.mem_limit == "8589934592") and (.network_mode == "none") and (.pids_limit == 256)' >/dev/null
df -B1 . /var/lib/docker > "$log.disk_before.txt"
docker image inspect pai-advanced1-final:step01 --format '{{.Id}}' > "$log.image_id.txt"
root="data/step04/$run"
out="$base"
if [[ "$mode" == smoke ]]; then root="$base/smoke_dataset"; out="$base/smoke_diagnostics"; fi
name="pai-step04-${run}-${mode}-${trial}"
"${compose[@]}" run --name "$name" --rm --no-deps --pull never pai \
  timeout --signal=TERM --kill-after=5s "${limit}s" python -u scripts/step04.py "$mode" --root "$root" --output "$out" --trial "$trial" \
  > "$log.log" 2>&1 &
job=$!
cleanup() {
  if kill -0 "$job" 2>/dev/null; then docker stop -t 5 "$name" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT INT TERM
while kill -0 "$job" 2>/dev/null; do
  date -Iseconds >> "$log.monitor.txt"
  if [[ "$mode" == smoke || "$mode" == record ]]; then
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits >> "$log.monitor.txt" || { cleanup; exit 3; }
    free_mib=$(tail -1 "$log.monitor.txt" | awk -F, '{print $4+0}')
    if (( free_mib < 4096 )); then echo 'VRAM reserve lost' >> "$log.monitor.txt"; cleanup; exit 3; fi
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{n++; m+=$1} END {print "compute_processes="n+0", memory_mib="m+0}' >> "$log.monitor.txt"
  fi
  docker stats --no-stream --format '{{.CPUPerc}} {{.MemUsage}} {{.PIDs}}' "$name" >> "$log.monitor.txt" 2>/dev/null || true
  sleep 5
done
set +e
wait "$job"
status=$?
set -e
trap - EXIT INT TERM
df -B1 . /var/lib/docker > "$log.disk_after.txt"
if [[ "$mode" == smoke || "$mode" == record ]]; then
  nvidia-smi -i "$PAI_GPU_DEVICE" --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits > "$log.gpu_after.txt"
fi
tail -5 "$log.log"
exit "$status"

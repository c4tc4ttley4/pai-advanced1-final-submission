#!/usr/bin/env bash
# Host orchestration only; all Python stays in the existing Docker + uv image.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode=${1:?audit|smoke|train|continue|reload|rollout}; run=${2:?unique run ID}; trial=${3:-0}; checkpoint=${4:-final}
[[ "$run" =~ ^[A-Za-z0-9_-]+$ && "$trial" =~ ^[0-2]$ ]] || exit 2
case "$mode" in audit|smoke|train|continue|reload|rollout) ;; *) exit 2;; esac
[[ "$checkpoint" == final || "$checkpoint" == continued ]] || exit 2
base="artifacts/step05A/$run"; mkdir -p "$base"
log="$base/${mode}_${trial}"
test ! -e "$log.log" || { echo 'Existing invocation; no overwrite' >&2; exit 2; }
cfg=configs/step05A_act.json
reserve=$(jq -r .min_disk_free_bytes "$cfg")
output_limit=$(jq -r .max_output_bytes "$cfg")
check_disk() {
  available=$(df -B1 --output=avail . | tail -1)
  size=$(du -sb "$base" "models/step05A/$run" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  ((available >= reserve && size < output_limit))
}
check_disk || { echo 'disk reserve/output limit' >&2; exit 3; }
compose=(docker compose -f compose.yaml)
limit=600
if [[ "$mode" == audit ]]; then
  compose+=(-f compose.step02-cpu.yaml)
else
  test -z "$(docker ps -q --filter name=pai-step05A-)" || { echo 'Another STEP 5A job is running' >&2; exit 3; }
  : "${PAI_GPU_DEVICE:=0}"; export PAI_GPU_DEVICE
  [[ "$PAI_GPU_DEVICE" =~ ^[0-9]+$ ]] || exit 2
  for sample in 1 2 3; do
    date -Iseconds >> "$log.preflight.txt"
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits >> "$log.preflight.txt"
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$log.baseline_apps.csv"
    cat "$log.baseline_apps.csv" >> "$log.preflight.txt"
    [[ "$sample" == 3 ]] || sleep 2
  done
  awk -F, '/^[0-9]+, GPU-/ {if ($4>30 || $6<16384) bad=1; if(n && ($5-prev>1024 || prev-$5>1024)) bad=1; prev=$5; n++} END {exit(bad || n!=3)}' "$log.preflight.txt" || { echo 'GPU gate deferred' >&2; exit 3; }
  printf '%s\n' 'Current conditional authorization: three low-load stable samples, >=16GiB free. One GPU job; CPU2/RAM8GiB/SHM1GiB/PID256/network none/nonroot. Initial ACT fp32 full resolution estimate <8GiB VRAM and <2GiB checkpoint output; smoke measures actual. Stop for new external compute PID, external VRAM increase >1GiB, free VRAM <8GiB, disk/output/time cap. Observation is not a guarantee of isolation.' >> "$log.preflight.txt"
  used=0
  if [[ -f "$base/active_seconds.tsv" ]]; then used=$(awk '{s+=$2} END {printf "%.0f",s}' "$base/active_seconds.tsv"); fi
  remaining=$((2700-used))
  ((remaining > 0)) || { echo '45 minute cumulative cap reached' >&2; exit 3; }
  case "$mode" in smoke) limit=300;; train|continue) limit=1800;; reload) limit=180;; rollout) limit=600;; esac
  ((limit <= remaining)) || limit=$remaining
fi
"${compose[@]}" config --format json | jq -e '.services.pai | (.cpus == 2) and (.mem_limit == "8589934592") and (.network_mode == "none") and (.pids_limit == 256)' >/dev/null
df -B1 . /var/lib/docker > "$log.disk_before.txt"
docker image inspect pai-advanced1-final:step01 --format '{{.Id}}' > "$log.image_id.txt"
sha256sum src/step05A_act.py src/step05A_rollout.py scripts/step05A.py scripts/step05A_run.sh configs/step05A_act.json docs/step05A_training_spec.md > "$log.sources.sha256"
name="pai-step05A-${run}-${mode}-${trial}"
start=$(date +%s)
"${compose[@]}" run --name "$name" --rm --no-deps --pull never \
  -v "$PWD/data:/workspace/data:ro" \
  -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro" \
  -v "$PWD/configs:/workspace/configs:ro" -v "$PWD/scenes:/workspace/scenes:ro" \
  -v "$PWD/third_party:/workspace/third_party:ro" \
  -e TORCH_HOME=/workspace/.cache/torch -e HF_DATASETS_OFFLINE=1 \
  pai timeout --signal=TERM --kill-after=5s "${limit}s" \
  uv run --no-sync python -u scripts/step05A.py "$mode" --run "$run" --trial "$trial" --checkpoint "$checkpoint" \
  > "$log.log" 2>&1 &
job=$!
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
monitor_reason=''
while kill -0 "$job" 2>/dev/null; do
  date -Iseconds >> "$log.monitor.txt"
  if [[ "$mode" != audit ]]; then
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits >> "$log.monitor.txt" || { monitor_reason='GPU observation failed'; cleanup; break; }
    free_mib=$(tail -1 "$log.monitor.txt" | awk -F, '{print $4+0}')
    if ((free_mib < 8192)); then monitor_reason='VRAM reserve lost'; cleanup; break; fi
    docker top "$name" -eo pid 2>/dev/null | awk 'NR>1 {print $1}' > "$log.own_pids.txt" || true
    cat "$log.own_pids.txt" >> "$log.own_pids_seen.txt"
    sort -un "$log.own_pids_seen.txt" -o "$log.own_pids_seen.txt"
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$log.current_apps.csv"
    cat "$log.current_apps.csv" >> "$log.monitor.txt"
    if [[ -s "$log.own_pids_seen.txt" ]]; then
      awk -F, 'NR==FNR {own[$1+0]=1; next} !own[$1+0] {print}' "$log.own_pids_seen.txt" "$log.current_apps.csv" > "$log.external_candidates.csv"
      # NVML can briefly retain a Python PID after it has exited. Check liveness,
      # and retain all previously observed own descendants across snapshots.
      : > "$log.external_apps.csv"
      while IFS=, read -r candidate memory; do
        if [[ "$candidate" =~ ^[0-9]+$ ]] && kill -0 "$candidate" 2>/dev/null; then
          printf '%s,%s\n' "$candidate" "$memory" >> "$log.external_apps.csv"
        fi
      done < "$log.external_candidates.csv"
      if ! awk -F, 'FILENAME==ARGV[1] {base[$1+0]=$2+0; b+=$2; next} {if(!($1+0 in base)) bad=1; c+=$2} END {exit(bad || c>b+1024)}' "$log.baseline_apps.csv" "$log.external_apps.csv"; then
        monitor_reason='New external compute process or external VRAM growth'; cleanup; break
      fi
    fi
  fi
  if ! check_disk; then monitor_reason='Disk/output limit'; cleanup; break; fi
  docker stats --no-stream --format '{{.CPUPerc}} {{.MemUsage}} {{.PIDs}}' "$name" >> "$log.monitor.txt" 2>/dev/null || true
  docker inspect "$name" --format '{{json .HostConfig}}' 2>/dev/null | jq '{NanoCpus,Memory,ShmSize,PidsLimit,NetworkMode,DeviceRequests,CapDrop,SecurityOpt}' > "$log.runtime_limits.json" || true
  sleep 3
done
set +e
wait "$job"; status=$?
set -e
if [[ -n "$monitor_reason" ]]; then status=3; fi
trap - EXIT INT TERM
elapsed=$(($(date +%s)-start))
if [[ "$mode" != audit ]]; then printf '%s %s %s\n' "${mode}_${trial}" "$elapsed" "$status" >> "$base/active_seconds.tsv"; fi
printf '%s\n' "exit=$status elapsed_s=$elapsed monitor_stop=$monitor_reason" > "$log.exit.txt"
df -B1 . /var/lib/docker > "$log.disk_after.txt"
nvidia-smi -i "${PAI_GPU_DEVICE:-0}" --query-gpu=index,uuid,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits > "$log.gpu_after.txt"
tail -5 "$log.log"
exit "$status"

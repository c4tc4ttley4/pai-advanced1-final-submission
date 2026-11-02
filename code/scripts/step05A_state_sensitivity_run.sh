#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
base=artifacts/step05A_state_sensitivity/state_sensitivity_20261031_01
name=pai-state-sensitivity-20261031-01
[[ ! -e "$base/attempt_started.txt" ]] || { echo 'No automatic retry'; exit 2; }
[[ -z "$(docker ps -aq --filter name=pai-state-sensitivity)" ]] || exit 3
[[ -z "$(docker ps -q --filter name=pai-jump-)" ]] || exit 3
export PAI_GPU_DEVICE=0
check_disk() {
  local available docker_available size
  available=$(df -B1 --output=avail . | tail -1)
  docker_available=$(df -B1 --output=avail /var/lib/docker | tail -1)
  size=$(du -sb "$base" | awk '{print $1}')
  ((available>=10737418240 && docker_available>=10737418240 && size<134217728))
}
check_disk
for sample in 1 2 3; do
  TZ=Asia/Tokyo date -Iseconds >> "$base/gpu_preflight.txt"
  nvidia-smi -i 0 --query-gpu=index,uuid,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits >> "$base/gpu_preflight.txt"
  nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$base/baseline_apps.csv"
  cat "$base/baseline_apps.csv" >> "$base/gpu_preflight.txt"
  [[ "$sample" == 3 ]] || sleep 2
done
awk -F, '/^0, GPU-/ {if($3>30 || $5<16384)bad=1; if(n && ($4-prev>1024 || prev-$4>1024))bad=1; prev=$4;n++} END{exit(bad || n!=3)}' "$base/gpu_preflight.txt"
docker compose -f compose.yaml config --format json | jq -e '.services.pai | (.cpus == 2) and (.mem_limit == "8589934592") and (.shm_size == "1073741824") and (.network_mode == "none") and (.pids_limit == 256) and (.user | startswith("0:") | not)' >/dev/null
sha256sum scripts/step05A_state_sensitivity.py scripts/step05A_state_sensitivity_run.sh > "$base/new_scripts.sha256"
start=$(date +%s)
TZ=Asia/Tokyo date -Iseconds > "$base/attempt_started.txt"
docker compose -f compose.yaml run --name "$name" --rm --no-deps --pull never \
  -v "$PWD:/workspace:ro" -v "$PWD/$base:/workspace/$base:rw" \
  -e HF_HOME=/tmp/huggingface -e HF_DATASETS_OFFLINE=1 -e XDG_CACHE_HOME=/tmp/cache \
  -e UV_CACHE_DIR=/tmp/uv -e TORCH_HOME=/tmp/torch -e PYTHONDONTWRITEBYTECODE=1 \
  -e MUJOCO_GL=disable \
  pai timeout --signal=TERM --kill-after=5s 100s \
  uv run --no-sync python -u scripts/step05A_state_sensitivity.py > "$base/container.log" 2>&1 &
job=$!
cleanup() { docker stop -t 3 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
reason=''
while kill -0 "$job" 2>/dev/null; do
  if (($(date +%s)-start>=110)); then reason='wall clock reserve'; cleanup; break; fi
  TZ=Asia/Tokyo date -Iseconds >> "$base/monitor.txt"
  if ! nvidia-smi -i 0 --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits >> "$base/monitor.txt"; then reason='GPU measurement failed'; cleanup; break; fi
  free_mib=$(tail -1 "$base/monitor.txt" | awk -F, '{print $4+0}')
  if ((free_mib<8192)); then reason='VRAM reserve'; cleanup; break; fi
  if docker inspect "$name" >/dev/null 2>&1; then
    docker top "$name" -eo pid 2>/dev/null | awk 'NR>1{print $1}' >> "$base/own_pids.txt" || true
    sort -un "$base/own_pids.txt" -o "$base/own_pids.txt"
    nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$base/current_apps.csv" || { reason='compute measurement failed'; cleanup; break; }
    awk -F, 'FILENAME==ARGV[1]{own[$1+0]=1;next} !own[$1+0]{print}' "$base/own_pids.txt" "$base/current_apps.csv" > "$base/external_apps.csv"
    if ! awk -F, 'FILENAME==ARGV[1]{base[$1+0]=$2+0;b+=$2;next} {if(!($1+0 in base))bad=1;c+=$2} END{exit(bad || c>b+1024)}' "$base/baseline_apps.csv" "$base/external_apps.csv"; then reason='external compute change'; cleanup; break; fi
    docker stats --no-stream --format '{{.CPUPerc}} {{.MemUsage}} {{.PIDs}}' "$name" >> "$base/monitor.txt" 2>/dev/null || true
  fi
  if ! check_disk; then reason='disk/output limit'; cleanup; break; fi
  sleep 2
done
set +e
wait "$job"; status=$?
set -e
[[ -z "$reason" ]] || status=3
trap - EXIT INT TERM
elapsed=$(($(date +%s)-start))
printf 'gpu %s %s\n' "$elapsed" "$status" > "$base/active_seconds.tsv"
printf 'exit=%s elapsed_s=%s monitor_stop=%s\n' "$status" "$elapsed" "$reason" > "$base/exit.txt"
TZ=Asia/Tokyo date -Iseconds > "$base/attempt_ended.txt"
check_disk || status=3
((elapsed<=120)) || status=3
tail -5 "$base/container.log"
exit "$status"

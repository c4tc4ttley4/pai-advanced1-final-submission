#!/usr/bin/env bash
# Diagnostic modes only. Never invoke the old training launcher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode=${1:-audit}; invocation=${2:-0}
case "$mode" in audit|capture|offline|summarize) ;; *) exit 2;; esac
[[ "$invocation" =~ ^[0-9]+$ ]] || exit 2
run=$(jq -r .run_id configs/step05A_jump_diagnosis.json)
base="artifacts/step05A_jump_diagnosis/$run"; test -d "$base"
log="$base/${mode}_${invocation}"
test ! -e "$log.log" || { echo 'Invocation exists: no overwrite' >&2; exit 2; }
test -z "$(docker ps -q --filter name=pai-jump-)" || { echo 'Diagnosis already running' >&2; exit 3; }
check_disk() {
  available=$(df -B1 --output=avail . | tail -1)
  docker_available=$(df -B1 --output=avail /var/lib/docker | tail -1)
  size=$(du -sb "$base" | awk '{print $1}')
  ((available>=10737418240 && docker_available>=10737418240 && size<1073741824))
}
check_disk || { echo 'disk reserve/output limit' >&2; exit 3; }
compose=(docker compose -f compose.yaml)
limit=300; gpu=false
if [[ "$mode" == capture || "$mode" == offline ]]; then
  gpu=true
  : "${PAI_GPU_DEVICE:=0}"; export PAI_GPU_DEVICE
  [[ "$PAI_GPU_DEVICE" =~ ^[0-9]+$ ]] || exit 2
  for sample in 1 2 3; do
    date -Iseconds >> "$log.preflight.txt"
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits >> "$log.preflight.txt"
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$log.baseline_apps.csv"
    cat "$log.baseline_apps.csv" >> "$log.preflight.txt"
    [[ "$sample" == 3 ]] || sleep 2
  done
  awk -F, '/^[0-9]+, GPU-/ {if($4>30 || $6<16384) bad=1; if(n && ($5-prev>1024 || prev-$5>1024)) bad=1; prev=$5; n++} END {exit(bad || n!=3)}' "$log.preflight.txt" || { echo 'GPU gate deferred' >&2; exit 3; }
  used=0
  if [[ -f "$base/active_seconds.tsv" ]]; then used=$(awk '{s+=$2} END {print s+0}' "$base/active_seconds.tsv"); fi
  remaining=$((900-used)); ((remaining>0)) || exit 3
  ((limit<=remaining)) || limit=$remaining
  printf '%s\n' 'Conditional user authorization; three low-load stable observations, >=16GiB free; same model and stopping gate. Sequential batch1 diagnostics; estimated <2GiB VRAM and <100MiB new output. Stop for new external compute PID/growth >1GiB, free<8GiB, disk/output/time cap. Discrete observations are not isolation guarantees.' >> "$log.preflight.txt"
else
  compose+=(-f compose.step02-cpu.yaml)
fi
"${compose[@]}" config --format json | jq -e '.services.pai | (.cpus == 2) and (.mem_limit == "8589934592") and (.network_mode == "none") and (.pids_limit == 256)' >/dev/null
name="pai-jump-${run}-${mode}-${invocation}"
start=$(date +%s)
"${compose[@]}" run --name "$name" --rm --no-deps --pull never \
  -v "$PWD:/workspace:ro" -v "$PWD/$base:/workspace/$base:rw" \
  -e HF_HOME=/tmp/huggingface -e HF_DATASETS_OFFLINE=1 -e XDG_CACHE_HOME=/tmp/cache \
  -e UV_CACHE_DIR=/tmp/uv -e TORCH_HOME=/tmp/torch \
  pai timeout --signal=TERM --kill-after=5s "${limit}s" \
  uv run --no-sync python -u src/step05A_jump_diagnosis.py "$mode" > "$log.log" 2>&1 &
job=$!
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
reason=''
while kill -0 "$job" 2>/dev/null; do
  date -Iseconds >> "$log.monitor.txt"
  if $gpu; then
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits >> "$log.monitor.txt" || { reason='GPU observation failed'; cleanup; break; }
    free_mib=$(tail -1 "$log.monitor.txt" | awk -F, '{print $4+0}')
    if ((free_mib<8192)); then reason='VRAM reserve'; cleanup; break; fi
    docker top "$name" -eo pid 2>/dev/null | awk 'NR>1 {print $1}' >> "$log.own_pids_seen.txt" || true
    sort -un "$log.own_pids_seen.txt" -o "$log.own_pids_seen.txt"
    nvidia-smi -i "$PAI_GPU_DEVICE" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$log.current_apps.csv"
    cat "$log.current_apps.csv" >> "$log.monitor.txt"
    if [[ -s "$log.own_pids_seen.txt" ]]; then
      awk -F, 'NR==FNR {own[$1+0]=1; next} !own[$1+0] {print}' "$log.own_pids_seen.txt" "$log.current_apps.csv" > "$log.external_candidates.csv"
      : > "$log.external_apps.csv"
      while IFS=, read -r candidate memory; do
        if [[ "$candidate" =~ ^[0-9]+$ ]] && kill -0 "$candidate" 2>/dev/null; then printf '%s,%s\n' "$candidate" "$memory" >> "$log.external_apps.csv"; fi
      done < "$log.external_candidates.csv"
      if ! awk -F, 'FILENAME==ARGV[1] {base[$1+0]=$2+0; b+=$2; next} {if(!($1+0 in base))bad=1; c+=$2} END {exit(bad || c>b+1024)}' "$log.baseline_apps.csv" "$log.external_apps.csv"; then reason='External compute change'; cleanup; break; fi
    fi
  fi
  if ! check_disk; then reason='Disk/output budget'; cleanup; break; fi
  docker stats --no-stream --format '{{.CPUPerc}} {{.MemUsage}} {{.PIDs}}' "$name" >> "$log.monitor.txt" 2>/dev/null || true
  sleep 3
done
set +e
wait "$job"; status=$?
set -e
[[ -z "$reason" ]] || status=3
trap - EXIT INT TERM
elapsed=$(($(date +%s)-start))
if $gpu; then printf '%s %s %s\n' "${mode}_${invocation}" "$elapsed" "$status" >> "$base/active_seconds.tsv"; fi
printf '%s\n' "exit=$status elapsed_s=$elapsed monitor_stop=$reason" > "$log.exit.txt"
df -B1 . /var/lib/docker > "$log.disk_after.txt"
tail -3 "$log.log"
exit "$status"

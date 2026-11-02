#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
trial=${1:?trial 0..4};[[ "$trial" =~ ^[0-4]$ ]] || exit 2
base=artifacts/step05A_warn_only/warn_only_20261031_01
log="$base/interval_$trial";name="pai-warn-only-20261031-01-$trial"
[[ ! -e "$log.started.txt" ]] || { echo 'No retry of an attempted interval';exit 2; }
[[ -z "$(docker ps -aq --filter name=pai-warn-only-)" ]] || exit 3
if ((trial>0));then jq -e '.pass_ == true' "$base/trial_$(printf '%02d' $((trial-1)))/review.json" >/dev/null;fi
check_disk(){
 local available docker_available size
 available=$(df -B1 --output=avail . | tail -1)
 docker_available=$(df -B1 --output=avail /var/lib/docker | tail -1)
 size=$(du -sb "$base" | awk '{print $1}')
 ((available>=10737418240 && docker_available>=10737418240 && size<5368709120))
}
check_disk
export PAI_GPU_DEVICE=0
for sample in 1 2 3;do
 TZ=Asia/Tokyo date -Iseconds >> "$log.preflight.txt"
 nvidia-smi -i 0 --query-gpu=index,uuid,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits >> "$log.preflight.txt"
 nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$log.baseline_apps.csv"
 cat "$log.baseline_apps.csv" >> "$log.preflight.txt"
 [[ "$sample" == 3 ]] || sleep 2
done
awk -F, '/^0, GPU-/{if($3>30 || $5<16384)bad=1;if(n && ($4-prev>1024 || prev-$4>1024))bad=1;prev=$4;n++} END{exit(bad || n!=3)}' "$log.preflight.txt"
docker compose -f compose.yaml config --format json | jq -e '.services.pai | (.cpus == 2) and (.mem_limit == "8589934592") and (.shm_size == "1073741824") and (.pids_limit == 256) and (.network_mode == "none") and ((.devices // [])|length == 0) and (.user|startswith("0:")|not)' >/dev/null
if [[ ! -f "$base/new_scripts.sha256" ]];then sha256sum scripts/step05A_warn_only.py scripts/step05A_warn_only_run.sh > "$base/new_scripts.sha256";fi
sha256sum -c "$base/new_scripts.sha256" > "$log.scripts_check.txt"
used=0
if [[ -e "$base/active_seconds.tsv" ]];then used=$(awk '{s+=$2}END{print s+0}' "$base/active_seconds.tsv");fi
remaining=$((3000-used));((remaining>0)) || exit 3
limit=590;((remaining>=limit)) || limit=$remaining
((limit>30)) || exit 3
inner=$((limit-15));start=$(date +%s)
TZ=Asia/Tokyo date -Iseconds > "$log.started.txt"
docker image inspect pai-advanced1-final:step01 --format '{{.Id}}' > "$log.image_id.txt"
docker compose -f compose.yaml run --name "$name" --rm --no-deps --pull never \
 -v "$PWD:/workspace:ro" -v "$PWD/$base:/workspace/$base:rw" \
 -e HF_HOME=/tmp/huggingface -e HF_DATASETS_OFFLINE=1 -e XDG_CACHE_HOME=/tmp/cache \
 -e UV_CACHE_DIR=/tmp/uv -e TORCH_HOME=/tmp/torch -e PYTHONDONTWRITEBYTECODE=1 \
 pai timeout --signal=TERM --kill-after=3s "${inner}s" \
 uv run --no-sync python -u scripts/step05A_warn_only.py "$trial" > "$log.log" 2>&1 &
job=$!
cleanup(){ docker stop -t 3 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
reason=''
while kill -0 "$job" 2>/dev/null;do
 if (($(date +%s)-start>=limit));then reason='wall clock interval/total reserve';cleanup;break;fi
 TZ=Asia/Tokyo date -Iseconds >> "$log.monitor.txt"
 if ! nvidia-smi -i 0 --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits >> "$log.monitor.txt";then reason='GPU measurement failed';cleanup;break;fi
 free_mib=$(tail -1 "$log.monitor.txt" | awk -F, '{print $4+0}')
 if ((free_mib<8192));then reason='VRAM reserve';cleanup;break;fi
 if docker inspect "$name" >/dev/null 2>&1;then
  docker inspect "$name" --format '{{json .HostConfig}}' > "$log.hostconfig.json"
  docker top "$name" -eo pid 2>/dev/null | awk 'NR>1{print $1}' >> "$log.own_pids.txt" || true
  nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits > "$log.current_apps.csv" || { reason='compute measurement failed';cleanup;break; }
  docker top "$name" -eo pid 2>/dev/null | awk 'NR>1{print $1}' >> "$log.own_pids.txt" || true
  sort -un "$log.own_pids.txt" -o "$log.own_pids.txt"
  awk -F, 'FILENAME==ARGV[1]{own[$1+0]=1;next}!own[$1+0]{print}' "$log.own_pids.txt" "$log.current_apps.csv" > "$log.external_candidates.csv"
  : > "$log.external_apps.csv"
  while IFS=, read -r candidate memory;do
   if [[ "$candidate" =~ ^[0-9]+$ ]] && kill -0 "$candidate" 2>/dev/null;then printf '%s,%s\n' "$candidate" "$memory" >> "$log.external_apps.csv";fi
  done < "$log.external_candidates.csv"
  cat "$log.current_apps.csv" >> "$log.monitor.txt"
  if ! awk -F, 'FILENAME==ARGV[1]{base[$1+0]=$2+0;b+=$2;next}{if(!($1+0 in base))bad=1;c+=$2}END{exit(bad || c>b+1024)}' "$log.baseline_apps.csv" "$log.external_apps.csv";then reason='external compute change';cleanup;break;fi
  docker stats --no-stream --format '{{.CPUPerc}} {{.MemUsage}} {{.PIDs}}' "$name" >> "$log.monitor.txt" 2>/dev/null || true
 fi
 if ! check_disk;then reason='disk/output reserve';cleanup;break;fi
 sleep 2
done
set +e
wait "$job";status=$?
set -e
[[ -z "$reason" ]] || status=3
trap - EXIT INT TERM
elapsed=$(($(date +%s)-start))
printf '%s %s %s\n' "$trial" "$elapsed" "$status" >> "$base/active_seconds.tsv"
printf 'exit=%s elapsed_s=%s monitor_stop=%s\n' "$status" "$elapsed" "$reason" > "$log.exit.txt"
TZ=Asia/Tokyo date -Iseconds > "$log.ended.txt"
((elapsed<=600 && elapsed+used<=3000)) || status=3
check_disk || status=3
tail -3 "$log.log"
exit "$status"

#!/bin/bash
# usage: lane.sh <lane_name> <results_file> <config.yaml> [<config.yaml> ...]
# Runs its queue sequentially; lanes are independent processes launched in parallel.
# Inversion is guarded: it waits for both attack-capture files (max 3 h) and refuses
# to run a capture-less attack.
LANE=$1; shift
RESULTS=$1; shift
cd "$(dirname "$0")"
PY=./.venv/bin/python
STATUS=results/lane_${LANE}_status.txt
echo "lane $LANE started $(date)" >> $STATUS

for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  if grep -q "^DONE $name$" "$STATUS" 2>/dev/null; then
    echo "SKIP $name" >> $STATUS
    continue
  fi
  echo "START $name $(date +%H:%M:%S)" >> $STATUS

  if [ "$name" = "inversion" ]; then
    ok=0
    for i in $(seq 1 360); do
      if [ -f results/artifacts/capture_nonprivate.pt ] && [ -f results/artifacts/capture_dp.pt ]; then
        ok=1; break
      fi
      sleep 30
    done
    if [ $ok -eq 0 ]; then
      echo "FAILED inversion (capture files missing)" >> $STATUS
      continue
    fi
  fi

  RESULTS_FILE=$RESULTS $PY run.py --config "$cfg" > "results/log_${name}.txt" 2>&1
  if grep -q "\[run\] done: ok" "results/log_${name}.txt"; then
    echo "DONE $name" >> $STATUS
  else
    echo "FAILED $name" >> $STATUS
  fi
done
echo "lane $LANE finished $(date)" >> $STATUS

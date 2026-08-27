#!/bin/bash
# Full experiment campaign — sequential, logged, resumable-by-inspection.
# Order: MNIST grid -> CIFAR-10 grid -> compression variants -> attack captures -> inversion
cd "$(dirname "$0")"
PY=./.venv/bin/python
STATUS=results/campaign_status.txt
echo "campaign started $(date)" > $STATUS

run_one() {
  local cfg=$1
  local name=$(basename $cfg .yaml)
  if grep -q "^DONE $name$" $STATUS 2>/dev/null; then
    echo "SKIP $name (already done)" >> $STATUS
    return
  fi
  echo "START $name $(date +%H:%M:%S)" >> $STATUS
  $PY run.py --config $cfg > results/log_$name.txt 2>&1
  if grep -q "\[run\] done: ok" results/log_$name.txt; then
    echo "DONE $name $(date +%H:%M:%S)" >> $STATUS
  else
    echo "FAILED $name $(date +%H:%M:%S)" >> $STATUS
  fi
}

# --- MNIST grid (RQ1/RQ2): baseline + remaining eps points (eps2 already complete)
run_one configs/mnist_base.yaml
run_one configs/mnist_eps0.5.yaml
run_one configs/mnist_eps1.yaml
run_one configs/mnist_eps4.yaml
run_one configs/mnist_eps8.yaml

# --- CIFAR-10 grid (RQ1/RQ2)
run_one configs/cifar_base.yaml
run_one configs/cifar_eps0.5.yaml
run_one configs/cifar_eps1.yaml
run_one configs/cifar_eps2.yaml
run_one configs/cifar_eps4.yaml
run_one configs/cifar_eps8.yaml

# --- Compression variants (RQ3)
run_one configs/mnist_comp_base.yaml
run_one configs/mnist_comp_eps2.yaml
run_one configs/mnist_comp_eps8.yaml
run_one configs/cifar_comp_base.yaml
run_one configs/cifar_comp_eps2.yaml
run_one configs/cifar_comp_eps8.yaml

# --- Defence experiment (RQ4): captures from mid-training, then attack
run_one configs/attack_capture_nonprivate.yaml
run_one configs/attack_capture_dp.yaml
run_one configs/inversion.yaml

echo "campaign finished $(date)" >> $STATUS

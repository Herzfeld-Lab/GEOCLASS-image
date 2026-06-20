#!/usr/bin/env bash
set -u
set -o pipefail

CONFIG=${CONFIG:-Config/greenland_enthalpy/jakobshavn-wv/jakobshavn-ice-stream-2018-2021-6class-labeling.config}
OUT_ROOT=${OUT_ROOT:-Output/jakobshavn_resnet_pilots_$(date +%Y%m%d_%H%M%S)}
PYTHON=${PYTHON:-python}
CUDA_FLAG=${CUDA_FLAG:-}
RUN_TEST=${RUN_TEST:-1}

mkdir -p "$OUT_ROOT/logs"

best_checkpoint() {
  local out_dir="$1"
  "$PYTHON" - "$out_dir" <<'PY'
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
loss_files = sorted((out / "losses").glob("*_valid_losses.npy"))
if not loss_files:
    raise SystemExit("")
losses = np.load(loss_files[-1])
best_epoch = int(np.argmin(losses))
checkpoint = out / "checkpoints" / f"epoch_{best_epoch}"
if checkpoint.exists():
    print(checkpoint)
PY
}

run_pilot() {
  local name="$1"
  shift
  local out_dir="$OUT_ROOT/$name"
  local log_file="$OUT_ROOT/logs/$name.log"

  echo "===== $name =====" | tee "$log_file"
  echo "Output: $out_dir" | tee -a "$log_file"

  if ! "$PYTHON" train.py "$CONFIG" \
      --model Resnet18 \
      --output_dir "$out_dir" \
      $CUDA_FLAG \
      "$@" 2>&1 | tee -a "$log_file"; then
    echo "FAILED training: $name" | tee -a "$log_file"
    return 0
  fi

  if [[ "$RUN_TEST" == "1" ]]; then
    local ckpt
    ckpt=$(best_checkpoint "$out_dir")
    if [[ -n "$ckpt" ]]; then
      echo "Testing best checkpoint: $ckpt" | tee -a "$log_file"
      "$PYTHON" test.py "$CONFIG" \
        --model Resnet18 \
        --load_checkpoint "$ckpt" \
        --output_dir "$out_dir" \
        $CUDA_FLAG 2>&1 | tee -a "$log_file" || true
    else
      echo "No checkpoint found for testing: $name" | tee -a "$log_file"
    fi
  fi
}

echo "Writing runs to $OUT_ROOT"
echo "Set CUDA_FLAG=--cuda to use CUDA. Set RUN_TEST=0 to skip prediction generation."
echo "Pretrained runs require torchvision's ResNet18 weights to be cached or downloadable."

run_pilot scratch_lr1e-4_weighted_aug \
  --learning_rate 1e-4 \
  --batch_size 8 \
  --num_epochs 35 \
  --class_weighted \
  --augment

run_pilot scratch_lr3e-4_sampler_aug \
  --learning_rate 3e-4 \
  --batch_size 8 \
  --num_epochs 35 \
  --weighted_sampler \
  --augment

run_pilot scratch_lr5e-5_weighted_sampler_aug \
  --learning_rate 5e-5 \
  --batch_size 8 \
  --num_epochs 45 \
  --class_weighted \
  --weighted_sampler \
  --augment

run_pilot pretrained_lr1e-4_weighted_aug \
  --learning_rate 1e-4 \
  --batch_size 8 \
  --num_epochs 35 \
  --resnet_pretrained \
  --class_weighted \
  --augment

run_pilot pretrained_lr3e-5_weighted_sampler_aug \
  --learning_rate 3e-5 \
  --batch_size 8 \
  --num_epochs 45 \
  --resnet_pretrained \
  --class_weighted \
  --weighted_sampler \
  --augment

run_pilot pretrained_lr1e-5_noaug_weighted \
  --learning_rate 1e-5 \
  --batch_size 8 \
  --num_epochs 35 \
  --resnet_pretrained \
  --class_weighted

echo "All requested pilots finished. Results are under $OUT_ROOT"

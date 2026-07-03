#!/usr/bin/env bash
set -u
set -o pipefail

CONFIG=${CONFIG:-Config/greenland_enthalpy/jakobshavn-wv/jakobshavn-ice-stream-2018-2021-6class-labeling.config}
OUT_ROOT=${OUT_ROOT:-Output/jakobshavn_variomlp_pilots_$(date +%Y%m%d_%H%M%S)}
PYTHON=${PYTHON:-python}
CUDA_FLAG=${CUDA_FLAG:-}
RUN_TEST=${RUN_TEST:-1}
ALLOW_STALE_TRAINING_FOLDER=${ALLOW_STALE_TRAINING_FOLDER:-0}

mkdir -p "$OUT_ROOT/logs"

check_training_folder_sync() {
  "$PYTHON" - "$CONFIG" "$ALLOW_STALE_TRAINING_FOLDER" <<'PY'
import sys
from pathlib import Path
import numpy as np
import yaml

cfg = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
allow_stale = sys.argv[2] == "1"
data = np.load(cfg["npy_path"], allow_pickle=True)
npy_labels = data[1][:, 4].astype(int)
npy_count = int((npy_labels >= 0).sum())
folder = Path(cfg["training_img_path"])
png_count = len(list(folder.glob("*/*.png"))) if folder.exists() else 0
print(f"Active npy labels: {npy_count}")
print(f"Training PNGs: {png_count} ({folder})")
if npy_count != png_count and not allow_stale:
    raise SystemExit(
        "Training PNG folder is stale. Run scripts/sync_jakobshavn_training_pngs_from_npy.py "
        "where the source TIFF volume is mounted, or set ALLOW_STALE_TRAINING_FOLDER=1 to proceed anyway."
    )
PY
}

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

test_supported_args() {
  local args=("$@")
  local filtered=()
  local i=0

  while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
      --hidden_layers|--vario_num_lag)
        filtered+=("${args[$i]}")
        i=$((i + 1))
        if [[ $i -lt ${#args[@]} ]]; then
          filtered+=("${args[$i]}")
        fi
        ;;
      --learning_rate|--batch_size|--num_epochs|--run_name)
        i=$((i + 1))
        ;;
      --class_weighted|--weighted_sampler|--augment|--resnet_pretrained)
        ;;
      *)
        echo "Ignoring train-only or unknown test arg: ${args[$i]}" >&2
        ;;
    esac
    i=$((i + 1))
  done

  printf '%s\n' "${filtered[@]}"
}

run_pilot() {
  local name="$1"
  shift
  local run_args=("$@")
  local out_dir="$OUT_ROOT/$name"
  local log_file="$OUT_ROOT/logs/$name.log"

  echo "===== $name =====" | tee "$log_file"
  echo "Output: $out_dir" | tee -a "$log_file"

  if ! "$PYTHON" train.py "$CONFIG" \
      --model VarioMLP \
      --output_dir "$out_dir" \
      $CUDA_FLAG \
      "${run_args[@]}" 2>&1 | tee -a "$log_file"; then
    echo "FAILED training: $name" | tee -a "$log_file"
    return 0
  fi

  if [[ "$RUN_TEST" == "1" ]]; then
    local ckpt
    ckpt=$(best_checkpoint "$out_dir")
    if [[ -n "$ckpt" ]]; then
      local test_args=()
      while IFS= read -r arg; do
        test_args+=("$arg")
      done < <(test_supported_args "${run_args[@]}")
      echo "Testing best checkpoint: $ckpt" | tee -a "$log_file"
      "$PYTHON" test.py "$CONFIG" \
        --model VarioMLP \
        --load_checkpoint "$ckpt" \
        --output_dir "$out_dir" \
        $CUDA_FLAG \
        "${test_args[@]}" 2>&1 | tee -a "$log_file" || true
    else
      echo "No checkpoint found for testing: $name" | tee -a "$log_file"
    fi
  fi
}

echo "Writing runs to $OUT_ROOT"
echo "Set CUDA_FLAG=--cuda to use CUDA. Set RUN_TEST=0 to skip prediction generation."
if ! check_training_folder_sync; then
  exit 1
fi

run_pilot mlp_lr1e-4_h3-1_weighted \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --num_epochs 60 \
  --hidden_layers "[3, 1]" \
  --class_weighted

run_pilot mlp_lr5e-5_h5-2_weighted \
  --learning_rate 5e-5 \
  --batch_size 16 \
  --num_epochs 80 \
  --hidden_layers "[5, 2]" \
  --class_weighted

run_pilot mlp_lr1e-5_h5-2_weighted \
  --learning_rate 1e-5 \
  --batch_size 16 \
  --num_epochs 80 \
  --hidden_layers "[5, 2]" \
  --class_weighted

run_pilot mlp_lr5e-5_h3-1_sampler \
  --learning_rate 5e-5 \
  --batch_size 16 \
  --num_epochs 80 \
  --hidden_layers "[3, 1]" \
  --weighted_sampler

run_pilot mlp_lr1e-4_h2-1_weighted_sampler \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --num_epochs 60 \
  --hidden_layers "[2, 1]" \
  --class_weighted \
  --weighted_sampler

run_pilot mlp_lr5e-6_h5-2_weighted_sampler \
  --learning_rate 5e-6 \
  --batch_size 16 \
  --num_epochs 100 \
  --hidden_layers "[5, 2]" \
  --class_weighted \
  --weighted_sampler

echo "All requested VarioMLP pilots finished. Results are under $OUT_ROOT"

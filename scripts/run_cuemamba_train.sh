#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# U-Mamba + 双分支挂件 训练脚本
# 基于现有 run_umamba_hcc_preprocess_copy_split_train_v2.sh 修改
# ============================================================

# 复用现有配置
CONDA_ENV="${CONDA_ENV:-/code/U-mamba/conda_envs/umamba_py310_old}"
SOURCE_PARENT="${SOURCE_PARENT:-/code/U-mamba}"
SRC_DIR="${SRC_DIR:-}"

DATASET_ID="${DATASET_ID:-501}"
DATASET_NAME="${DATASET_NAME:-Dataset501_HCC_MRI}"
TRAIN_DATASET_ARG="${TRAIN_DATASET_ARG:-$DATASET_NAME}"
CONFIG="${CONFIG:-3d_fullres}"
FOLD="${FOLD:-0}"  # 固定使用 fold 0

# 使用新的带挂件的 Trainer
TRAINER="${TRAINER:-nnUNetTrainerUMambaBotPlugin}"

REAL_NNUNET_RAW="${REAL_NNUNET_RAW:-/data/zhangqinhu/HCC_MRI/nnunet}"
REAL_NNUNET_PREPROCESSED="${REAL_NNUNET_PREPROCESSED:-/code/U-mamba/prepare/umamba_nnUNet_preprocessed}"
REAL_NNUNET_RESULTS="${REAL_NNUNET_RESULTS:-/output/umamba_nnUNet_results_plugin}"

OLD_SPLITS_FILE="${OLD_SPLITS_FILE:-/code/nnunet/prepare/nnUNet_preprocessed/Dataset501_HCC_MRI/splits_final.json}"

LOG_DIR="${LOG_DIR:-/output/umamba_logs_plugin}"
RUN_PREPROCESS="${RUN_PREPROCESS:-0}"  # 默认不重新预处理
CLEAR_PREPROCESSED_DATASET="${CLEAR_PREPROCESSED_DATASET:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
FIX_NUMPY="${FIX_NUMPY:-0}"
PREDICT_AFTER_TRAIN="${PREDICT_AFTER_TRAIN:-0}"
PRED_DIR="${PRED_DIR:-/output/umamba_predictions_plugin/${TRAINER}_${CONFIG}_fold${FOLD}}"

# 双分支挂件参数
PLUGIN_POWER="${PLUGIN_POWER:-2.0}"

export PYTHONUNBUFFERED=1
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

mkdir -p "$LOG_DIR" "$REAL_NNUNET_PREPROCESSED" "$REAL_NNUNET_RESULTS"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
backup_tag() { date "+%Y%m%d_%H%M%S"; }
log() { echo "[$(timestamp)] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

find_conda_sh() {
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH:-}" ]]; then
    echo "$CONDA_SH"
    return 0
  fi

  local candidates=(
    "/output/miniconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  )

  for f in "${candidates[@]}"; do
    if [[ -f "$f" ]]; then
      echo "$f"
      return 0
    fi
  done

  return 1
}

detect_umamba_source_root() {
  if [[ -n "$SRC_DIR" ]]; then
    [[ -d "$SRC_DIR/umamba" ]] && echo "$SRC_DIR" && return 0
    return 1
  fi

  [[ -d "$SOURCE_PARENT/umamba" ]] && echo "$SOURCE_PARENT" && return 0

  local candidates=(
    "$SOURCE_PARENT/U-Mamba"
    "$SOURCE_PARENT/U-Mamba-main"
    "$SOURCE_PARENT/U-Mamba-master"
  )

  local d
  for d in "${candidates[@]}"; do
    [[ -d "$d/umamba" ]] && echo "$d" && return 0
  done

  while IFS= read -r d; do
    [[ -d "$d/umamba" ]] && echo "$d" && return 0
  done < <(find "$SOURCE_PARENT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

  return 1
}

safe_link_dir_with_backup() {
  local target="$1"
  local link="$2"
  local tag
  tag="$(backup_tag)"

  [[ -d "$target" ]] || die "Target directory does not exist: $target"

  if [[ -L "$link" ]]; then
    local current_target
    current_target="$(readlink "$link")"
    if [[ "$current_target" == "$target" ]]; then
      log "Symlink already correct: $link -> $target"
      return 0
    fi
    log "Removing old symlink: $link -> $current_target"
    rm -f "$link"
  elif [[ -e "$link" ]]; then
    local backup="${link}.backup_${tag}"
    log "Path exists and is not a symlink: $link"
    log "Backing it up to: $backup"
    mv "$link" "$backup"
  fi

  ln -s "$target" "$link"
  log "Created symlink: $link -> $target"
}

log "========== U-Mamba + 双分支挂件 训练开始 =========="
log "CONDA_ENV=$CONDA_ENV"
log "SOURCE_PARENT=$SOURCE_PARENT"
log "DATASET_ID=$DATASET_ID"
log "DATASET_NAME=$DATASET_NAME"
log "TRAIN_DATASET_ARG=$TRAIN_DATASET_ARG"
log "CONFIG=$CONFIG"
log "FOLD=$FOLD"
log "TRAINER=$TRAINER"
log "PLUGIN_POWER=$PLUGIN_POWER"
log "REAL_NNUNET_RAW=$REAL_NNUNET_RAW"
log "REAL_NNUNET_PREPROCESSED=$REAL_NNUNET_PREPROCESSED"
log "REAL_NNUNET_RESULTS=$REAL_NNUNET_RESULTS"
log "OLD_SPLITS_FILE=$OLD_SPLITS_FILE"
log "RUN_PREPROCESS=$RUN_PREPROCESS"
log "CLEAR_PREPROCESSED_DATASET=$CLEAR_PREPROCESSED_DATASET"

hostname || true
nvidia-smi || true

log "========== Activate conda environment =========="
CONDA_SH_PATH="$(find_conda_sh)" || die "Could not find conda.sh. Set CONDA_SH manually."
source "$CONDA_SH_PATH"

[[ -d "$CONDA_ENV" ]] || die "Existing conda environment not found: $CONDA_ENV"
conda activate "$CONDA_ENV"

log "Using python: $(which python)"
log "Using pip: $(which pip)"
python --version

log "========== Detect uploaded U-Mamba source =========="
[[ -d "$SOURCE_PARENT" ]] || die "SOURCE_PARENT not found: $SOURCE_PARENT"
UMAMBA_SRC_ROOT="$(detect_umamba_source_root)" || {
  echo "[ERROR] Could not detect U-Mamba source under $SOURCE_PARENT"
  find "$SOURCE_PARENT" -maxdepth 2 -type d | sort | head -80
  exit 1
}
log "Detected U-Mamba source root: $UMAMBA_SRC_ROOT"

log "========== Copy dual-branch plugin to U-Mamba source =========="
# 复制挂件模块到 U-Mamba 源码目录
PLUGIN_SRC="/code/U-mamba/dual_branch_plugin.py"
PLUGIN_DST="$UMAMBA_SRC_ROOT/umamba/nnunetv2/nets/dual_branch_plugin.py"

if [[ -f "$PLUGIN_SRC" ]]; then
  cp "$PLUGIN_SRC" "$PLUGIN_DST"
  log "Copied dual_branch_plugin.py to U-Mamba source"
else
  log "WARNING: dual_branch_plugin.py not found at $PLUGIN_SRC"
  log "         Make sure to upload the plugin file first"
fi

log "========== Create U-Mamba data symlinks =========="
mkdir -p "$UMAMBA_SRC_ROOT/data"

safe_link_dir_with_backup "$REAL_NNUNET_RAW" "$UMAMBA_SRC_ROOT/data/nnUNet_raw"
safe_link_dir_with_backup "$REAL_NNUNET_PREPROCESSED" "$UMAMBA_SRC_ROOT/data/nnUNet_preprocessed"
safe_link_dir_with_backup "$REAL_NNUNET_RESULTS" "$UMAMBA_SRC_ROOT/data/nnUNet_results"

ls -lah "$UMAMBA_SRC_ROOT/data"

export nnUNet_raw="$REAL_NNUNET_RAW"
export nnUNet_preprocessed="$REAL_NNUNET_PREPROCESSED"
export nnUNet_results="$REAL_NNUNET_RESULTS"

log "Exported nnUNet_raw=$nnUNet_raw"
log "Exported nnUNet_preprocessed=$nnUNet_preprocessed"
log "Exported nnUNet_results=$nnUNet_results"

log "========== Check raw dataset and old split =========="
[[ -d "$REAL_NNUNET_RAW/$DATASET_NAME" ]] || die "Raw dataset not found: $REAL_NNUNET_RAW/$DATASET_NAME"
[[ -f "$REAL_NNUNET_RAW/$DATASET_NAME/dataset.json" ]] || die "dataset.json not found: $REAL_NNUNET_RAW/$DATASET_NAME/dataset.json"
[[ -f "$OLD_SPLITS_FILE" ]] || die "Old splits_final.json not found: $OLD_SPLITS_FILE"

log "Raw dataset:"
ls -lah "$REAL_NNUNET_RAW/$DATASET_NAME"
log "dataset.json:"
cat "$REAL_NNUNET_RAW/$DATASET_NAME/dataset.json"

log "Training image count:"
find "$REAL_NNUNET_RAW/$DATASET_NAME/imagesTr" -maxdepth 1 -name "*.nii.gz" | wc -l
log "Training label count:"
find "$REAL_NNUNET_RAW/$DATASET_NAME/labelsTr" -maxdepth 1 -name "*.nii.gz" | wc -l

log "Old splits file:"
ls -lah "$OLD_SPLITS_FILE"

log "========== Check Python packages =========="
python - <<'PY'
import sys
import numpy
import torch
print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
if numpy.__version__.startswith("2."):
    print("[WARNING] NumPy is 2.x. Use FIX_NUMPY=1 if torch reports _ARRAY_API errors.")
PY

if [[ "$INSTALL_DEPS" == "1" ]]; then
  log "========== Install local U-Mamba source only =========="
  cd "$UMAMBA_SRC_ROOT/umamba"
  python -m pip install -e .
else
  log "INSTALL_DEPS=0, skipping dependency installation."
fi

if [[ "$FIX_NUMPY" == "1" ]]; then
  log "========== Force NumPy 1.26.4 =========="
  python -m pip install "numpy==1.26.4" --force-reinstall
fi

which nnUNetv2_plan_and_preprocess
which nnUNetv2_train

if [[ "$RUN_PREPROCESS" == "1" ]]; then
  log "========== Clear old U-Mamba preprocessed dataset if requested =========="
  if [[ "$CLEAR_PREPROCESSED_DATASET" == "1" ]]; then
    rm -rf "$REAL_NNUNET_PREPROCESSED/$DATASET_NAME"
    log "Removed: $REAL_NNUNET_PREPROCESSED/$DATASET_NAME"
  fi

  log "========== Run U-Mamba-compatible preprocessing =========="
  nnUNetv2_plan_and_preprocess \
    -d "$DATASET_ID" \
    -c "$CONFIG" \
    -np 8 \
    -npfp 8 \
    --verify_dataset_integrity
else
  log "RUN_PREPROCESS=0, skipping preprocessing."
fi

log "========== Copy old splits_final.json into U-Mamba preprocessed folder =========="
NEW_PREPROCESSED_DATASET="$REAL_NNUNET_PREPROCESSED/$DATASET_NAME"
NEW_SPLITS_FILE="$NEW_PREPROCESSED_DATASET/splits_final.json"

[[ -d "$NEW_PREPROCESSED_DATASET" ]] || die "New preprocessed dataset folder not found: $NEW_PREPROCESSED_DATASET"

if [[ -f "$NEW_SPLITS_FILE" ]]; then
  BACKUP_SPLIT="${NEW_SPLITS_FILE}.before_replace_$(backup_tag)"
  cp "$NEW_SPLITS_FILE" "$BACKUP_SPLIT"
  log "Backed up new/default split to: $BACKUP_SPLIT"
fi

cp "$OLD_SPLITS_FILE" "$NEW_SPLITS_FILE"
log "Copied old split:"
log "  from: $OLD_SPLITS_FILE"
log "  to:   $NEW_SPLITS_FILE"

log "========== Validate copied split and generated plans =========="
PLANS_FILE="$NEW_PREPROCESSED_DATASET/nnUNetPlans.json"
[[ -f "$PLANS_FILE" ]] || die "Plans file not found: $PLANS_FILE"
[[ -f "$NEW_SPLITS_FILE" ]] || die "New splits_final.json not found after copy: $NEW_SPLITS_FILE"

python - <<PY
import json
from pathlib import Path

plans_file = Path("$PLANS_FILE")
split_file = Path("$NEW_SPLITS_FILE")
old_split_file = Path("$OLD_SPLITS_FILE")
config = "$CONFIG"

plans = json.loads(plans_file.read_text())
cfg = plans.get("configurations", {}).get(config)
if cfg is None:
    raise RuntimeError(f"Configuration {config!r} not found in {plans_file}")

print("Plan keys for", config, ":", sorted(cfg.keys()))
missing = [k for k in ["conv_kernel_sizes", "pool_op_kernel_sizes"] if k not in cfg]
if missing:
    raise RuntimeError(
        f"Plans missing keys {missing}. "
        f"This means preprocessing probably did not use U-Mamba's fork."
    )

new_splits = json.loads(split_file.read_text())
old_splits = json.loads(old_split_file.read_text())

if new_splits != old_splits:
    raise RuntimeError("Copied splits_final.json does not match the original old split file.")

fold = int("$FOLD")
if fold >= len(new_splits):
    raise RuntimeError(f"Requested fold {fold} but split file has only {len(new_splits)} folds.")

print("Split file copied OK.")
print("Number of folds:", len(new_splits))
print("Fold", fold, "train cases:", len(new_splits[fold]["train"]))
print("Fold", fold, "val cases:", len(new_splits[fold]["val"]))
print("Plans validation OK.")
PY

log "========== Train U-Mamba with Dual-Branch Plugin =========="
TRAIN_LOG="$LOG_DIR/train_${TRAINER}_${DATASET_NAME}_${CONFIG}_fold${FOLD}_plugin.log"

# 设置插件参数环境变量
export PLUGIN_POWER="$PLUGIN_POWER"

nnUNetv2_train \
  "$TRAIN_DATASET_ARG" \
  "$CONFIG" \
  "$FOLD" \
  -tr "$TRAINER" \
  2>&1 | tee "$TRAIN_LOG"

log "Training finished. Log saved to: $TRAIN_LOG"

if [[ "$PREDICT_AFTER_TRAIN" == "1" ]]; then
  log "========== Predict test set =========="
  mkdir -p "$PRED_DIR"

  nnUNetv2_predict \
    -i "$REAL_NNUNET_RAW/$DATASET_NAME/imagesTs" \
    -o "$PRED_DIR" \
    -d "$TRAIN_DATASET_ARG" \
    -c "$CONFIG" \
    -f "$FOLD" \
    -tr "$TRAINER" \
    -chk checkpoint_best.pth \
    --disable_tta \
    2>&1 | tee "$LOG_DIR/predict_${TRAINER}_${DATASET_NAME}_${CONFIG}_fold${FOLD}_plugin.log"

  log "Prediction output: $PRED_DIR"
  log "Ground truth test labels: $REAL_NNUNET_RAW/$DATASET_NAME/test_labels"
fi

log "========== Done =========="

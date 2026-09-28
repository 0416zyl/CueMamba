#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# U-Mamba + 双分支挂件 — 测试集推理 & 评估
# ============================================================

# ---------- 配置 ----------
CONDA_ENV="${CONDA_ENV:-/code/U-mamba/conda_envs/umamba_py310_old}"
SOURCE_PARENT="${SOURCE_PARENT:-/code/U-mamba}"

DATASET_NAME="${DATASET_NAME:-Dataset501_HCC_MRI}"
CONFIG="${CONFIG:-3d_fullres}"
FOLD="${FOLD:-0}"
TRAINER="${TRAINER:-nnUNetTrainerUMambaBotPlugin}"

REAL_NNUNET_RAW="${REAL_NNUNET_RAW:-/data/zhangqinhu/HCC_MRI/nnunet}"
REAL_NNUNET_PREPROCESSED="${REAL_NNUNET_PREPROCESSED:-/code/U-mamba/prepare/umamba_nnUNet_preprocessed}"

# 你已经把文件放在这个目录下
PLUGIN_DIR="/code/U-mamba/nnUNet_results_plugin"

# 所有输出放在 /code/ 下
LOCAL_RESULTS="/code/U-mamba/nnUNet_results_plugin"
PRED_DIR="${PRED_DIR:-/code/U-mamba/predictions_plugin_test}"
EVAL_LOG="${EVAL_LOG:-/code/U-mamba/logs_plugin/eval_plugin_test.log}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ---------- 激活 conda ----------
find_conda_sh() {
  local candidates=(
    "/output/miniconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  )
  for f in "${candidates[@]}"; do
    [[ -f "$f" ]] && echo "$f" && return 0
  done
  return 1
}

log "========== 激活 conda 环境 =========="
CONDA_SH_PATH="$(find_conda_sh)" || die "找不到 conda.sh"
source "$CONDA_SH_PATH"
conda activate "$CONDA_ENV"
log "Python: $(which python)"
python --version

# ---------- 检测 U-Mamba 源码 ----------
detect_umamba_source_root() {
  [[ -d "$SOURCE_PARENT/umamba" ]] && echo "$SOURCE_PARENT" && return 0
  local d
  for d in "$SOURCE_PARENT/U-Mamba" "$SOURCE_PARENT/U-Mamba-main" "$SOURCE_PARENT/U-Mamba-master"; do
    [[ -d "$d/umamba" ]] && echo "$d" && return 0
  done
  return 1
}

UMAMBA_SRC_ROOT="$(detect_umamba_source_root)" || die "找不到 U-Mamba 源码"
log "U-Mamba 源码: $UMAMBA_SRC_ROOT"

# ---------- 检查你放的文件 ----------
log "========== 检查 plugin 目录 =========="
[[ -f "$PLUGIN_DIR/checkpoint_best.pth" ]] || die "checkpoint 不存在: $PLUGIN_DIR/checkpoint_best.pth"
[[ -f "$PLUGIN_DIR/dataset.json" ]] || die "dataset.json 不存在: $PLUGIN_DIR/dataset.json"
[[ -f "$PLUGIN_DIR/plans.json" ]] || die "plans.json 不存在: $PLUGIN_DIR/plans.json"
log "三个文件都在: $PLUGIN_DIR/"

# ---------- 构建 nnUNetv2_predict 需要的目录结构 ----------
# nnUNetv2_predict 期望: {nnUNet_results}/{dataset}/{trainer}__nnUNetPlans__{config}/...
log "========== 构建模型目录结构 =========="
MODEL_DIR="$LOCAL_RESULTS/$DATASET_NAME/${TRAINER}__nnUNetPlans__${CONFIG}"
mkdir -p "$MODEL_DIR/fold_${FOLD}"

# 链接/复制文件到正确位置
ln -sf "$PLUGIN_DIR/dataset.json" "$MODEL_DIR/dataset.json"
ln -sf "$PLUGIN_DIR/plans.json" "$MODEL_DIR/plans.json"
ln -sf "$PLUGIN_DIR/checkpoint_best.pth" "$MODEL_DIR/fold_${FOLD}/checkpoint_best.pth"

# 复制 splits_final.json（如果存在）
SPLITS="$REAL_NNUNET_PREPROCESSED/$DATASET_NAME/splits_final.json"
if [[ -f "$SPLITS" ]]; then
  cp "$SPLITS" "$MODEL_DIR/splits_final.json"
fi

log "模型目录结构:"
find "$MODEL_DIR" -type f -o -type l | head -10

# ---------- 设置环境变量 ----------
export nnUNet_raw="$REAL_NNUNET_RAW"
export nnUNet_preprocessed="$REAL_NNUNET_PREPROCESSED"
export nnUNet_results="$LOCAL_RESULTS"

log "nnUNet_results=$nnUNet_results"

# ---------- 检查测试集 ----------
log "========== 检查测试集 =========="
TEST_IMAGES="$REAL_NNUNET_RAW/$DATASET_NAME/imagesTs"
TEST_LABELS="$REAL_NNUNET_RAW/$DATASET_NAME/test_labels"

[[ -d "$TEST_IMAGES" ]] || die "测试集目录不存在: $TEST_IMAGES"
TEST_COUNT=$(find "$TEST_IMAGES" -maxdepth 1 -name "*.nii.gz" | wc -l)
log "测试集图像数: $TEST_COUNT"

# ---------- 推理 ----------
log "========== 测试集推理开始 =========="
mkdir -p "$PRED_DIR"
mkdir -p "$(dirname "$EVAL_LOG")"

DATASET_JSON="$PLUGIN_DIR/dataset.json"
PLANS_JSON="$PLUGIN_DIR/plans.json"

nnUNetv2_predict \
    -i "$TEST_IMAGES" \
    -o "$PRED_DIR" \
    -d "$DATASET_NAME" \
    -c "$CONFIG" \
    -f "$FOLD" \
    -tr "$TRAINER" \
    -chk "$PLUGIN_DIR/checkpoint_best.pth" \
    --disable_tta \
    2>&1 | tee "${EVAL_LOG%.log}_predict.log"

PRED_COUNT=$(find "$PRED_DIR" -maxdepth 1 -name "*.nii.gz" | wc -l)
log "推理完成，预测文件数: $PRED_COUNT"

if [[ "$PRED_COUNT" -ne "$TEST_COUNT" ]]; then
  die "预测文件数($PRED_COUNT) != 测试集数($TEST_COUNT)，推理可能不完整"
fi

# ---------- 评估 ----------
log "========== 评估开始 =========="

nnUNetv2_evaluate_folder \
    "$TEST_LABELS" \
    "$PRED_DIR" \
    -djfile "$DATASET_JSON" \
    -pfile "$PLANS_JSON" \
    -o "$PRED_DIR/summary.json" \
    2>&1 | tee "${EVAL_LOG%.log}_eval.log"

log "========== 完成 =========="
log "预测结果: $PRED_DIR"
log "评估结果: $PRED_DIR/summary.json"
log ""
log "下载 summary.json 回本地对比"

#!/bin/bash
# ============================================
# U-Mamba + 双分支挂件 肝脏动脉期分割训练脚本
# 适配超算平台 (PyTorch 1.8 镜像)
# ============================================

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:/code/U-Mamba/umamba"
cd /code/U-Mamba

echo "=========================================="
echo "步骤1: 安装依赖"
echo "=========================================="

# 安装 U-Mamba 依赖
pip install -e .

# 安装额外依赖
pip install causal-conv1d>=1.2.0
pip install mamba-ssm --no-cache-dir
pip install nibabel
pip install scipy

echo "=========================================="
echo "步骤2: 检查数据集"
echo "=========================================="

DATA_DIR="/data/zhangqinhu/HCC_MRI"
echo "数据集目录: $DATA_DIR"

# 检查数据集结构
if [ -d "$DATA_DIR/imagesTr" ] && [ -d "$DATA_DIR/labelsTr" ]; then
    echo "✓ 数据集结构正确"
    echo "  imagesTr 文件数: $(ls $DATA_DIR/imagesTr/*.nii.gz 2>/dev/null | wc -l)"
    echo "  labelsTr 文件数: $(ls $DATA_DIR/labelsTr/*.nii.gz 2>/dev/null | wc -l)"
else
    echo "✗ 数据集结构不正确，请检查目录"
    exit 1
fi

# 检查 dataset.json
if [ -f "$DATA_DIR/dataset.json" ]; then
    echo "✓ dataset.json 存在"
else
    echo "✗ dataset.json 不存在，请创建"
    exit 1
fi

echo "=========================================="
echo "步骤3: 预处理数据（如果需要）"
echo "=========================================="

# nnU-Net 格式预处理
# 如果数据已经是 nnU-Net 格式，可以跳过这一步
# nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

echo "=========================================="
echo "步骤4: 开始训练"
echo "=========================================="

# 训练参数配置
DATASET_ID=1  # 数据集ID，根据你的 dataset.json 设置
CONFIG=3d_fullres  # 配置：2d, 3d_fullres, 3d_cascade_fullres
TRAINER=nnUNetTrainerUMambaEnc  # 训练器：UMambaEnc 或 UMambaBot
PLUGIN_POWER=2.0  # 双分支挂件的幂次参数

# 创建输出目录
mkdir -p /code/U-Mamba/output

# 运行训练
python -c "
import torch
import sys
import os

# 添加路径
sys.path.insert(0, '/code/U-Mamba/umamba/nnunetv2/nets')

print('PyTorch 版本:', torch.__version__)
print('CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU 设备:', torch.cuda.get_device_name(0))

# 导入模型
from UMambaEnc_plugin import UMambaEncWithPlugin

print('模型导入成功，开始训练...')
"

# 使用 nnU-Net 框架训练
# 注意：需要先配置好 nnU-Net 的环境变量
export nnUNet_raw="/data/zhangqinhu/HCC_MRI/raw"
export nnUNet_preprocessed="/data/zhangqinhu/HCC_MRI/preprocessed"
export nnUNet_results="/code/U-Mamba/output"

# 训练命令
nnUNetv2_train $DATASET_ID $CONFIG 0 -tr $TRAINER \
    --custom_args '{"plugin_power": $PLUGIN_POWER}'

echo "=========================================="
echo "训练完成！"
echo "=========================================="

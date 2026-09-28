#!/bin/bash
# ============================================
# 超算平台启动脚本
# U-Mamba + 双分支挂件 肝脏动脉期分割
# ============================================

# 进入代码目录
cd /code/U-Mamba

echo "=========================================="
echo "安装依赖..."
echo "=========================================="

# 安装依赖（PyTorch 1.8 环境）
pip install nibabel scipy tqdm

# 安装 Mamba SSM（如果需要完整 U-Mamba）
# pip install causal-conv1d>=1.2.0
# pip install mamba-ssm --no-cache-dir

echo "=========================================="
echo "检查数据集..."
echo "=========================================="

DATA_DIR="/data/zhangqinhu/HCC_MRI"
echo "数据集: $DATA_DIR"
ls -la $DATA_DIR/

echo "=========================================="
echo "开始训练..."
echo "=========================================="

# 运行独立训练脚本
python train_standalone.py \
    --data_dir /data/zhangqinhu/HCC_MRI \
    --output_dir /code/U-Mamba/output \
    --epochs 150 \
    --batch_size 2 \
    --lr 1e-4 \
    --plugin_power 2.0 \
    --num_classes 2

echo "=========================================="
echo "训练完成！"
echo "=========================================="

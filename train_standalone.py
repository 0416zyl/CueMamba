"""
独立训练脚本 - U-Mamba + 双分支挂件
适配超算平台 PyTorch 1.8

使用方法：
1. 上传到超算平台 /code/U-Mamba/
2. 在sh脚本中调用：python /code/U-Mamba/train_standalone.py
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'umamba', 'nnunetv2', 'nets'))

# ============================================
# 数据集类
# ============================================
class LiverArterialDataset(Dataset):
    """肝脏动脉期CT数据集"""

    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: 数据集目录，包含 imagesTr 和 labelsTr
            split: 'train' 或 'val'
            transform: 数据增强
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # 读取数据列表
        self.images_dir = self.data_dir / 'imagesTr'
        self.labels_dir = self.data_dir / 'labelsTr'

        # 获取所有图像文件
        self.image_files = sorted(list(self.images_dir.glob('*.nii.gz')))

        # 按 8:2 划分训练集和验证集
        split_idx = int(len(self.image_files) * 0.8)
        if split == 'train':
            self.image_files = self.image_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]

        print(f"[{split}] 数据集大小: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图像
        image_path = self.image_files[idx]
        label_path = self.labels_dir / image_path.name

        # 加载 NIfTI 文件
        image_nii = nib.load(image_path)
        label_nii = nib.load(label_path)

        image = image_nii.get_fdata().astype(np.float32)
        label = label_nii.get_fdata().astype(np.int64)

        # 归一化
        image = (image - image.mean()) / (image.std() + 1e-8)

        # 添加通道维度
        image = np.expand_dims(image, axis=0)  # [1, D, H, W]

        # 数据增强
        if self.transform:
            image, label = self.transform(image, label)

        return torch.from_numpy(image), torch.from_numpy(label)


# ============================================
# 双分支挂件模块（简化版，兼容 PyTorch 1.8）
# ============================================
class DualBranchPlugin3D(nn.Module):
    """3D双分支挂件模块"""

    def __init__(self, in_channels, stage_level=1, power=2.0):
        super().__init__()
        self.in_channels = in_channels
        self.power = power

        # 核大小
        kernel_sizes = {1: 15, 2: 11, 3: 7, 4: 5}
        self.kernel_size = kernel_sizes.get(stage_level, 7)

        reduced_channels = max(in_channels // 4, 8)

        # Branch 1: 局部差异分支
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, reduced_channels, 1),
            nn.InstanceNorm3d(reduced_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(reduced_channels, in_channels, 1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Branch 2: 置信度分支
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, reduced_channels, 3, padding=1),
            nn.InstanceNorm3d(reduced_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(reduced_channels, 1, 1),
            nn.Sigmoid()
        )

        # 空间聚合模块
        self.spatial_agg = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # 相似性计算模块
        self.similarity = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1),
            nn.InstanceNorm3d(in_channels),
            nn.Sigmoid()
        )

        # 门控融合模块
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels * 3, in_channels, 1),
            nn.InstanceNorm3d(in_channels)
        )

        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        # Branch 1: 局部差异
        F_L = nn.functional.avg_pool3d(
            x, kernel_size=self.kernel_size,
            padding=self.kernel_size // 2, stride=1
        )
        E_diff = x - F_L
        E = self.branch1(E_diff)

        # Branch 2: 置信度
        P = self.branch2(x)
        S = torch.pow(P, self.power)
        S_expand = S * x
        lesion_proto = self.spatial_agg(S_expand)
        M = self.similarity(torch.cat([lesion_proto, x], dim=1))

        # 门控融合
        D = torch.abs(E - M)
        F_c = self.gate(torch.cat([x, E, M], dim=1))
        output = x + self.alpha * D * F_c

        return output


# ============================================
# 简化的 3D U-Net 模型（带双分支挂件）
# ============================================
class SimpleUNet3DWithPlugin(nn.Module):
    """简化的3D U-Net，带双分支挂件"""

    def __init__(self, in_channels=1, num_classes=2, plugin_power=2.0):
        super().__init__()

        # 编码器
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)

        # 下采样
        self.pool = nn.MaxPool3d(2)

        # 瓶颈层
        self.bottleneck = self._conv_block(256, 512)

        # 上采样
        self.up4 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)

        # 解码器
        self.dec4 = self._conv_block(512, 256)
        self.dec3 = self._conv_block(256, 128)
        self.dec2 = self._conv_block(128, 64)
        self.dec1 = self._conv_block(64, 32)

        # 双分支挂件
        self.plugins = nn.ModuleList([
            DualBranchPlugin3D(32, stage_level=1, power=plugin_power),
            DualBranchPlugin3D(64, stage_level=2, power=plugin_power),
            DualBranchPlugin3D(128, stage_level=3, power=plugin_power),
            DualBranchPlugin3D(256, stage_level=4, power=plugin_power),
        ])

        # 输出层
        self.final = nn.Conv3d(32, num_classes, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e1 = self.plugins[0](e1)  # 挂件1
        e2 = self.enc2(self.pool(e1))
        e2 = self.plugins[1](e2)  # 挂件2
        e3 = self.enc3(self.pool(e2))
        e3 = self.plugins[2](e3)  # 挂件3
        e4 = self.enc4(self.pool(e3))
        e4 = self.plugins[3](e4)  # 挂件4

        # 瓶颈层
        b = self.bottleneck(self.pool(e4))

        # 解码器
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================
# 损失函数
# ============================================
class DiceLoss(nn.Module):
    """Dice 损失"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = nn.functional.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ============================================
# 训练函数
# ============================================
def train():
    parser = argparse.ArgumentParser(description='U-Mamba + 双分支挂件训练')
    parser.add_argument('--data_dir', type=str, default='/data/zhangqinhu/HCC_MRI',
                        help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='/code/U-Mamba/output',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--plugin_power', type=float, default=2.0,
                        help='双分支挂件幂次参数')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据集
    train_dataset = LiverArterialDataset(args.data_dir, split='train')
    val_dataset = LiverArterialDataset(args.data_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    # 模型
    model = SimpleUNet3DWithPlugin(
        in_channels=1,
        num_classes=args.num_classes,
        plugin_power=args.plugin_power
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 损失函数
    dice_loss = DiceLoss()
    ce_loss = nn.CrossEntropyLoss()

    # 训练循环
    best_val_dice = 0.0
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = 0.5 * dice_loss(outputs, labels) + 0.5 * ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            dice = 1.0 - dice_loss(outputs, labels).item()
            train_dice += dice

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = 0.5 * dice_loss(outputs, labels) + 0.5 * ce_loss(outputs, labels)

                val_loss += loss.item()
                pred = torch.argmax(outputs, dim=1)
                dice = 1.0 - dice_loss(outputs, labels).item()
                val_dice += dice

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step()

        # 打印日志
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  -> 保存最佳模型 (Val Dice: {val_dice:.4f})")

        # 定期保存 checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth'))

    print(f"\n训练完成！最佳 Val Dice: {best_val_dice:.4f}")


if __name__ == '__main__':
    train()

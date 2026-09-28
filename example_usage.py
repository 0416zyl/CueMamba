"""
双分支挂件使用示例

展示如何在U-Mamba中使用双分支挂件进行肝脏动脉期分割
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'umamba', 'nnunetv2', 'nets'))

from dual_branch_plugin import DualBranchPlugin3D


def create_simple_umamba_with_plugin():
    """
    创建一个简化的U-Mamba模型（带双分支挂件）

    注意：这是一个简化的演示版本，实际使用时需要完整的U-Mamba架构
    """

    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage1 = nn.Sequential(
                nn.Conv3d(1, 64, 3, padding=1),
                nn.InstanceNorm3d(64),
                nn.LeakyReLU(0.01)
            )
            self.stage2 = nn.Sequential(
                nn.Conv3d(64, 128, 3, stride=2, padding=1),
                nn.InstanceNorm3d(128),
                nn.LeakyReLU(0.01)
            )
            self.stage3 = nn.Sequential(
                nn.Conv3d(128, 256, 3, stride=2, padding=1),
                nn.InstanceNorm3d(256),
                nn.LeakyReLU(0.01)
            )
            self.stage4 = nn.Sequential(
                nn.Conv3d(256, 512, 3, stride=2, padding=1),
                nn.InstanceNorm3d(512),
                nn.LeakyReLU(0.01)
            )

        def forward(self, x):
            features = []
            x = self.stage1(x)
            features.append(x)
            x = self.stage2(x)
            features.append(x)
            x = self.stage3(x)
            features.append(x)
            x = self.stage4(x)
            features.append(x)
            return features

    class SimpleDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.up3 = nn.ConvTranspose3d(512, 256, 2, stride=2)
            self.conv3 = nn.Sequential(
                nn.Conv3d(512, 256, 3, padding=1),
                nn.InstanceNorm3d(256),
                nn.LeakyReLU(0.01)
            )
            self.up2 = nn.ConvTranspose3d(256, 128, 2, stride=2)
            self.conv2 = nn.Sequential(
                nn.Conv3d(256, 128, 3, padding=1),
                nn.InstanceNorm3d(128),
                nn.LeakyReLU(0.01)
            )
            self.up1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
            self.conv1 = nn.Sequential(
                nn.Conv3d(128, 64, 3, padding=1),
                nn.InstanceNorm3d(64),
                nn.LeakyReLU(0.01)
            )
            self.seg = nn.Conv3d(64, 2, 1)

        def forward(self, features):
            f1, f2, f3, f4 = features

            # Stage 3
            x = self.up3(f4)
            x = torch.cat([x, f3], dim=1)
            x = self.conv3(x)

            # Stage 2
            x = self.up2(x)
            x = torch.cat([x, f2], dim=1)
            x = self.conv2(x)

            # Stage 1
            x = self.up1(x)
            x = torch.cat([x, f1], dim=1)
            x = self.conv1(x)

            # 分割输出
            x = self.seg(x)
            return x

    class SimpleUMambaWithPlugin(nn.Module):
        def __init__(self, plugin_power=2.0):
            super().__init__()
            self.encoder = SimpleEncoder()
            self.decoder = SimpleDecoder()

            # 为编码器的每个阶段添加双分支挂件
            encoder_channels = [64, 128, 256, 512]
            self.encoder_plugins = nn.ModuleList([
                DualBranchPlugin3D(ch, stage_level=i+1, power=plugin_power)
                for i, ch in enumerate(encoder_channels)
            ])

            # 为解码器的每个阶段添加双分支挂件
            decoder_channels = [256, 128, 64]
            self.decoder_plugins = nn.ModuleList([
                DualBranchPlugin3D(ch, stage_level=i+1, power=plugin_power)
                for i, ch in enumerate(decoder_channels)
            ])

        def forward(self, x):
            # 编码器
            features = []
            enc_stages = [self.encoder.stage1, self.encoder.stage2,
                         self.encoder.stage3, self.encoder.stage4]

            for i, stage in enumerate(enc_stages):
                x = stage(x)
                # 应用双分支挂件
                x = self.encoder_plugins[i](x)
                features.append(x)

            # 解码器
            # 注意：这里简化了跳跃连接的处理
            x = self.decoder(features)

            return x

    return SimpleUMambaWithPlugin()


def example_liver_segmentation():
    """
    肝脏动脉期分割示例
    """
    print("=" * 60)
    print("肝脏动脉期分割示例")
    print("=" * 60)

    # 创建模型
    model = create_simple_umamba_with_plugin()
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建模拟输入 (batch_size=2, channels=1, D=64, H=128, W=128)
    # 实际使用时应该是真实的CT/MRI数据
    x = torch.randn(2, 1, 64, 128, 128)
    print(f"输入形状: {x.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(x)

    print(f"输出形状: {output.shape}")
    print(f"输出类别数: {output.shape[1]}")

    # 如果有GPU，测试GPU推理
    if torch.cuda.is_available():
        print("\n测试GPU推理:")
        model = model.cuda()
        x = x.cuda()
        with torch.no_grad():
            output = model(x)
        print(f"输出设备: {output.device}")
        print(f"输出形状: {output.shape}")

    print("\n✓ 示例完成!")


def example_plugin_visualization():
    """
    可视化挂件输出（用于调试和理解）
    """
    print("\n" + "=" * 60)
    print("挂件输出可视化示例")
    print("=" * 60)

    # 创建挂件
    plugin = DualBranchPlugin3D(in_channels=64, stage_level=1, power=2.0)

    # 创建输入
    x = torch.randn(1, 64, 32, 32, 32)

    # 获取中间输出
    B, C, H, W, D = x.shape

    # Branch 1: 局部差异
    F_L = torch.nn.functional.avg_pool3d(
        x,
        kernel_size=plugin.kernel_size,
        padding=plugin.kernel_size // 2,
        stride=1
    )
    E_diff = x - F_L
    E = plugin.branch1(E_diff)

    # Branch 2: 置信度
    P = plugin.branch2(x)
    S = torch.pow(P, plugin.power)

    print(f"\nBranch 1 输出 (差异图 E):")
    print(f"  形状: {E.shape}")
    print(f"  均值: {E.mean():.4f}")
    print(f"  标准差: {E.std():.4f}")

    print(f"\nBranch 2 输出 (置信度 P):")
    print(f"  形状: {P.shape}")
    print(f"  均值: {P.mean():.4f}")
    print(f"  标准差: {P.std():.4f}")

    print(f"\n置信度加权 (S = P^r):")
    print(f"  形状: {S.shape}")
    print(f"  均值: {S.mean():.4f}")
    print(f"  标准差: {S.std():.4f}")

    # 完整前向传播
    output = plugin(x)
    D = torch.abs(E - torch.randn_like(E))  # 模拟M
    print(f"\n最终输出:")
    print(f"  形状: {output.shape}")
    print(f"  均值: {output.mean():.4f}")
    print(f"  标准差: {output.std():.4f}")


if __name__ == "__main__":
    example_liver_segmentation()
    example_plugin_visualization()

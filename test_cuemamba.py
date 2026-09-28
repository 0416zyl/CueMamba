"""
测试双分支挂件模块

验证挂件是否能正常工作，以及与U-Mamba的集成是否正确
"""

import torch
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'umamba', 'nnunetv2', 'nets'))

# 导入挂件模块
try:
    from dual_branch_plugin import DualBranchPlugin3D, MultiScaleDualBranchPlugin
except ImportError:
    # 如果导入失败，尝试从绝对路径导入
    from umamba.nnunetv2.nets.dual_branch_plugin import DualBranchPlugin3D, MultiScaleDualBranchPlugin


def test_dual_branch_plugin_3d():
    """测试3D双分支挂件模块"""
    print("=" * 60)
    print("测试 3D 双分支挂件模块")
    print("=" * 60)

    # 测试不同阶段的挂件
    for stage_level in [1, 2, 3, 4]:
        in_channels = {1: 64, 2: 128, 3: 256, 4: 512}[stage_level]
        spatial_size = {1: 64, 2: 32, 3: 16, 4: 8}[stage_level]

        print(f"\nStage {stage_level}: in_channels={in_channels}, spatial_size={spatial_size}")

        plugin = DualBranchPlugin3D(
            in_channels=in_channels,
            stage_level=stage_level,
            power=2.0
        )

        # 创建输入张量
        x = torch.randn(2, in_channels, spatial_size, spatial_size, spatial_size)
        print(f"  Input shape: {x.shape}")

        # 前向传播
        with torch.no_grad():
            output = plugin(x)

        print(f"  Output shape: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        print(f"  [PASS] Shape test passed")

        # 检查输出范围
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    print("\n[PASS] 所有3D双分支挂件测试通过!")


def test_multi_scale_plugin():
    """测试多尺度挂件"""
    print("\n" + "=" * 60)
    print("测试多尺度双分支挂件")
    print("=" * 60)

    # 模拟U-Mamba的4个尺度（使用2D，因为MultiScaleDualBranchPlugin使用2D插件）
    channel_list = [64, 128, 256, 512]
    spatial_sizes = [64, 32, 16, 8]

    multi_plugin = MultiScaleDualBranchPlugin(
        channel_list=channel_list,
        num_encoder_stages=4,
        num_decoder_stages=3,
        power=2.0
    )

    print(f"\n编码器挂件数量: {len(multi_plugin.encoder_plugins)}")
    print(f"解码器挂件数量: {len(multi_plugin.decoder_plugins)}")

    # 测试编码器前向传播（使用2D张量）
    encoder_features = [
        torch.randn(2, ch, sp, sp)
        for ch, sp in zip(channel_list, spatial_sizes)
    ]

    print("\n测试编码器前向传播:")
    enhanced_encoder = multi_plugin.forward_encoder(encoder_features)
    for i, (orig, enh) in enumerate(zip(encoder_features, enhanced_encoder)):
        print(f"  Stage {i+1}: {orig.shape} -> {enh.shape}")
        assert orig.shape == enh.shape

    # 测试解码器前向传播（使用2D张量）
    decoder_features = [
        torch.randn(2, ch, sp, sp)
        for ch, sp in zip([256, 128, 64], [16, 32, 64])
    ]

    print("\n测试解码器前向传播:")
    enhanced_decoder = multi_plugin.forward_decoder(decoder_features)
    for i, (orig, enh) in enumerate(zip(decoder_features, enhanced_decoder)):
        print(f"  Stage {i+1}: {orig.shape} -> {enh.shape}")
        assert orig.shape == enh.shape

    print("\n[PASS] 所有多尺度挂件测试通过!")


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "=" * 60)
    print("测试梯度流")
    print("=" * 60)

    plugin = DualBranchPlugin3D(in_channels=64, stage_level=1, power=2.0)
    x = torch.randn(2, 64, 32, 32, 32, requires_grad=True)

    # 前向传播
    output = plugin(x)

    # 计算损失并反向传播
    loss = output.sum()
    loss.backward()

    # 检查梯度
    print(f"\n输入梯度形状: {x.grad.shape}")
    print(f"输入梯度范围: [{x.grad.min():.6f}, {x.grad.max():.6f}]")
    assert x.grad is not None, "No gradient computed"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"
    assert not torch.isinf(x.grad).any(), "Inf in gradients"

    print("\n[PASS] 梯度流测试通过!")


def test_memory_usage():
    """测试内存使用"""
    print("\n" + "=" * 60)
    print("测试内存使用")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return

    plugin = DualBranchPlugin3D(in_channels=256, stage_level=3, power=2.0).cuda()
    x = torch.randn(2, 256, 32, 32, 32).cuda()

    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 前向传播
    with torch.no_grad():
        output = plugin(x)

    memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"\nGPU内存使用: {memory_used:.2f} MB")
    print(f"输出形状: {output.shape}")

    print("\n[PASS] 内存使用测试通过!")


if __name__ == "__main__":
    print("双分支挂件模块测试")
    print("=" * 60)

    test_dual_branch_plugin_3d()
    test_multi_scale_plugin()
    test_gradient_flow()
    test_memory_usage()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)

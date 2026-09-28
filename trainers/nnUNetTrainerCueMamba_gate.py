"""
带双分支挂件的 nnUNetTrainerUMambaBot - 只改门控融合版本

基于现有 nnUNetTrainerUMambaBot 扩展
添加双分支挂件用于增强肝脏动脉期分割

使用方法:
    nnUNetv2_train Dataset501_HCC_MRI 3d_fullres 0 -tr nnUNetTrainerUMambaBotPluginGate
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union, List

# 导入原始 Trainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot

# 导入双分支挂件 - 只改门控融合版本
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'nets'))
from dual_branch_plugin_simple_gate import DualBranchPlugin3D_SimplifiedGate as DualBranchPlugin3D


# ============================================================
# 独立函数（因为 build_network_architecture 是 @staticmethod）
# ============================================================

def _get_stage_level(module_name):
    """根据模块名称推断阶段级别"""
    if 'encoder' in module_name.lower():
        for i in range(1, 5):
            if f'stage{i}' in module_name.lower() or f'blocks.{i-1}' in module_name.lower():
                return i
    elif 'decoder' in module_name.lower():
        for i in range(1, 5):
            if f'stage{i}' in module_name.lower():
                return i
    return 1


def _add_plugin_to_network(network):
    """向网络中添加双分支挂件"""
    plugin_power = float(os.environ.get('PLUGIN_POWER', '2.0'))
    print(f"Adding dual-branch plugins (simplified gate) to network... (power={plugin_power})")

    for name, module in network.named_modules():
        if 'MambaLayer' in type(module).__name__:
            # 获取维度
            if hasattr(module, 'dim'):
                dim = module.dim
            else:
                for param in module.parameters():
                    dim = param.shape[0]
                    break

            stage_level = _get_stage_level(name)

            plugin = DualBranchPlugin3D(
                in_channels=dim,
                stage_level=stage_level,
                power=plugin_power
            )

            module.dual_branch_plugin = plugin
            print(f"  Added plugin to {name}: dim={dim}, stage={stage_level}")

    return network


# ============================================================
# Plugin Trainer - Simplified Gate Version
# ============================================================

class nnUNetTrainerUMambaBotPluginGate(nnUNetTrainerUMambaBot):
    """
    带双分支挂件的 U-Mamba Bot 训练器 - 只改门控融合版本
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.plugin_power = float(os.environ.get('PLUGIN_POWER', '2.0'))
        self.print_to_log_file(f"Dual-Branch Plugin (Simplified Gate) enabled: power={self.plugin_power}")
        self.print_to_log_file(f"Output folder: {self.output_folder}")

    @staticmethod
    def build_network_architecture(plans_manager, dataset_json,
                                   configuration_manager, num_input_channels,
                                   enable_deep_supervision=False):
        """
        构建带双分支挂件的网络架构
        必须是 @staticmethod，与父类签名一致，推理时才能正确调用
        """
        # 调用父类构建原始网络
        network = nnUNetTrainerUMambaBot.build_network_architecture(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, enable_deep_supervision
        )

        # 在网络中添加双分支挂件
        network = _add_plugin_to_network(network)

        return network

    def train_step(self, batch):
        """训练步骤"""
        return super().train_step(batch)

    def on_train_epoch_end(self, train_logs):
        """训练 epoch 结束时的回调"""
        super().on_train_epoch_end(train_logs)

        if self.current_epoch % 10 == 0:
            self._save_plugin_state()

    def _save_plugin_state(self):
        """保存挂件状态"""
        plugin_state = {}
        for name, module in self.network.named_modules():
            if hasattr(module, 'dual_branch_plugin'):
                plugin_state[name] = module.dual_branch_plugin.state_dict()

        if plugin_state:
            save_path = os.path.join(self.output_folder, f'plugin_state_epoch{self.current_epoch}.pth')
            torch.save(plugin_state, save_path)
            self.print_to_log_file(f"Saved plugin state to {save_path}")

    def load_checkpoint(self, filename_or_checkpoint):
        """加载检查点，同时加载挂件状态"""
        super().load_checkpoint(filename_or_checkpoint)

        if isinstance(filename_or_checkpoint, str):
            plugin_path = filename_or_checkpoint.replace('checkpoint', 'plugin_state')
            if os.path.exists(plugin_path):
                plugin_state = torch.load(plugin_path, map_location=self.device)
                for name, module in self.network.named_modules():
                    if name in plugin_state and hasattr(module, 'dual_branch_plugin'):
                        module.dual_branch_plugin.load_state_dict(plugin_state[name])
                        self.print_to_log_file(f"Loaded plugin state for {name}")

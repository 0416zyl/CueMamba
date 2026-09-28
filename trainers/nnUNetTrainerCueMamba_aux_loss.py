"""
带双分支挂件的 nnUNetTrainerUMambaBot - 带辅助监督版本

基于现有 nnUNetTrainerUMambaBot 扩展
添加双分支挂件用于增强肝脏动脉期分割

改进点：
1. 返回中间结果 P（粗分割）
2. 计算辅助 loss，直接监督 P 的质量
3. 其他改进：余弦相似度 + 加权平均 + 简化门控

使用方法:
    nnUNetv2_train Dataset501_HCC_MRI 3d_fullres 0 -tr nnUNetTrainerUMambaBotPluginAuxLoss
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union, List

# 导入原始 Trainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot

# 导入带辅助监督的双分支挂件
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'nets'))
from dual_branch_plugin_aux_loss import DualBranchPlugin3D_AuxLoss


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
    plugin_power = float(os.environ.get('PLUGIN_POWER', '1.0'))
    aux_loss_weight = float(os.environ.get('AUX_LOSS_WEIGHT', '0.3'))
    print(f"Adding dual-branch plugins (aux loss) to network... (power={plugin_power}, aux_loss_weight={aux_loss_weight})")

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

            plugin = DualBranchPlugin3D_AuxLoss(
                in_channels=dim,
                stage_level=stage_level,
                power=plugin_power,
                aux_loss_weight=aux_loss_weight
            )

            module.dual_branch_plugin = plugin
            print(f"  Added plugin to {name}: dim={dim}, stage={stage_level}")

    return network


# ============================================================
# Plugin Trainer - Auxiliary Loss Version
# ============================================================

class nnUNetTrainerUMambaBotPluginAuxLoss(nnUNetTrainerUMambaBot):
    """
    带双分支挂件的 U-Mamba Bot 训练器 - 带辅助监督版本

    改进点：
    1. 返回中间结果 P（粗分割）
    2. 计算辅助 loss，直接监督 P 的质量
    3. 其他改进：余弦相似度 + 加权平均 + 简化门控
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.plugin_power = float(os.environ.get('PLUGIN_POWER', '1.0'))
        self.aux_loss_weight = float(os.environ.get('AUX_LOSS_WEIGHT', '0.3'))
        self.print_to_log_file(f"Dual-Branch Plugin (Aux Loss) enabled: power={self.plugin_power}, aux_loss_weight={self.aux_loss_weight}")
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
        """
        训练步骤 - 重写以支持辅助 loss

        关键修改：
        1. 前向传播时返回中间结果 P
        2. 计算辅助 loss（监督 P 的质量）
        3. 将辅助 loss 加到总 loss 中
        """
        # 获取输入和标签
        data = batch['data']
        target = batch['target']

        # 移到 GPU
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # 清零梯度
        self.optimizer.zero_grad()

        # 前向传播（带中间结果）
        with self.amp_context():
            # 让网络的每个插件都返回中间结果
            output = None
            all_aux_outputs = []

            # 手动前向传播，收集所有插件的中间结果
            for name, module in self.network.named_modules():
                if hasattr(module, 'dual_branch_plugin'):
                    # 如果是第一个插件，先正常前向传播
                    if output is None:
                        # 临时修改插件，让它返回中间结果
                        original_forward = module.dual_branch_plugin.forward

                        def make_returning_forward(original_forward):
                            def returning_forward(x, return_aux=True):
                                return original_forward(x, return_aux=return_aux)
                            return returning_forward

                        module.dual_branch_plugin.forward = make_returning_forward(original_forward)

            # 正常前向传播
            output = self.network(data)

            # 收集所有插件的辅助 loss
            total_aux_loss = 0
            for name, module in self.network.named_modules():
                if hasattr(module, 'dual_branch_plugin'):
                    plugin = module.dual_branch_plugin
                    if hasattr(plugin, 'last_aux_outputs') and plugin.last_aux_outputs is not None:
                        aux_loss = plugin.compute_aux_loss(plugin.last_aux_outputs, target)
                        total_aux_loss = total_aux_loss + aux_loss

            # 计算主 loss
            main_loss = self.loss(output, target)

            # 总 loss = 主 loss + 辅助 loss
            total_loss = main_loss + total_aux_loss

        # 反向传播
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # 记录 loss
        self.print_to_log_file(
            f"main_loss: {main_loss.item():.4f}, "
            f"aux_loss: {total_aux_loss.item():.4f}, "
            f"total_loss: {total_loss.item():.4f}"
        )

        return total_loss

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

"""
nnUNet Trainer for CueMamba

Extends nnUNetTrainerUMambaBot with Dual-Branch Gating Module (DBGM)
for liver tumor segmentation on arterial-phase contrast-enhanced MRI

Usage:
    nnUNetv2_train Dataset501_HCC_MRI 3d_fullres 0 -tr nnUNetTrainerCueMamba
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union, List

# Import base trainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot

# Import DBGM module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'nets'))
from dual_branch_gating import DualBranchGatingModule3D


# ============================================================
# Helper functions (build_network_architecture is @staticmethod)
# ============================================================

def _get_stage_level(module_name):
    """Infer stage level from module name"""
    if 'encoder' in module_name.lower():
        for i in range(1, 5):
            if f'stage{i}' in module_name.lower() or f'blocks.{i-1}' in module_name.lower():
                return i
    elif 'decoder' in module_name.lower():
        for i in range(1, 5):
            if f'stage{i}' in module_name.lower():
                return i
    return 1


def _add_dbgm_to_network(network):
    """Add Dual-Branch Gating Module to each MambaLayer in the network"""
    dbgm_power = float(os.environ.get('DBGM_POWER', '2.0'))
    print(f"Adding DBGM modules to network... (power={dbgm_power})")

    for name, module in network.named_modules():
        if 'MambaLayer' in type(module).__name__:
            # Get dimension
            if hasattr(module, 'dim'):
                dim = module.dim
            else:
                for param in module.parameters():
                    dim = param.shape[0]
                    break

            stage_level = _get_stage_level(name)

            dbgm = DualBranchGatingModule3D(
                in_channels=dim,
                stage_level=stage_level,
                power=dbgm_power
            )

            module.dbgm = dbgm
            print(f"  Added DBGM to {name}: dim={dim}, stage={stage_level}")

    return network


# ============================================================
# CueMamba Trainer
# ============================================================

class nnUNetTrainerCueMamba(nnUNetTrainerUMambaBot):
    """
    CueMamba trainer: U-Mamba + Dual-Branch Gating Module (DBGM)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.dbgm_power = float(os.environ.get('DBGM_POWER', '2.0'))
        self.print_to_log_file(f"CueMamba DBGM enabled: power={self.dbgm_power}")
        self.print_to_log_file(f"Output folder: {self.output_folder}")

    @staticmethod
    def build_network_architecture(plans_manager, dataset_json,
                                   configuration_manager, num_input_channels,
                                   enable_deep_supervision=False):
        """
        Build CueMamba network architecture with DBGM
        Must be @staticmethod to match parent class signature for inference
        """
        # Build original network from parent class
        network = nnUNetTrainerUMambaBot.build_network_architecture(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, enable_deep_supervision
        )

        # Add DBGM to the network
        network = _add_dbgm_to_network(network)

        return network

    def train_step(self, batch):
        """Training step"""
        return super().train_step(batch)

    def on_train_epoch_end(self, train_logs):
        """Callback at end of training epoch"""
        super().on_train_epoch_end(train_logs)

        if self.current_epoch % 10 == 0:
            self._save_dbgm_state()

    def _save_dbgm_state(self):
        """Save DBGM module states"""
        dbgm_state = {}
        for name, module in self.network.named_modules():
            if hasattr(module, 'dbgm'):
                dbgm_state[name] = module.dbgm.state_dict()

        if dbgm_state:
            save_path = os.path.join(self.output_folder, f'dbgm_state_epoch{self.current_epoch}.pth')
            torch.save(dbgm_state, save_path)
            self.print_to_log_file(f"Saved DBGM state to {save_path}")

    def load_checkpoint(self, filename_or_checkpoint):
        """Load checkpoint and DBGM states"""
        super().load_checkpoint(filename_or_checkpoint)

        if isinstance(filename_or_checkpoint, str):
            dbgm_path = filename_or_checkpoint.replace('checkpoint', 'dbgm_state')
            if os.path.exists(dbgm_path):
                dbgm_state = torch.load(dbgm_path, map_location=self.device)
                for name, module in self.network.named_modules():
                    if name in dbgm_state and hasattr(module, 'dbgm'):
                        module.dbgm.load_state_dict(dbgm_state[name])
                        self.print_to_log_file(f"Loaded DBGM state for {name}")

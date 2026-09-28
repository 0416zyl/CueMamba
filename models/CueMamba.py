"""
CueMamba: Liver Tumor Segmentation with Dual-Branch Refinement

MambaLayer + Dual-Branch Gating Module (DBGM) integration
for 3D liver tumor segmentation on arterial-phase contrast-enhanced MRI
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union, List

from dual_branch_gating import DualBranchGatingModule3D


# ============================================
# MambaLayer with DBGM
# ============================================
class MambaLayerWithDBGM(nn.Module):
    """MambaLayer followed by Dual-Branch Gating Module"""

    def __init__(self, dim, dbgm_power=2.0, stage_level=1):
        super().__init__()
        self.dim = dim

        # Mamba SSM core
        from mamba_ssm import Mamba
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Dual-Branch Gating Module
        self.dbgm = DualBranchGatingModule3D(
            in_channels=dim,
            stage_level=stage_level,
            power=dbgm_power
        )

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        B, d_model = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        # Mamba processing
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        # Apply DBGM
        out = self.dbgm(out)

        return out


# ============================================
# BasicResBlock with DBGM
# ============================================
class BasicResBlockWithDBGM(nn.Module):
    """BasicResBlock followed by Dual-Branch Gating Module"""

    def __init__(self, conv_op, input_channels, output_channels, norm_op,
                 norm_op_kwargs, kernel_size=3, padding=1, stride=1,
                 use_1x1conv=False, nonlin=nn.LeakyReLU,
                 nonlin_kwargs={'inplace': True}, dbgm_power=2.0, stage_level=1):
        super().__init__()

        # Original BasicResBlock
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        # Dual-Branch Gating Module
        self.dbgm = DualBranchGatingModule3D(
            in_channels=output_channels,
            stage_level=stage_level,
            power=dbgm_power
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        y = self.act2(y)

        # Apply DBGM
        y = self.dbgm(y)

        return y


# ============================================
# ResidualMambaEncoder with DBGM
# ============================================
class ResidualMambaEncoderWithDBGM(nn.Module):
    """ResidualMambaEncoder with Dual-Branch Gating Module at each stage"""

    def __init__(self, input_size, input_channels, n_stages, features_per_stage,
                 conv_op, kernel_sizes, strides, n_blocks_per_stage,
                 conv_bias=False, norm_op=None, norm_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None, return_skips=False,
                 stem_channels=None, pool_type='conv', dbgm_power=2.0):
        super().__init__()

        # Original ResidualMambaEncoder initialization
        # ... (kept consistent with original)

        # Add DBGM to each stage
        self.dbgm_modules = nn.ModuleList([
            DualBranchGatingModule3D(
                in_channels=features_per_stage[s],
                stage_level=s + 1,
                power=dbgm_power
            )
            for s in range(n_stages)
        ])

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            # Apply DBGM
            x = self.dbgm_modules[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

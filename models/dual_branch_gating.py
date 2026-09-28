"""
Dual-Branch Gating Module (DBGM) for CueMamba

Branch 1: Local-Background Difference Branch - captures boundary-sensitive information
Branch 2: Prototype Similarity Branch - provides foreground-consistency information
Gated Fusion: adaptively combines the two branch outputs via residual fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DualBranchGatingModule(nn.Module):
    """
    Dual-Branch Gating Module (DBGM) - 2D version

    Branch 1: F -> AvgPool -> F_L -> F - F_L -> Conv -> E (local-background difference)
    Branch 2: F -> Conv -> P (coarse seg) -> S = P^r -> S*F -> Conv -> Proto -> M (similarity)
    Gated Fusion: D = |E - M|, F_c = Conv([F, E, M]), F' = F + alpha * D * F_c
    """

    def __init__(
        self,
        in_channels: int,
        stage_level: int = 1,
        power: float = 2.0,
        reduction: int = 4
    ):
        """
        Args:
            in_channels: number of input feature channels
            stage_level: stage index (1-4), determines pooling kernel size
            power: confidence power exponent (r), default 2.0
            reduction: channel reduction ratio, default 4
        """
        super().__init__()
        self.in_channels = in_channels
        self.stage_level = stage_level
        self.power = power

        # Adaptive kernel size based on stage level
        kernel_sizes = {1: 15, 2: 11, 3: 7, 4: 5}
        self.kernel_size = kernel_sizes.get(stage_level, 7)

        # Channel reduction
        reduced_channels = max(in_channels // reduction, 8)

        # ========== Branch 1: Local-Background Difference ==========
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # ========== Branch 2: Prototype Similarity ==========
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, 1),
            nn.Sigmoid()
        )

        # Spatial aggregation
        self.spatial_agg = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Similarity computation
        self.similarity = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # ========== Gated Fusion ==========
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

        # Learnable scaling parameter alpha
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

        # Residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input feature (B, C, H, W)
        Returns:
            output: enhanced feature (B, C, H, W)
        """
        B, C, H, W = x.shape

        # ========== Branch 1: Local-Background Difference ==========
        F_L = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1
        )
        E_diff = x - F_L
        E = self.branch1(E_diff)

        # ========== Branch 2: Prototype Similarity ==========
        P = self.branch2(x)
        S = torch.pow(P, self.power)
        S_expand = S * x
        lesion_proto = self.spatial_agg(S_expand)
        M = self.similarity(torch.cat([lesion_proto, x], dim=1))

        # ========== Gated Fusion ==========
        D = torch.abs(E - M)
        F_c = self.gate(torch.cat([x, E, M], dim=1))
        output = x + self.alpha * D * F_c

        return output


class MultiScaleDualBranchGating(nn.Module):
    """
    Multi-scale DBGM that applies DualBranchGatingModule to each encoder/decoder stage
    """

    def __init__(
        self,
        channel_list: List[int],
        num_encoder_stages: int = 4,
        num_decoder_stages: int = 3,
        power: float = 2.0
    ):
        """
        Args:
            channel_list: channel count per stage
            num_encoder_stages: number of encoder stages
            num_decoder_stages: number of decoder stages
            power: confidence power exponent
        """
        super().__init__()
        self.num_encoder_stages = num_encoder_stages
        self.num_decoder_stages = num_decoder_stages

        # Encoder DBGM modules (independent parameters per scale)
        self.encoder_dbgm = nn.ModuleList([
            DualBranchGatingModule(ch, stage_level=i + 1, power=power)
            for i, ch in enumerate(channel_list[:num_encoder_stages])
        ])

        # Decoder DBGM modules (independent parameters per scale, reversed kernel sizes)
        decoder_channels = list(reversed(channel_list[:num_decoder_stages + 1]))
        self.decoder_dbgm = nn.ModuleList([
            DualBranchGatingModule(ch, stage_level=num_decoder_stages - i + 1, power=power)
            for i, ch in enumerate(decoder_channels[1:])
        ])

    def forward_encoder(self, features_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply DBGM to each encoder stage"""
        return [
            dbgm(f) for dbgm, f in zip(self.encoder_dbgm, features_list)
        ]

    def forward_decoder(self, features_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply DBGM to each decoder stage"""
        return [
            dbgm(f) for dbgm, f in zip(self.decoder_dbgm, features_list)
        ]


class DualBranchGatingModule3D(nn.Module):
    """
    3D version of the Dual-Branch Gating Module for volumetric medical image segmentation
    """

    def __init__(
        self,
        in_channels: int,
        stage_level: int = 1,
        power: float = 2.0,
        reduction: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stage_level = stage_level
        self.power = power

        # 3D kernel sizes
        kernel_sizes = {1: 15, 2: 11, 3: 7, 4: 5}
        self.kernel_size = kernel_sizes.get(stage_level, 7)

        reduced_channels = max(in_channels // reduction, 8)

        # Branch 1: Local-Background Difference
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, reduced_channels, 1),
            nn.InstanceNorm3d(reduced_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(reduced_channels, in_channels, 1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Branch 2: Prototype Similarity
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, reduced_channels, 3, padding=1),
            nn.InstanceNorm3d(reduced_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(reduced_channels, 1, 1),
            nn.Sigmoid()
        )

        # Spatial aggregation
        self.spatial_agg = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Similarity computation
        self.similarity = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1),
            nn.InstanceNorm3d(in_channels),
            nn.Sigmoid()
        )

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels * 3, in_channels, 1),
            nn.InstanceNorm3d(in_channels)
        )

        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """3D forward pass"""
        # Branch 1: Local-Background Difference
        F_L = F.avg_pool3d(
            x,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1
        )
        E_diff = x - F_L
        E = self.branch1(E_diff)

        # Branch 2: Prototype Similarity
        P = self.branch2(x)
        S = torch.pow(P, self.power)
        S_expand = S * x
        lesion_proto = self.spatial_agg(S_expand)
        M = self.similarity(torch.cat([lesion_proto, x], dim=1))

        # Gated Fusion
        D = torch.abs(E - M)
        F_c = self.gate(torch.cat([x, E, M], dim=1))
        output = x + self.alpha * D * F_c

        return output

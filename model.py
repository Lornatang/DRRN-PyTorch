# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Realize the model definition function."""

import torch
from torch import nn


class RecursiveBlock(nn.Module):
    def __init__(self, num_channels: int, num_residual_unit: int):
        super(RecursiveBlock, self).__init__()
        self.num_residual_unit = num_residual_unit

        self.residual_unit = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1), bias=False),
        )

    def forward(self, x: torch.Tensor):
        out = x
        for _ in range(self.num_residual_unit):
            out = self.residual_unit(out)
            out = torch.add(out, x)

        return out


class DRRN(nn.Module):
    def __init__(self, num_residual_unit: int) -> None:
        super(DRRN, self).__init__()
        # Input layer
        self.conv1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(1, 128, (3, 3), (1, 1), (1, 1), bias=False),
        )

        # Features trunk blocks
        self.trunk = RecursiveBlock(128, num_residual_unit)

        # Output layer
        self.conv2 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 1, (3, 3), (1, 1), (1, 1), bias=False),
        )

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)
        out = torch.add(identity, out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

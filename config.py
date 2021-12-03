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
# ==============================================================================
"""Realize the parameter configuration function of dataset, model, training and verification code."""
import torch
from torch.backends import cudnn as cudnn

# ==============================================================================
# General configuration
# ==============================================================================
torch.manual_seed(0)
device = torch.device("cuda", 0)
cudnn.benchmark = True
upscale_factor = 2
num_residual_unit = 9
mode = "train"
exp_name = "DRRN_B1U9"

# ==============================================================================
# Training configuration
# ==============================================================================
if mode == "train":
    # Dataset
    # Image format
    train_image_dir = "data/TB291/DRRN/train"
    valid_image_dir = "data/TB291/DRRN/valid"
    # LMDB format
    train_lr_lmdb_path = "data/train_lmdb/DRRN/TB291_LR_lmdb"
    train_hr_lmdb_path = "data/train_lmdb/DRRN/TB291_HR_lmdb"
    valid_lr_lmdb_path = "data/valid_lmdb/DRRN/TB291_LR_lmdb"
    valid_hr_lmdb_path = "data/valid_lmdb/DRRN/TB291_HR_lmdb"

    image_size = 31
    batch_size = 128
    num_workers = 4

    # Incremental training and migration training
    resume = False
    strict = True
    start_epoch = 0
    resume_weight = ""

    # Total num epochs
    epochs = 80

    # SGD optimizer parameter (less training and low PSNR)
    model_optimizer_name = "sgd"
    model_lr = 1e-1
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False
    model_clip_gradient = 0.01

    # Adam optimizer parameter (faster training and better PSNR)
    # model_optimizer_name = "adam"
    # model_lr = 1e-1
    # model_betas = (0.9, 0.999)
    # model_clip_gradient = 0.01

    # Optimizer scheduler parameter
    lr_scheduler_name = "StepLR"
    lr_scheduler_step_size = 10
    lr_scheduler_gamma = 0.5

    print_frequency = 100

# ==============================================================================
# Verify configuration
# ==============================================================================
if mode == "valid":
    # Test data address
    lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/last.pth"
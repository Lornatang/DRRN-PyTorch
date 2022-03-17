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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 2
# number of residual units
num_residual_unit = 9
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "DRRN_B1U9"

if mode == "train":
    # Dataset
    train_image_dir = "data/TB291/DRRN/train"
    valid_image_dir = "data/TB291/DRRN/valid"
    test_lr_image_dir = f"data/Set5/GTmod12"
    test_hr_image_dir = f"data/Set5/GTmod12"

    image_size = 31
    batch_size = 128
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = ""

    # Total num epochs
    epochs = 50

    # SGD optimizer parameter
    model_lr = 1e-1
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    # Optimizer scheduler parameter
    lr_scheduler_step_size = 10
    lr_scheduler_gamma = 0.5

    # gradient clipping constant
    clip_gradient = 0.01

    print_frequency = 1000

if mode == "valid":
    # Test data address
    lr_dir = f"data/Set5/GTmod12"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/best.pth.tar"

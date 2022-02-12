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
"""Realize the function of dataset preparation."""
import os

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

import imgproc

__all__ = ["ImageDataset"]


class ImageDataset(Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.
    Args:
        dataroot         (str): Training data set address
    """

    def __init__(self, dataroot: str) -> None:
        super(ImageDataset, self).__init__()
        self.image_file_name = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]

        lr_dir_path = os.path.join(dataroot, "lr")
        hr_dir_path = os.path.join(dataroot, "hr")
        self.image_file_name = os.listdir(lr_dir_path)
        self.lr_filenames = [os.path.join(lr_dir_path, x) for x in self.image_file_name]
        self.hr_filenames = [os.path.join(hr_dir_path, x) for x in self.image_file_name]

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        # Read a batch of image data
        lr_image = Image.open(self.lr_filenames[batch_index])
        hr_image = Image.open(self.hr_filenames[batch_index])

        # Only extract the image data of the Y channel
        lr_image = np.array(lr_image).astype(np.float32)
        hr_image = np.array(hr_image).astype(np.float32)
        lr_ycbcr_image = imgproc.convert_rgb_to_ycbcr(lr_image)
        hr_ycbcr_image = imgproc.convert_rgb_to_ycbcr(hr_image)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_y_tensor = imgproc.image2tensor(lr_ycbcr_image, range_norm=False, half=False)
        hr_y_tensor = imgproc.image2tensor(hr_ycbcr_image, range_norm=False, half=False)

        return lr_y_tensor, hr_y_tensor

    def __len__(self) -> int:
        return len(self.image_file_name)

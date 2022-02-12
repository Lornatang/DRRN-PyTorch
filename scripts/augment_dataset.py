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
import argparse
import os
import shutil

from multiprocessing import Pool

from PIL import Image
from tqdm import tqdm


def main(args) -> None:
    if os.path.exists(args.lr_output_dir):
        shutil.rmtree(args.lr_output_dir)
    if os.path.exists(args.hr_output_dir):
        shutil.rmtree(args.hr_output_dir)
    os.makedirs(args.lr_output_dir)
    os.makedirs(args.hr_output_dir)

    # Get all image paths
    image_file_names = os.listdir(args.images_dir)

    # Splitting images with multiple threads
    progress_bar = tqdm(total=len(image_file_names), unit="image", desc="Data augment")
    workers_pool = Pool(args.num_workers)
    for image_file_name in image_file_names:
        workers_pool.apply_async(worker, args=(image_file_name, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()


def worker(image_file_name, args) -> None:
    raw_image = Image.open(f"{args.images_dir}/{image_file_name}").convert("RGB")

    index = 1
    # Data augment
    for rotate_angle in [0, 90, 180, 270]:
        for flip_probability in [0, 1]:
            for scale_factor in [2, 3, 4]:
                hr_image = raw_image.rotate(rotate_angle, expand=True) if rotate_angle != 0 else raw_image
                hr_image = hr_image.transpose(Image.FLIP_LEFT_RIGHT) if flip_probability != 0 else hr_image
                # Process LR image
                lr_image = hr_image.resize([hr_image.width // scale_factor, hr_image.height // scale_factor], resample=Image.BICUBIC)
                lr_image = lr_image.resize([hr_image.width, hr_image.height], resample=Image.BICUBIC)
                # Save all images
                lr_image.save(f"{args.lr_output_dir}/{image_file_name.split('.')[-2]}_{index:04d}.{image_file_name.split('.')[-1]}")
                hr_image.save(f"{args.hr_output_dir}/{image_file_name.split('.')[-2]}_{index:04d}.{image_file_name.split('.')[-1]}")

                index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--images_dir", type=str, help="Path to input image directory.")
    parser.add_argument("--lr_output_dir", type=str, help="Path to generator lr image directory.")
    parser.add_argument("--hr_output_dir", type=str, help="Path to generator hr image directory.")
    parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.")
    args = parser.parse_args()

    main(args)

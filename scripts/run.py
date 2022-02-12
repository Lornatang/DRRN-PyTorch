import os

# Prepare dataset
# os.system("python ./augment_dataset.py --images_dir ../data/TB291/original --lr_output_dir ../data/TB291/DRRN/original/lr --hr_output_dir ../data/TB291/DRRN/original/hr")
# os.system("python ./prepare_dataset.py --lr_images_dir ../data/TB291/DRRN/original/lr --hr_images_dir ../data/TB291/DRRN/original/hr --lr_output_dir ../data/TB291/DRRN/train/lr --hr_output_dir ../data/TB291/DRRN/train/hr --image_size 32 --step 21 --num_workers 10")

# Split train and valid
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/TB291/DRRN/train --valid_images_dir ../data/TB291/DRRN/valid --valid_samples_ratio 0.1")

import os

# Prepare dataset
os.system("python ./prepare_dataset.py --inputs_dir ../data/TB291/original --output_dir ../data/TB291/DRRN/")

# Split train and valid
os.system("python ./split_train_valid_dataset.py --inputs_dir ../data/TB291/DRRN")

# Create LMDB database file
os.system("python ./create_lmdb_dataset.py --image_dir ../data/TB291/DRRN/train/inputs --lmdb_path ../data/train_lmdb/DRRN/TB291_LR_lmdb")
os.system("python ./create_lmdb_dataset.py --image_dir ../data/TB291/DRRN/train/target --lmdb_path ../data/train_lmdb/DRRN/TB291_HR_lmdb")

os.system("python ./create_lmdb_dataset.py --image_dir ../data/TB291/DRRN/valid/inputs --lmdb_path ../data/valid_lmdb/DRRN/TB291_LR_lmdb")
os.system("python ./create_lmdb_dataset.py --image_dir ../data/TB291/DRRN/valid/target --lmdb_path ../data/valid_lmdb/DRRN/TB291_HR_lmdb")

import os
import random
import shutil
from pathlib import Path

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_data(src_folder, dest_folder):
    # Create destination directories
    train_dir = os.path.join(dest_folder, 'train')
    test_dir = os.path.join(dest_folder, 'eval_source')
    val_dir = os.path.join(dest_folder, 'val_source')
    visual_test_dir = os.path.join(dest_folder, 'visual_test_source')
    
    for directory in [train_dir, test_dir, val_dir, visual_test_dir]:
        create_directory(directory)
    
    # Get all images in the source folder
    image_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    random.shuffle(image_files)
    
    # Calculate split sizes
    total_images = len(image_files)
    train_split = int(total_images * 0.70)
    test_split = int(total_images * 0.12)
    val_split = int(total_images * 0.12)
    visual_test_split = total_images - (train_split + test_split + val_split)
    
    # Split images into different sets
    train_files = image_files[:train_split]
    test_files = image_files[train_split:train_split + test_split]
    val_files = image_files[train_split + test_split:train_split + test_split + val_split]
    visual_test_files = image_files[train_split + test_split + val_split:]
    
    # Function to copy files
    def copy_files(file_list, dest):
        for file_name in file_list:
            shutil.copy(os.path.join(src_folder, file_name), os.path.join(dest, file_name))
    
    # Copy files to respective directories
    copy_files(train_files, train_dir)
    copy_files(test_files, test_dir)
    copy_files(val_files, val_dir)
    copy_files(visual_test_files, visual_test_dir)
    
    print(f"Total images: {total_images}")
    print(f"Train: {len(train_files)}, Test: {len(test_files)}, Val: {len(val_files)}, Visual Test: {len(visual_test_files)}")
    print("Data split and copied successfully.")

# Usage
src_folder = './masks'
dest_folder = 'my_dataset'
split_data(src_folder, dest_folder)

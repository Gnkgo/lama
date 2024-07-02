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
    validation_dir = os.path.join(dest_folder, 'validation')
    
    for directory in [train_dir, validation_dir]:
        create_directory(directory)
    
    # Get all images in the source folder
    image_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    random.shuffle(image_files)
    
    # Calculate split sizes
    total_images = len(image_files)
    train_split = int(total_images * 0.80)    
    # Split images into different sets
    train_files = image_files[:train_split]
    validation_files = image_files[train_split:]
    
    # Function to move files
    def move_files(file_list, dest):
        for file_name in file_list:
            shutil.move(os.path.join(src_folder, file_name), os.path.join(dest, file_name))
    
    # Move files to respective directories
    move_files(train_files, train_dir)
    move_files(validation_files, validation_dir)

    
    print(f"Total images: {total_images}")
    print(f"Train: {len(train_files)}, Val: {len(validation_files)}")
    print("Data split and moved successfully.")

# Usage
src_folder = './masks'
dest_folder = 'my_dataset'
split_data(src_folder, dest_folder)

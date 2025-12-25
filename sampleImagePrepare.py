import os
import shutil
import random

def pick_images_from_folders(source_dir, dest_dir, num_images):
    # Loop through each subfolder in the source directory
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        if os.path.isdir(folder_path):
            # Destination subfolder
            dest_subfolder = os.path.join(dest_dir, folder_name)
            os.makedirs(dest_subfolder, exist_ok=True)

            # Get list of image files in the folder
            image_files = [f for f in os.listdir(folder_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            # Randomly pick images (or all if fewer)
            selected_images = random.sample(image_files, min(num_images, len(image_files)))

            # Copy selected images into the corresponding destination subfolder
            for img in selected_images:
                src = os.path.join(folder_path, img)
                dst = os.path.join(dest_subfolder, img)
                shutil.copy2(src, dst)
                print(f"Copied: {src} -> {dst}")



# Example usage
# source_directory = "data/DevanagariHandwrittenCharacterDataset/Train"
# destination_directory = "data/DevanagariHandwrittenCharacterDataset/SmallDataSet800/Train"
# x = 800  # Number of images to pick per folder

source_directory = "data/DevanagariHandwrittenCharacterDataset/Test"
destination_directory = "data/DevanagariHandwrittenCharacterDataset/SmallDataSet200/Test"
x = 200  # Number of images to pick per folder

pick_images_from_folders(source_directory, destination_directory, x)


import os
import cv2
import shutil
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import numpy as np
from mypath import Path

base_dir = Path.db_root_dir('7b52009b64fd0a2a49e6d8a939753077792b0554')

input_video_folder = os.path.join(base_dir, 'video_frame')
output_frame_folder = os.path.join(base_dir, 'video_frame')
image_folder = os.path.join(output_frame_folder, "TBD/images")
gt_folder = os.path.join(output_frame_folder, "TBD/gt")
output_train_folder = os.path.join(base_dir, 'train')
output_test_folder = os.path.join(base_dir, 'test')
output_val_folder = os.path.join(base_dir, 'val')

num_augmentations = 10  # Number of augmented images to create per original image

# Function to create a folder if it doesn't exist
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Function to apply color jitter to an image
def apply_color_jitter(image):
    pil_image = Image.fromarray(image)
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    augmented_pil_image = color_jitter(pil_image)
    augmented_image = np.array(augmented_pil_image)
    return augmented_image

# Function to apply noise to an image
def apply_noise(image):
    noise = np.random.randn(*image.shape) * 25
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

# Function to apply Gaussian blur to an image
def apply_gaussian_blur(image):
    kernel_size = (5, 5)
    return cv2.GaussianBlur(image, kernel_size, 0)

# Function to copy files from source to destination
def copy_files(filenames, source_folder, dest_folder):
    for filename in filenames:
        source_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(dest_folder, filename)
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        shutil.copy2(source_path, dest_path)

# Load filenames
all_frame_filenames_sat = [filename for filename in os.listdir(image_folder) if filename.endswith("_sat.png")]
all_frame_filenames_gt = [filename for filename in os.listdir(gt_folder) if filename.endswith("_gt.png")]

# Data augmentation on _sat.png images and _gt.png masks
os.makedirs(output_frame_folder, exist_ok=True)

# Create lists to store augmented filenames
augmented_filenames_sat = []
augmented_filenames_gt = []

# Find the last index in the output folder
existing_image_files = os.listdir(os.path.join(output_frame_folder, "images"))
existing_indices = [int(filename.split("_")[0][5:]) for filename in existing_image_files]

if existing_indices:
    last_index = max(existing_indices)
else:
    last_index = -1  # If no files exist, start from -1

for sat_filename, gt_filename in zip(all_frame_filenames_sat, all_frame_filenames_gt):
    frame_number = int(sat_filename.split("_")[0][5:])
    sat_image_path = os.path.join(image_folder, sat_filename)
    gt_path = os.path.join(gt_folder, gt_filename)

    # Load _sat.png and _gt.png images
    sat_image = cv2.imread(sat_image_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    for i in range(num_augmentations):
        augmented_frame_number = last_index + 1 + i  # Start numbering from the next index

        augmented_sat_image = apply_color_jitter(sat_image)
        augmented_sat_image = apply_noise(augmented_sat_image)
        augmented_sat_image = apply_gaussian_blur(augmented_sat_image)

        # Save augmented _sat.png and _gt.png images back to video_frame
        augmented_sat_filename = f"frame{augmented_frame_number:04d}_sat.png"
        augmented_gt_filename = f"frame{augmented_frame_number:04d}_gt.png"
        augmented_sat_path = os.path.join(output_frame_folder, "images", augmented_sat_filename)
        augmented_gt_path = os.path.join(output_frame_folder, "gt", augmented_gt_filename)

        cv2.imwrite(augmented_sat_path, augmented_sat_image)
        cv2.imwrite(augmented_gt_path, gt_mask)

        # Append augmented filenames to lists
        augmented_filenames_sat.append(augmented_sat_filename)
        augmented_filenames_gt.append(augmented_gt_filename)

# Check if the augmented dataset is empty
if augmented_filenames_sat:
    total_samples = len(augmented_filenames_sat)
    split_ratio = 0.2  # 20% for test and validation each

    # Calculate the number of samples for test and validation
    test_val_samples = int(total_samples * split_ratio)

    # Calculate the number of samples for train
    train_samples = total_samples - 2 * test_val_samples

    # Split the filenames into train, test, and validation sets based on order
    train_filenames_sat = augmented_filenames_sat[:train_samples]
    test_filenames_sat = augmented_filenames_sat[train_samples:train_samples + test_val_samples]
    val_filenames_sat = augmented_filenames_sat[train_samples + test_val_samples:]

    train_filenames_gt = augmented_filenames_gt[:train_samples]
    test_filenames_gt = augmented_filenames_gt[train_samples:train_samples + test_val_samples]
    val_filenames_gt = augmented_filenames_gt[train_samples + test_val_samples:]

    print(f"Number of train _sat.png images: {len(train_filenames_sat)}")
    print(f"Number of test _sat.png images: {len(test_filenames_sat)}")
    print(f"Number of validation _sat.png images: {len(val_filenames_sat)}")

    print(f"Number of train _gt.png images: {len(train_filenames_gt)}")
    print(f"Number of test _gt.png images: {len(test_filenames_gt)}")
    print(f"Number of validation _gt.png images: {len(val_filenames_gt)}")

    # Copy files to appropriate folders
    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)
    os.makedirs(output_val_folder, exist_ok=True)

    # Copy files to train, test, and val folders
    copy_files(train_filenames_sat, os.path.join(output_frame_folder, "images"), os.path.join(output_train_folder, "images"))
    copy_files(test_filenames_sat, os.path.join(output_frame_folder, "images"), os.path.join(output_test_folder, "images"))
    copy_files(val_filenames_sat, os.path.join(output_frame_folder, "images"), os.path.join(output_val_folder, "images"))

    copy_files(train_filenames_gt, os.path.join(output_frame_folder, "gt"), os.path.join(output_train_folder, "gt"))
    copy_files(test_filenames_gt, os.path.join(output_frame_folder, "gt"), os.path.join(output_test_folder, "gt"))
    copy_files(val_filenames_gt, os.path.join(output_frame_folder, "gt"), os.path.join(output_val_folder, "gt"))

else:
    print("No augmented images to split.")

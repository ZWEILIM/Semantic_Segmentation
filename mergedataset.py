import os
import shutil
import random
from mypath import Path

# Define the dataset names
dataset_names = ['7b52009b64fd0a2a49e6d8a939753077792b0554', 'c1dfd96eea8cc2b62785275bca38ac261256e278','bd307a3ec329e10a2cff8fb87480823da114f8f4','b1d5781111d84f7b3fe45a0852e59758cd7a87e5','17ba0791499db908433b80f37c5fbc89b870084b']

# Define the target directory for the merged dataset
merged_dataset_dir = r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\mergedataset'

# Define the target directory for the merged dataset within the video_frame directory
video_frame_merged_dataset_dir = os.path.join(merged_dataset_dir, 'video_frame')

# Create the merged dataset directory within the video_frame directory if it doesn't exist
if not os.path.exists(video_frame_merged_dataset_dir):
    os.makedirs(video_frame_merged_dataset_dir)

# Create the merged dataset directory if it doesn't exist
if not os.path.exists(merged_dataset_dir):
    os.makedirs(merged_dataset_dir)

# # Initialize the last assigned index for both training and validation images
# last_index = 0 # Use a list to store the value as a mutable object

# Define the number of images to take from each dataset
num_images_per_category = 3  # for training
num_images_per_category_val = 1  # for validation

# Create directories for training and validation in the merged dataset
train_dir = os.path.join(merged_dataset_dir, 'train')
val_dir = os.path.join(merged_dataset_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Create subdirectories for gt and images in train and val
subdirectories = ['gt', 'images']
for subdir in subdirectories:
    train_subdir = os.path.join(train_dir, subdir)
    val_subdir = os.path.join(val_dir, subdir)
    os.makedirs(train_subdir, exist_ok=True)
    os.makedirs(val_subdir, exist_ok=True)

# Create directories for connect_8_d1 and connect_8_d3 in the video_frame directory
connect_d1_dir = os.path.join(video_frame_merged_dataset_dir, 'connect_8_d1')
connect_d3_dir = os.path.join(video_frame_merged_dataset_dir, 'connect_8_d3')
dest_video_frame_gt_dir = os.path.join(video_frame_merged_dataset_dir, 'gt')
dest_video_frame_sat_dir = os.path.join(video_frame_merged_dataset_dir, 'images')
os.makedirs(connect_d1_dir, exist_ok=True)
os.makedirs(connect_d3_dir, exist_ok=True)
os.makedirs(dest_video_frame_gt_dir, exist_ok=True)
os.makedirs(dest_video_frame_sat_dir, exist_ok=True)

# Create a list to keep track of copied image filenames
copied_image_filenames = []

# Create a dictionary to store the mapping between original and renamed filenames
filename_mapping = {}

# Create a dictionary to store the mapping between original connect label filenames and their corresponding image filenames
connect_label_mapping = {}

# Function to copy and rename images
def copy_and_rename_images(image_list, dest_dir, last_index, train_counter, val_counter):
    for i, src_gt in enumerate(image_list):
        # Use the original filename to access images
        original_gt = src_gt.replace('_gt', '_sat')
        src_img = os.path.join(images_dir, original_gt)

        print(f'Dest dir: {dest_dir}')
        print(f'last_index: {last_index[0]}')  # Access the value inside the list

        # Increment the index by 1 for renaming purposes
        last_index[0] += 1  # Update the value inside the list

        dest_gt = os.path.join(dest_dir, 'gt', f'frame{last_index[0]:04d}_gt.png')
        dest_img = os.path.join(dest_dir, 'images', f'frame{last_index[0]:04d}_sat.png')

        print(f'dest_gt: {dest_gt}')

        # Store the mapping between original and renamed filenames
        filename_mapping[src_gt] = f'frame{last_index[0]:04d}_gt.png'
        filename_mapping[src_img] = f'frame{last_index[0]:04d}_sat.png'

        shutil.copyfile(os.path.join(gt_dir, src_gt), dest_gt)
        shutil.copyfile(src_img, dest_img)

        copied_image_filenames.append((src_gt, dest_gt))
        copied_image_filenames.append((src_img, dest_img))

        # Copy images back to the video_frame directory
        dest_gt_video_frame = os.path.join(dest_video_frame_gt_dir, f'frame{last_index[0]:04d}_gt.png')
        dest_img_video_frame = os.path.join(dest_video_frame_sat_dir, f'frame{last_index[0]:04d}_sat.png')

        shutil.copyfile(os.path.join(gt_dir, src_gt), dest_gt_video_frame)
        shutil.copyfile(src_img, dest_img_video_frame)

        if dest_dir == train_dir:
            train_counter[0] += 1
        elif dest_dir == val_dir:
            val_counter[0] += 1

        # Generate connect label filenames using the original_gt filename
        for variant in range(3):
            original_connect_gt = original_gt.replace('_sat', f'_gt_{variant}')
            connect_filename = f'frame{last_index[0]:04d}_gt_{variant}.png'
            src_connect_d1 = os.path.join(dataset_dir, 'video_frame/connect_8_d1', f'{original_connect_gt}')
            src_connect_d3 = os.path.join(dataset_dir, 'video_frame/connect_8_d3', f'{original_connect_gt}')

            dest_connect_d1 = os.path.join(connect_d1_dir, connect_filename)
            dest_connect_d3 = os.path.join(connect_d3_dir, connect_filename)

            shutil.copyfile(src_connect_d1, dest_connect_d1)
            shutil.copyfile(src_connect_d3, dest_connect_d3)

            # Store the mapping between original connect label filenames and their corresponding image filenames
            connect_label_mapping[connect_filename] = (original_gt, src_gt)

# Initialize counters for training and validation
train_counter = [0]
val_counter = [0]

# Initialize last_index to -1
last_index = [-1]

# Loop through each dataset in dataset_names
for dataset_name in dataset_names:
    dataset_dir = Path.db_root_dir(dataset_name)
    gt_dir = os.path.join(dataset_dir, 'video_frame/gt')
    images_dir = os.path.join(dataset_dir, 'video_frame/images')
    
    gt_files = os.listdir(gt_dir)
    random.shuffle(gt_files)  # Shuffle the list for random selection
    
    # Split the shuffled images into training and validation sets
    train_images = gt_files[:num_images_per_category]
    val_images = gt_files[num_images_per_category + 1:num_images_per_category + num_images_per_category_val +1]

    # Copy images for training
    copy_and_rename_images(train_images[:num_images_per_category], train_dir, last_index, train_counter, val_counter)

    # Copy images for validation
    copy_and_rename_images(val_images[:num_images_per_category_val], val_dir, last_index, train_counter, val_counter)

# Print the final value of last_index
print(f'Final last_index: {last_index[0]}')

print(f'Total images in train set: {train_counter[0]}')
print(f'Total images in validation set: {val_counter[0]}')
# Now you can use the connect_label_mapping dictionary to access the original filenames of connect label images
# and their corresponding image filenames. For example:
# original_image_filename, original_gt_filename = connect_label_mapping[connect_label_filename]

# Print a list of copied image filenames for reference
for src, dest in copied_image_filenames:
    print(f'Copied: {src} --> {dest}')

print('Image copying and renaming complete.')
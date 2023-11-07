import numpy as np
import cv2
import os, time
from tqdm import tqdm
from mypath import Path  
import re


# set PYTHONPATH=E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet
# set PYTHONPATH=C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet

def find_latest_connect_label(savedir):
    connect_labels = [filename for filename in os.listdir(savedir) if re.match(r'frame(\d+)_gt_\d\.png', filename)]
    if connect_labels:
        latest_connect_label = max(connect_labels, key=lambda x: int(re.search(r'frame(\d+)_gt_\d\.png', x).group(1)))
        return int(re.search(r'frame(\d+)_gt_\d\.png', latest_connect_label).group(1))
    else:
        return -1  # If no connect labels are found, start from -1

def direction_process_d1(imgpath, savedir, base_dir):
    # print("Input image path:", imgpath)  # Add this line
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = np.where(img > 0, 1, 0)
    shp = img.shape

    # img_pad = np.zeros([shp[0] + 4, shp[0] + 4])
    img_pad = np.zeros([shp[0] + 4, shp[1] + 4])  # Adjust the shape here
    img_pad[2:-2, 2:-2] = img
    dir_array0 = np.zeros([shp[0], shp[1], 3])
    dir_array1 = np.zeros([shp[0], shp[1], 3])
    dir_array2 = np.zeros([shp[0], shp[1], 3])

    for i in range(shp[0]):
        for j in range(shp[1]):
            if img[i, j] == 0:
                continue
            dir_array0[i, j, 0] = img_pad[i, j]
            dir_array0[i, j, 1] = img_pad[i, j + 2]
            dir_array0[i, j, 2] = img_pad[i, j + 4]
            dir_array1[i, j, 0] = img_pad[i + 2, j]
            dir_array1[i, j, 1] = img_pad[i + 2, j + 2]
            dir_array1[i, j, 2] = img_pad[i + 2, j + 4]
            dir_array2[i, j, 0] = img_pad[i + 4, j]
            dir_array2[i, j, 1] = img_pad[i + 4, j + 2]
            dir_array2[i, j, 2] = img_pad[i + 4, j + 4]

    # Find the latest connect label
    latest_connect_label = find_latest_connect_label(savedir)

    # Increment the latest label
    latest_connect_label += 1

    # Update the connect label for the saved images
    file_prefix = f"frame{latest_connect_label:04d}_gt"  # Use the latest label without increment
    file_path_0 = os.path.join(savedir, f"{file_prefix}_0.png")
    file_path_1 = os.path.join(savedir, f"{file_prefix}_1.png")
    file_path_2 = os.path.join(savedir, f"{file_prefix}_2.png")


    # file_prefix = imgpath.split('\\')[-1].split('.')[0]
    # file_path_0 = r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d1' + f"\\{file_prefix}_0.png"
    # file_path_1 = r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d1' + f"\\{file_prefix}_1.png"
    # file_path_2 = r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d1' + f"\\{file_prefix}_2.png"
    # file_path_0 = r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d1' + f"\\{file_prefix}_0.png"
    # file_path_1 = r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d1' + f"\\{file_prefix}_1.png"
    # file_path_2 = r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d1' + f"\\{file_prefix}_2.png"


    # Print the full paths
    # print("File path 0:", file_path_0)
    # print("File path 1:", file_path_1)
    # print("File path 2:", file_path_2)

    try:
        # Save the images
        cv2.imwrite(file_path_0, dir_array0 * 255)
        cv2.imwrite(file_path_1, dir_array1 * 255)
        cv2.imwrite(file_path_2, dir_array2 * 255)
        print("Images saved successfully")
    except Exception as e:
        print("Error saving images:", e)

    
def direction_process_d3(imgpath, savedir, base_dir):
    
    # print("Input image path:", imgpath)  # Add this line
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = np.where(img > 0, 1, 0)
    shp = img.shape

    # img_pad = np.zeros([shp[0] + 8, shp[0] + 8])
    img_pad = np.zeros([shp[0] + 8, shp[1] + 8])  # Adjust the shape here
    img_pad[4:-4, 4:-4] = img
    dir_array0 = np.zeros([shp[0], shp[1], 3])
    dir_array1 = np.zeros([shp[0], shp[1], 3])
    dir_array2 = np.zeros([shp[0], shp[1], 3])

    for i in range(shp[0]):
        for j in range(shp[1]):
            if img[i, j] == 0:
                continue
            dir_array0[i, j, 0] = img_pad[i, j]
            dir_array0[i, j, 1] = img_pad[i, j + 4]
            dir_array0[i, j, 2] = img_pad[i, j + 8]
            dir_array1[i, j, 0] = img_pad[i + 4, j]
            dir_array1[i, j, 1] = img_pad[i + 4, j + 4]
            dir_array1[i, j, 2] = img_pad[i + 4, j + 8]
            dir_array2[i, j, 0] = img_pad[i + 8, j]
            dir_array2[i, j, 1] = img_pad[i + 8, j + 4]
            dir_array2[i, j, 2] = img_pad[i + 8, j + 8]

    
    # Find the latest connect label
    latest_connect_label = find_latest_connect_label(savedir)

    # Increment the latest label
    latest_connect_label += 1

    # Update the connect label for the saved images
    file_prefix = f"frame{latest_connect_label:04d}_gt"  # Use the latest label without increment
    file_path_0 = os.path.join(savedir, f"{file_prefix}_0.png")
    file_path_1 = os.path.join(savedir, f"{file_prefix}_1.png")
    file_path_2 = os.path.join(savedir, f"{file_prefix}_2.png")

    # file_prefix = imgpath.split('\\')[-1].split('.')[0]
    # file_path_0 = r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d3' + f"\\{file_prefix}_0.png"
    # file_path_1 = r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d3' + f"\\{file_prefix}_1.png"
    # file_path_2 = r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d3' + f"\\{file_prefix}_2.png"
    # file_path_0 = r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d3' + f"\\{file_prefix}_0.png"
    # file_path_1 = r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d3' + f"\\{file_prefix}_1.png"
    # file_path_2 = r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\7b52009b64fd0a2a49e6d8a939753077792b0554\video_frame\connect_8_d3' + f"\\{file_prefix}_2.png"


    # Print the full paths
    # print("File path 0:", file_path_0)
    # print("File path 1:", file_path_1)
    # print("File path 2:", file_path_2)

    try:
        # Save the images
        cv2.imwrite(file_path_0, dir_array0 * 255)
        cv2.imwrite(file_path_1, dir_array1 * 255)
        cv2.imwrite(file_path_2, dir_array2 * 255)
        print("Images saved successfully")
    except Exception as e:
        print("Error saving images:", e)

    # cv2.imwrite(os.path.join(savedir, imgpath.split('/')[-1].split('.')[0] + '_0' + '.png'), dir_array0 * 255)
    # cv2.imwrite(os.path.join(savedir, imgpath.split('/')[-1].split('.')[0] + '_1' + '.png'), dir_array1 * 255)
    # cv2.imwrite(os.path.join(savedir, imgpath.split('/')[-1].split('.')[0] + '_2' + '.png'), dir_array2 * 255)

# def batch_process(gt_dir, savedir_d1, savedir_d3, base_dir):  
#     for i in tqdm(os.listdir(gt_dir)):
#         if i.split('.')[-1] != 'png':
#             continue
#         img_path = os.path.join(gt_dir, i)  # Full path to the input image
#         direction_process_d1(img_path, savedir_d1, base_dir)
#         direction_process_d3(img_path, savedir_d3, base_dir)

def batch_process(gt_dir, savedir_d1, savedir_d3, base_dir):
    latest_connect_label = find_latest_connect_label(savedir_d1)  # Get the latest connect label

    gt_images = [img for img in os.listdir(gt_dir) if img.endswith('.png')]
    total_images = len(gt_images)  # Total number of GT images

    # Calculate the number of remaining images to process
    remaining_images = total_images - (latest_connect_label + 1)

    if remaining_images <= 0:
        print("No remaining images to process.")
        return

    for i in tqdm(range(remaining_images)):
        # Generate file paths
        file_prefix = f"frame{latest_connect_label + i + 1:04d}_gt"  # Start from the next label

        # Look for the corresponding GT image in the "gt" folder
        gt_file = f"{file_prefix}.png"
        gt_img_path = os.path.join(gt_dir, gt_file)

        # Check if the GT image exists
        if not os.path.exists(gt_img_path):
            print(f"GT image not found for label: {latest_connect_label + i + 1}")
            continue

        # Rest of your code to process and save the images
        direction_process_d1(gt_img_path, savedir_d1, base_dir)
        direction_process_d3(gt_img_path, savedir_d3, base_dir)

        # Generate file paths for the saved images in connect_d1
        file_path_0_d1 = os.path.join(savedir_d1, f"{file_prefix}_0.png")
        file_path_1_d1 = os.path.join(savedir_d1, f"{file_prefix}_1.png")
        file_path_2_d1 = os.path.join(savedir_d1, f"{file_prefix}_2.png")

        # Generate file paths for the saved images in connect_d3
        file_path_0_d3 = os.path.join(savedir_d3, f"{file_prefix}_0.png")
        file_path_1_d3 = os.path.join(savedir_d3, f"{file_prefix}_1.png")
        file_path_2_d3 = os.path.join(savedir_d3, f"{file_prefix}_2.png")

        # Print the full paths for both directories
        print("File path 0 (connect_d1):", file_path_0_d1)
        print("File path 1 (connect_d1):", file_path_1_d1)
        print("File path 2 (connect_d1):", file_path_2_d1)

        print("File path 0 (connect_d3):", file_path_0_d3)
        print("File path 1 (connect_d3):", file_path_1_d3)
        print("File path 2 (connect_d3):", file_path_2_d3)

    print(f"Processed {remaining_images} images, 0 images remaining.")

def main():
    base_dir = Path.db_root_dir('7b52009b64fd0a2a49e6d8a939753077792b0554')
    gt_path = os.path.join(base_dir, 'video_frame', 'gt')
    connect_d1_path = os.path.join(base_dir, 'video_frame', 'connect_8_d1')
    connect_d3_path = os.path.join(base_dir, 'video_frame', 'connect_8_d3')

    start = time.time()
    batch_process(gt_path, connect_d1_path, connect_d3_path, base_dir)
    end = time.time()
    print('Finished Creating connectivity cube, time {0}s'.format(end - start))

if __name__ == "__main__":
    main()
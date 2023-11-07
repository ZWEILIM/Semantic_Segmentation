import cv2
import numpy as np
import os
# from utils.metrics import Evaluator
# from sklearn.metrics import jaccard_score, f1_score


# Define the Args class
class Args:
    def __init__(self):
       self.dataset = '356a192b7913b04c54574d18c28d46e6395428ab'  # Updated dataset path

# Create an instance of Args
args = Args()

# Define the base paths
input_base_path = r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\out_imgs_1300'  # Updated input base path
output_base_path = r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\out_imgs_1300'  # Updated output base path

dataset = args.dataset

# Folder containing all the images
image_folder = os.path.join(input_base_path, dataset, 'result6')
gt_folder = os.path.join(input_base_path, dataset, 'result6')
# gt_mask_folder = os.path.join(gt_folder, dataset, 'result6')
output_directory = os.path.join(output_base_path, dataset, 'OCV_result6')


# iou_scores = []
# dice_scores = []

os.makedirs(output_directory, exist_ok=True)

# Get a list of all image filenames
image_filenames = [filename for filename in os.listdir(image_folder) if filename.endswith("_sat.png")]

# Create an Evaluator for accuracy metrics
# evaluator = Evaluator(num_class=2)  # Assuming you have 2 classes, modify accordingly

for image_filename in image_filenames:
    # Construct paths to the original image, predicted mask image, and output directory
    original_image_path = os.path.join(image_folder, image_filename)
    pred_mask_image_path = os.path.join(image_folder, image_filename.replace("_sat.png", "_pred.png"))
    output_image_path = os.path.join(output_directory, image_filename.replace("_sat.png", "_result.png"))
    # gt_mask_image_path = os.path.join(gt_mask_folder, image_filename.replace("_sat.png", "_gt.png"))


    # Load the original image
    original_image = cv2.imread(original_image_path)

    # Load the predicted mask image
    pred_mask_image = cv2.imread(pred_mask_image_path, cv2.IMREAD_GRAYSCALE)

    # Load the ground truth mask image
    # gt_mask_image = cv2.imread(gt_mask_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the mask is binary (either 0 or 255)
    _, pred_mask_image = cv2.threshold(pred_mask_image, 1, 255, cv2.THRESH_BINARY)

     # Ensure the mask is binary (either 0 or 255)
    # _, gt_mask_image = cv2.threshold(gt_mask_image, 1, 255, cv2.THRESH_BINARY)

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(original_image, original_image, mask=pred_mask_image)

    # Save the result image to the output directory
    cv2.imwrite(output_image_path, result_image)

    # Convert the result image to grayscale
    result_image_gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Add the predicted mask and the result image to the evaluator
    # evaluator.add_batch(pred_mask_image, result_image_gray)

    # # Calculate the IoU and Dice Coefficient
    # iou = jaccard_score(gt_mask_image.flatten(), pred_mask_image.flatten())
    # dice = f1_score(gt_mask_image.flatten(), pred_mask_image.flatten())

    # iou_scores.append(iou)
    # dice_scores.append(dice)

    # # Calculate pixel accuracy
    # correct_pixels = np.sum((gt_mask_image == pred_mask_image) & (gt_mask_image > 0))
    # total_pixels = np.sum(gt_mask_image > 0)
    # pixel_accuracy = correct_pixels / total_pixels
    # pixel_accuracy_scores.append(pixel_accuracy)


# # Calculate the mean IoU, Dice Coefficient, and Pixel Accuracy
# mean_iou = np.mean(iou_scores)
# mean_dice = np.mean(dice_scores)
# mean_pixel_accuracy = np.mean(pixel_accuracy_scores)

# print(f"Mean IoU: {mean_iou}")
# print(f"Mean Dice Coefficient: {mean_dice}")
# print(f"Mean Pixel Accuracy: {mean_pixel_accuracy * 100:.2f}%")

# Calculate accuracy metrics
# Acc = evaluator.Pixel_Accuracy()
# Acc_class = evaluator.Pixel_Accuracy_Class()
# mIoU = evaluator.Mean_Intersection_over_Union()
# IoU = evaluator.Intersection_over_Union()
# Precision = evaluator.Pixel_Precision()
# Recall = evaluator.Pixel_Recall()
# F1 = evaluator.Pixel_F1()

# # Print or use the metrics as needed
# print("Pixel Accuracy:", Acc)
# print("Pixel Accuracy Class:", Acc_class)
# print("Mean IoU:", mIoU)
# print("IoU:", IoU)
# print("Precision:", Precision)
# print("Recall:", Recall)
# print("F1 Score:", F1)


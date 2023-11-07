import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.hub

# #set PYTHONPATH=C:\Program Files (x86)\Study\Degree\Intern\Intern Project\yolov5;%PYTHONPATH%
#ujsing deeplab_resnet50

# # Load the DeepLabV3 model with default weights
# model = deeplabv3_resnet50(pretrained=False)
# model.eval()


# # Load the pre-trained weights manually
# weights_path = "path_to_downloaded_weights/deeplabv3_resnet50_coco-cd0a2569.pth"
# model.load_state_dict(torch.load(weights_path))

# Attempt to load the deeplabv3_resnet50 model from torch.hub
model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
model.eval()

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up transformations to preprocess the frames
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Path to your input .ts videos
input_videos_path = "C:/Program Files (x86)/Study/Degree/Intern/Intern Project/CoANet/data/7b52009b64fd0a2a49e6d8a939753077792b0554"

# Output folders for the ground truth and images with road masks overlaid
output_gt_folder = "C:/Program Files (x86)/Study/Degree/Intern/Intern Project/CoANet/data/7b52009b64fd0a2a49e6d8a939753077792b0554/gt"
output_images_folder = "C:/Program Files (x86)/Study/Degree/Intern/Intern Project/CoANet/data/7b52009b64fd0a2a49e6d8a939753077792b0554/images"

# Create output directories if they don't exist
os.makedirs(output_gt_folder, exist_ok=True)
os.makedirs(output_images_folder, exist_ok=True)

def segment_frame(frame):
    # Preprocess the frame
    input_tensor = preprocess(frame).unsqueeze(0).to(device)
    
    # Run the forward pass to get the predictions
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Convert the output to a mask (1 for road, 0 for others)
    mask = (output.argmax(0) == 15).cpu().numpy().astype(np.uint8)
    
    return mask

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame and get the road mask
        road_mask = segment_frame(frame)
        
        # Save the ground truth road mask
        gt_save_path = os.path.join(output_gt_folder, f"gt_{i:04d}.png")
        cv2.imwrite(gt_save_path, road_mask * 255)
        
        # Overlay the road mask on the original frame
        overlay_frame = cv2.addWeighted(frame, 0.7, cv2.cvtColor(road_mask * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        # Save the image with road mask overlaid
        image_save_path = os.path.join(output_images_folder, f"image_{i:04d}.png")
        cv2.imwrite(image_save_path, overlay_frame)
    
    cap.release()

# Process all .ts videos in the input folder
for video_file in os.listdir(input_videos_path):
    if video_file.endswith(".ts"):
        video_path = os.path.join(input_videos_path, video_file)
        process_video(video_path)

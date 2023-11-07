import cv2
import os
from mypath import Path

def extract_frames_from_ts_videos(base_dir, output_folder, max_frames=30):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the base directory
    for filename in os.listdir(base_dir):
        if filename.endswith(".ts"):  # Check if the file has a .ts extension
            # Get the full path of the video file
            video_path = os.path.join(base_dir, filename)

            # Open the video file
            video = cv2.VideoCapture(video_path)

            # Get some video properties
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)

            # Determine the number of frames to extract (up to max_frames)
            num_frames_to_extract = min(max_frames, frame_count)

            # Loop through each frame and save them
            for frame_number in range(num_frames_to_extract):
                ret, frame = video.read()
                if not ret:
                    break  # Break the loop if the video ends

                # Save the frame with a unique name directly in the output folder
                frame_name = f"frame{frame_number:04d}_sat.png"
                frame_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(frame_path, frame)

            # Release the video object
            video.release()

            print(f"Extracted {num_frames_to_extract} frames from '{filename}' and saved in '{output_folder}' at {fps} FPS.")

# Provide the base directory and use it to construct the output folder
base_dir = Path.db_root_dir('fa35e192121eabf3dabf9f5ea6abdbcbc107ac3b')
input_folder = os.path.join(base_dir, 'video')
output_folder = os.path.join(base_dir, 'video_frame', 'TBD/images')
output_folder_gt = os.path.join(base_dir, 'video_frame', 'TBD/gt')

if not os.path.exists(output_folder_gt):
    os.makedirs(output_folder_gt)

extract_frames_from_ts_videos(input_folder, output_folder, max_frames=1)

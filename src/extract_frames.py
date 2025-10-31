import cv2
import os

def extract_frames(video_path, output_dir, interval=1, guest='Real 1'):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    # Get the base name of the video file (without extension)
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame every 'interval' seconds (assuming 30 FPS)
        if frame_count % (30 * interval) == 0:
            # Include the guest name in the file to avoid name conflicts
            frame_filename = os.path.join(output_dir, f"{guest}_{video_base}_frame_{saved_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1
    cap.release()

# Define the guest folder name (change as needed for Guest2, etc.)
guest_folder = "Fake"

# Directory containing videos for the specified guest
real_video_dir = f"Selected/{guest_folder}"
# Output directory for extracted frames
real_output_dir = "Selected/"
os.makedirs(real_output_dir, exist_ok=True)

# Process each video in the guest's folder
for video_file in os.listdir(real_video_dir):
    video_path = os.path.join(real_video_dir, video_file)
    extract_frames(video_path, real_output_dir, interval=1, guest=guest_folder)

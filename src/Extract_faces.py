from mtcnn import MTCNN
import cv2
import os

detector = MTCNN()

def crop_face(image_path, save_dir):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)
    if results:
        x, y, w, h = results[0]['box']
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

# Process all frames
guest_folder = "Extracted Fake"

input_dir = f"Selected/{guest_folder}"
output_dir = "Selected/Fake Crop"
os.makedirs(output_dir, exist_ok=True)

for frame_file in os.listdir(input_dir):
    frame_path = os.path.join(input_dir, frame_file)
    crop_face(frame_path, output_dir)
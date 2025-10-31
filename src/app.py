import gradio as gr
import tensorflow as tf
import numpy as np
from mtcnn.mtcnn import MTCNN  # Import MTCNN for face detection
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('deepfake_model_02.keras')

# Initialize the MTCNN face detector
detector = MTCNN()

def crop_face(image):
    """
    Detects faces in the image using MTCNN and crops the image to the first detected face.
    If no face is detected, returns the original image.
    """
    # Convert the PIL image to a numpy array
    image_np = np.array(image)
    
    # Detect faces in the image
    faces = detector.detect_faces(image_np)
    
    if len(faces) == 0:
        print("No face detected, using the original image.")
        return image
    
    # Use the first detected face
    face = faces[0]
    x, y, width, height = face['box']
    # Ensure the bounding box has non-negative coordinates
    x = max(0, x)
    y = max(0, y)
    
    # Crop the image to the face region
    face_image = image.crop((x, y, x + width, y + height))
    return face_image

def preprocess_image(image):
    """
    Resizes and normalizes the image.
    """
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

def predict(image):
    """
    Detects and crops the face from the input image before processing it for prediction.
    """
    # Crop the face from the uploaded image
    face_image = crop_face(image)
    
    # Preprocess the cropped face
    processed_image = preprocess_image(face_image)
    
    # Make a prediction with the model
    pred_prob = model.predict(processed_image)[0][0]
    return {"Real": float(1 - pred_prob), "Fake": float(pred_prob)}

# Gradio interface
title = "Deepfake Detector 🔍"
description = (
    "Upload an image to detect the face and check if it's real or AI-generated. "
    "The app uses MTCNN for face detection and a MobileNetV2 model trained on FakeForensics."
)

gr.Interface(
    fn=predict,  # Prediction function
    inputs=gr.Image(type="pil", label="Upload Image"),  # Input: Image upload
    outputs=gr.Label(num_top_classes=2),  # Output: Label with probabilities
    title=title,
    description=description,
    examples=["Real2.jpg", "Fake2.jpg"]  # Example images
).launch()




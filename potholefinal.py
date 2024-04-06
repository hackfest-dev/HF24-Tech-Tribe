import cv2
import pygame
from inference_sdk import InferenceHTTPClient

# Initialize Pygame
pygame.init()

# Define audio file paths for each language
audio_files = {
    "english": "potholedetected.mp3",
    "hindi": "samnegaddahai.mp3",
    "kannada":"mundegundiede.mp3"
    # Add more languages as needed
}

# Create a dictionary to map language names to their respective audio file paths
language_to_audio_path = {
    "english": audio_files["english"],
    "hindi": audio_files["hindi"],
    "kannada":audio_files["kannada"],

    # Add more languages here
}

# Create a function to play the notification sound based on the selected language
def play_notification_sound(language):
    if language in language_to_audio_path:
        audio_file_path = language_to_audio_path[language]
        notification_sound = pygame.mixer.Sound(audio_file_path)
        notification_sound.play()
    else:
        print("Language not supported")

# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Jj6pl8L4vQMxOdxpeGwK")

# Capture video from the camera
cap = cv2.VideoCapture(0)  # Change the argument to the appropriate camera index if you have multiple cameras

# Allow the user to select their preferred language
selected_language = input("Select your preferred language (e.g., 'english', 'hindi','kannada'): ").lower()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Run inference on the frame
    result = CLIENT.infer(frame, model_id="tech-tribe/1")

    # Extract and print object names
    if 'predictions' in result:
        object_names = [prediction['class'] for prediction in result['predictions']]
     

    # Check if "pothole" is detected
    if 'pothole' in object_names:
        # Play notification sound based on the selected language
        play_notification_sound(selected_language)

    # Draw bounding boxes
    for bounding_box in result['predictions']:
        x1 = bounding_box['x'] - bounding_box['width'] / 2
        x2 = bounding_box['x'] + bounding_box['width'] / 2
        y1 = bounding_box['y'] - bounding_box['height'] / 2
        y2 = bounding_box['y'] + bounding_box['height'] / 2
        box = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, box[0], box[1], (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Quit Pygame
pygame.quit()

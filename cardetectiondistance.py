import cv2
import pygame
import time  # Import the time module
from inference_sdk import InferenceHTTPClient

# Initialize Pygame
pygame.init()

# Define audio file paths for each language
audio_files = {
    "english": "cardetected.mp3",
    "hindi": "saamnegaadihai.mp3",
    "kannada":"mundegaadiede.mp3"
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

# Function to calculate object height from bounding box
def calculate_object_height(bbox):
    _, y_min, _, y_max = bbox
    object_height_pixels = abs(y_max - y_min)
    return object_height_pixels 

def calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, scale_height):
    # Calculate the scale factor
    scale_factor = scale_height / scale_height_pixels
    # Calculate the width of the scale
    height_of_scale = scale_pixels * scale_factor
    # Calculate the distance from the camera using similar triangles
    distance_from_camera = (scale_distance * scale_height_pixels) / scale_height
    # Adjust the distance by adding the scale's distance from the camera
    updated_distance = distance_from_camera + scale_distance
    return updated_distance  # Return only updated_distance


# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="PIT9ordmgBwkNFwOiLft")

# Capture video from the camera
cap = cv2.VideoCapture(0)  # Change the argument to the appropriate camera index if you have multiple cameras

# Allow the user to select their preferred language
selected_language = input("Select your preferred language (e.g., 'english', 'hindi','kannada'): ").lower()

# Given data for distance calculation
scale_pixels = 40
scale_height_pixels = 480
scale_height = 15  # in cm
scale_distance = 2 # in cm

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Run inference on the frame
    result = CLIENT.infer(frame, model_id="car_detection-3zagb/1")

    # Extract and print object names
    if 'predictions' in result:
        object_names = [prediction['class'] for prediction in result['predictions']]
      
    # Check if "pothole" is detected
    if '1' in object_names:
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

           # Calculate object height and distance from camera
        object_height_pixels = calculate_object_height((x1, y1, x2, y2))
        updated_distance = calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, object_height_pixels)


        # Display the distance from camera on the frame
        cv2.putText(frame, f"Updated Distance: {updated_distance:.2f} cm", (box[0][0], box[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Add a delay of 5 seconds
time.sleep(7)



# Quit Pygame
pygame.quit()
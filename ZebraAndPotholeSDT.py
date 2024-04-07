import cv2
import pygame
import time
from inference_sdk import InferenceHTTPClient

# Initialize Pygame
pygame.init()

# Define audio file paths for each language for zebracrossing detection
zebracrossing_detection_audio_files = {
    "english": "zebracrossingdetected.mp3",
    "hindi": "samnezebracrossinghai.mp3",
    "kannada": "mundezebracrossingede.mp3"
}

# Define audio file paths for each language for pothole detection
pothole_detection_audio_files = {
    "english": "potholedetected.mp3",
    "hindi": "samnegaddahai.mp3",
    "kannada": "mundegundiede.mp3"
}

# Create a dictionary to map language names to their respective audio file paths for zebracrossing detection
zebracrossing_detection_language_to_audio_path = {
    "english": zebracrossing_detection_audio_files["english"],
    "hindi": zebracrossing_detection_audio_files["hindi"],
    "kannada": zebracrossing_detection_audio_files["kannada"]
}

# Create a dictionary to map language names to their respective audio file paths for pothole detection
pothole_detection_language_to_audio_path = {
    "english": pothole_detection_audio_files["english"],
    "hindi": pothole_detection_audio_files["hindi"],
    "kannada": pothole_detection_audio_files["kannada"]
}

# Function to play the notification sound based on the selected language and type of detection
def play_notification_sound(language, detection_type):
    if detection_type == "zebracrossing":
        audio_files = zebracrossing_detection_audio_files
        language_to_audio_path = zebracrossing_detection_language_to_audio_path
    elif detection_type == "pothole":
        audio_files = pothole_detection_audio_files
        language_to_audio_path = pothole_detection_language_to_audio_path
    else:
        print("Invalid detection type")
        return

    if language in language_to_audio_path:
        audio_file_path = language_to_audio_path[language]
        notification_sound = pygame.mixer.Sound(audio_file_path)
        notification_sound.play()
    else:
        print("Language not supported")

def calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, object_height):
    # Calculate the distance from the camera using similar triangles
    distance_from_camera = (scale_distance * scale_height_pixels) / object_height
    # Adjust the distance by adding the scale's distance from the camera
    updated_distance = distance_from_camera + scale_distance
    return updated_distance

# Create inference clients for zebracrossing and pothole detection
ZEBRACROSSING_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Jj6pl8L4vQMxOdxpeGwK"
)

POTHOLE_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Jj6pl8L4vQMxOdxpeGwK"
)

# Capture video from the camera
cap = cv2.VideoCapture(0)  # Change the argument to the appropriate camera index if you have multiple cameras

# Allow the user to select their preferred language
selected_language = input("Select your preferred language (e.g., 'english', 'hindi', 'kannada'): ").lower()

# Given data for distance calculation
scale_pixels = 40
scale_height_pixels = 480
scale_height = 15  # in cm
scale_distance = 2 # in cm

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Run zebracrossing detection inference on the frame
    zebracrossing_result = ZEBRACROSSING_CLIENT.infer(frame, model_id="zebra-crossing-qh5uu/1")

    # Run pothole detection inference on the frame
    pothole_result = POTHOLE_CLIENT.infer(frame, model_id="tech-tribe/1")

    # Extract and print object names for zebracrossing detection
    zebracrossing_object_names = [prediction['class'] for prediction in zebracrossing_result['predictions']]

    # Extract and print object names for pothole detection
    pothole_object_names = [prediction['class'] for prediction in pothole_result['predictions']]

    # Check if "zebracrossing" is detected
    if '0' in zebracrossing_object_names:
        # Play notification sound for zebracrossing detection based on the selected language
        play_notification_sound(selected_language, "zebracrossing")

    # Check if "pothole" is detected
    if 'pothole' in pothole_object_names:
        # Play notification sound for pothole detection based on the selected language
        play_notification_sound(selected_language, "pothole")

    # Draw bounding boxes for zebracrossing detection
    for bounding_box in zebracrossing_result['predictions']:
        x1 = bounding_box['x'] - bounding_box['width'] / 2
        x2 = bounding_box['x'] + bounding_box['width'] / 2
        y1 = bounding_box['y'] - bounding_box['height'] / 2
        y2 = bounding_box['y'] + bounding_box['height'] / 2
        box = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, box[0], box[1], (0, 255, 0), 2)

        # Calculate distance from camera for zebracrossing
        distance_from_camera = calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, bounding_box['height'])
        cv2.putText(frame, f"Updated Distance: {distance_from_camera:.2f} cm", (box[0][0], box[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw bounding boxes for pothole detection
    for bounding_box in pothole_result['predictions']:
        x1 = bounding_box['x'] - bounding_box['width'] / 2
        x2 = bounding_box['x'] + bounding_box['width'] / 2
        y1 = bounding_box['y'] - bounding_box['height'] / 2
        y2 = bounding_box['y'] + bounding_box['height'] / 2
        box = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 2)  # Red color for pothole detection

        # Calculate distance from camera for pothole
        distance_from_camera = calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, bounding_box['height'])
        cv2.putText(frame, f"Updated Distance: {distance_from_camera:.2f} cm", (box[0][0], box[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(2)
# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Quit Pygame
pygame.quit()
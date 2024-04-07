import cv2
from ultralytics import YOLO
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def calculate_object_height(bbox):
    _, y_min, _, y_max = bbox
    object_height_pixels = abs(y_max - y_min)
    return object_height_pixels 

def calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, scale_height):
    scale_factor = scale_height / scale_height_pixels
    height_of_scale = scale_pixels * scale_factor
    distance_from_camera = (scale_distance * scale_height_pixels) / scale_height
    updated_distance = distance_from_camera + scale_distance
    return updated_distance

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use camera index 0 for the default camera

# Given data for distance calculation
scale_pixels = 40
scale_height_pixels = 480
scale_height = 15  # in cm
scale_distance = 10  # in cm

while True:
    # Capture frame-by-frame
    ret, frame = cap.read() 

    if ret:
        # Perform object detection on the frame
        results = model.predict(frame)

        # Calculate object height and distance from camera
        for result in results:
            for box in result.boxes.xyxy.tolist():
                object_height_pixels = calculate_object_height(box)
                updated_distance = calculate_distance_from_camera(scale_pixels, scale_height_pixels, scale_distance, object_height_pixels)

                # Draw bounding box
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Display the distance from camera on the frame
                cv2.putText(frame, f"Updated Distance: {updated_distance:.2f} cm", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Speak the distance in speech
                engine.say(f"The distance is {updated_distance:.2f} centimeters")
                engine.runAndWait()

        # Perform object tracking
        tracked_results = model.track(frame, persist=True, classes=[0, 2])

        # Display the resulting frame with tracking
        cv2.imshow('frame', tracked_results[0].plot())

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
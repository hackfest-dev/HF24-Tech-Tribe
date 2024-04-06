import cv2
from inference_sdk import InferenceHTTPClient

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

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Jj6pl8L4vQMxOdxpeGwK")

# Capture video from the camera
cap = cv2.VideoCapture(1)  # Use camera index 0 for the default camera

# Given data for distance calculation
scale_pixels = 40
scale_height_pixels = 480
scale_height = 15  # in cm
scale_distance = 2 # in cm

while True:
    # Capture frame-by-frame
    ret, frame = cap.read() 

    # Run inference on the frame
    result = CLIENT.infer(frame, model_id="tech-tribe/1")

    # Draw bounding boxes, calculate object height, and estimate distance
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
import cv2
from inference_sdk import InferenceHTTPClient

# create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Jj6pl8L4vQMxOdxpeGwK")

# Capture video from the camera
cap = cv2.VideoCapture(0)  # Change the argument to the appropriate camera index if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # run inference on the frame
    result = CLIENT.infer(frame, model_id="zebra-crossing-qh5uu/1")

    # Extract and print object names
    if 'predictions' in result:
        object_names = [prediction['class'] for prediction in result['predictions']]
        print( object_names)

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

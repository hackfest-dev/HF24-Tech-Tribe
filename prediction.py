import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')


# Open the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Detect objects in the frame
        results = model.track(frame, persist=True, classes=[0,1,2,3,7,13])

        # Display the annotated frame
        annotated_frame = results[0].plot()  # Assuming there's at least one result
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
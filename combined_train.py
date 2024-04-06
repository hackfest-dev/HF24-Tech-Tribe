import cv2
from ultralytics import YOLO

# Initialize both YOLO models
model1 = YOLO('best_pothole.pt')
model2 = YOLO('best_manhole.pt')

# Open the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Detect objects using both models
        results1 = model1.track(frame, persist=True)
        results2 = model2.track(frame, persist=True)

        # Merge the results from both models
        merged_results = results1 + results2

        # Display the annotated frame
        annotated_frame = merged_results[0].plot() if merged_results else frame
        cv2.imshow("YOLOv8n and Best1 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()


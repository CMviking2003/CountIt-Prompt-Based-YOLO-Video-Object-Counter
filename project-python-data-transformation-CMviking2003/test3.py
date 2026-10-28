import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt\


# --------  CONFIG  --------
VIDEO_SOURCE = "people_video.mp4"  # OR 0 for webcam, or RTSP stream
MODEL_PATH = "yolov8n.pt"

# Restricted zone polygon
restricted_zone = [(100, 100), (500, 100), (500, 400), (100, 400)]
# --------------------------

model = YOLO(MODEL_PATH).to("cpu")

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: could not open video source.")
    exit()

def point_inside_polygon(pt, poly):
    """Return True if point is inside polygon."""
    return cv2.pointPolygonTest(
        np.array(poly, dtype=np.int32),
        pt,
        False
    ) >= 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    intruder_detected = False

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls != 0:  # only detect people
            continue

        # Extract bounding box
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Check if inside restricted zone
        if point_inside_polygon((cx, cy), restricted_zone):
            intruder_detected = True
            cv2.putText(frame, "INTRUDER!", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # Draw restricted area
    cv2.polylines(frame, [np.array(restricted_zone)], True, (0, 0, 255), 3)

    # Big alert on screen
    if intruder_detected:
        cv2.putText(frame, "⚠ INTRUDER IN RESTRICTED AREA ⚠",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)

    # STREAM THE FRAME LIVE
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.001)
    plt.clf()

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

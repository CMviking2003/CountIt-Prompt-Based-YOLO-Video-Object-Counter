import cv2
from ultralytics import YOLO
import pandas as pd

video = "people_video.mp4"
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
model = YOLO("yolov8n.pt").to("cpu")

frame_interval = int(fps)  # 1 frame per second
#frame_interval = 1
frame_id = 0
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        results = model(frame)
        people = 0
        for r in results[0].boxes.cls:
            if int(r) == 0: # class 0 is 'person' in COCO dataset
                people += 1
        timestamp = frame_id 
        data.append({"frame": timestamp, "people": people})

    frame_id += 1

cap.release()

df = pd.DataFrame(data)
df.to_csv("people_counts.csv", index=False)

# use df.plot and save the figure
if not df.empty:
    ax = df.plot(x="frame", y="people", marker="o", linestyle="-", title="People count over time")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("people_counts.png")    
    print("Saved plot to people_counts.png")
else:
    print("No data to plot")




import cv2
import sys
from ultralytics import YOLO
import pandas as pd

'''
obj = sys.argv[1]    


if obj in "INSERT FUCKING DICTIONARY":


else:

    print("Object ")


    '''

video = "crows.mp4"
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
        thing = 0
        for r in results[0].boxes.cls:
            
            if int(r) != 0:

                continue

            elif int(r) == 0: # class 0 is 'person' in COCO dataset
                thing += 1
        timestamp = frame_id 
        data.append({"frame": timestamp, f"{obj}": thing})

    frame_id += 1

cap.release()

df = pd.DataFrame(data)
df.to_csv(f"{obj}_counts.csv", index=False)

# use df.plot and save the figure
if not df.empty:
    ax = df.plot(x="frame", y=f"{obj}", marker="o", linestyle="-", title=f"{obj} count over time")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(f"{obj}_counts.png")    
    print(f"Saved plot to {obj}_counts.png")
else:
    print("No data to plot")
import cv2
import sys 
from ultralytics import YOLO
import pandas as pd

import time
import os

obj = sys.argv[1].lower()
vid_name = sys.argv[2]

video = f"{vid_name}.mp4"
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
model = YOLO("yolov8n.pt").to("cpu")

frame_interval = int(fps)  # 1 frame per second
#frame_interval = 1
frame_id = 0
data = []

save_dir = 'detected_frames'
os.makedirs(save_dir, exist_ok = True)

count = 0



COCO_CLASSES = {

    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
    33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}


target_id = next((i for i, name in COCO_CLASSES.items() if name == obj), None)

if target_id is None:
    print(f"Error: '{obj}' not found in COCO class list.")
    sys.exit()



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:

        results = model(frame)
        target = 0
        con_scores = []



            
        for box in results[0].boxes:

            if int(box.cls) == target_id:

                con_scores.append(float(box.conf))
                target += 1

        if target > 0: # class 0 is 'person' in COCO dataset

            
            frame_thing = results[0].plot(conf=True)
            filename = os.path.join(save_dir, f"frames_{count:04d}.jpg")
            cv2.imwrite(filename, frame_thing)
                
            count += 1

        timestamp = frame_id 

        avg_con_score = sum(con_scores) / len(con_scores) if con_scores else 0


        data.append({"Frame": timestamp, f"{obj}": target, f"Average {obj} Score": avg_con_score})

    frame_id += 1

cap.release()

df = pd.DataFrame(data)



df.to_csv(f"{obj}_counts.csv", index=False)



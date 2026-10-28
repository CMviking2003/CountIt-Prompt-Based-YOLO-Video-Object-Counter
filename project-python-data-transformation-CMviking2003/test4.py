import cv2
import os

input_video = "SkibidiDop.mp4"
output_video = "output.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# Get input video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create video writer ONCE
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Write frame to output video
    out.write(frame)

cap.release()
out.release()

print("Done! Saved:", output_video)
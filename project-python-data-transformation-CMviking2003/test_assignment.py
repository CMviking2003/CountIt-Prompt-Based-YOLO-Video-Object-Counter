# test_main.py
import os

def test_additional_video_exists():
    # Check if there is a video file other than 'people_video.mp4'
    video_files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi', '.mov'))]
    assert any(video != "people_video.mp4" for video in video_files), "No additional video file found."

def test_main_uses_different_video():
    # Check if main.py uses a video other than 'people_video.mp4'
    with open("main.py", "r") as f:
        content = f.read()
    assert '"people_video.mp4"' not in content, "main.py is still using 'people_video.mp4'."

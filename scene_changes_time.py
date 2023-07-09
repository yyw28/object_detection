from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import cv2
from random import sample
from moviepy import *
import moviepy
from moviepy.editor import *
from PIL import Image

def detect_scene_changes(video_path, frame_location, threshold=500000):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    prev_frame = None
    scene_changes = [0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for simplicity
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Compute absolute difference between current and previous frame
            diff = cv2.absdiff(gray_frame, prev_frame)
            diff_sum = diff.sum()

            if diff_sum > threshold:
                scene_changes.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        prev_frame = gray_frame
    cap.release()

    i = 0
    for timestamp in scene_changes_time:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
          print("Failed to open video capture.")

        video_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        ret, frame = video_capture.read()

        filename = f"frame_{timestamp}"
        output_dir = frame_location + '/' + str(i) + '.jpg'
        cv2.imwrite(output_dir, frame)
        print("Frame saved successfully as", output_dir)
        i+=1

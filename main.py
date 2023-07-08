from visual_features import VisualFeatures
import cv2
import argparse
import os
from face_detection_er import *
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='video.mp4')
    parser.add_argument('--frame_interval', type=int, default=20)
    parser.add_argument('--frame_location', type=str, default='./frames')
    parser.add_argument('--weapons', type=bool, default=True)
    parser.add_argument('--ocr', type=bool, default=True)
    parser.add_argument('--objects', type=bool, default=True)
    parser.add_argument('--save_json', type=str, default='save_file.json')
    parser.add_argument('--emotion_recognition', type=bool, default=True)

    args = parser.parse_args()
    video_parser = detect_scene_changes(args.video_path)
    #video_parser.frame_extraction(args.frame_interval,args.frame_location)

    extractor = VisualFeatures()
    entries = os.listdir(args.frame_location)

    features = {}

    for entry in entries:
        print("Processing Frame:", entry)
        if entry!='video':
          img = cv2.imread(args.frame_location+'/'+entry)
          features[entry] = []
          if args.weapons:
              output = extractor.weapon_detection(img)
              features[entry].append(output)
          if args.ocr:
              output = extractor.text_ocr(img)
              features[entry].append(output)
          if args.objects:
              output = extractor.object_detection(img)
              features[entry].append(output)
          if args.emotion_recognition:
              output = extractor.emotion_recognition(img)
              features[entry].append(output)
    
    # json_object = json.dumps(features, indent = 4) 
    with open(args.save_json, 'w') as f:
        json.dump(features, f)



from flask import *
import cv2
import torch
import math
import pandas as pd
import requests
import ultralytics

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
camModel = torch.hub.load('ultralytics/yolov5', 'custom', path='bestModel.pt', force_reload=True)

faceLimit = 1
faceCount = 0

image_path = 'test.jpg'
frame = cv2.imread(image_path)

def process_image(frame):
    global faceLimit
    global faceCount
    faceCount = 0
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = camModel(frameRGB)
    for box in results.xyxy[0]:
        if faceCount < faceLimit:
            faceCount += 1
            if box[5] == 1:
                className = "Male:"
                bgr = (230, 216, 173)
            elif box[5] == 0:
                className = "Female:"
                bgr = (203, 192, 255)

            conf = math.floor(box[4] * 100)
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])

            cv2.rectangle(frame, (xA, yA), (xB, yB), (bgr), 4)
            cv2.rectangle(frame, (xA, yA - 50), (xA + 180, yA), (bgr), -1)
            cv2.putText(frame, str(conf), (xA + 130, yA - 13), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255))
            cv2.putText(frame, className, (xA, yA - 15), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255))
        else:
            break
    return frame

processed_frame = process_image(frame)
output_path = 'result.jpg'
cv2.imwrite(output_path, processed_frame)

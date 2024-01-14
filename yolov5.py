#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:41:45 2023

A simple example that show the use of OpenCV and the YOLOv5 model 
(from the torch.hub repository) for real-time object detection using a webcam. 

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True): 
oads the YOLOv5 model. Specifically, it loads the 'yolov5s' variant, 
which is a smaller and faster model, suitable for real-time applications. 
The pretrained=True argument ensures that the model is loaded with weights that 
have been pre-trained on a large dataset.
"""

import cv2
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Render the results on the frame
        frame = results.render()[0]

        # Display the frame
        cv2.imshow('YOLOv5 Webcam', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

import ultralytics
import torch
import cv2
from ultralytics import  YOLO
import os
from IPython.display import  display,Image
model = YOLO(r'C:\Users\muham\Downloads\10000_pic_30_epoch.pt')

import cv2
from ultralytics import YOLO



cap = cv2.VideoCapture(0)


while cap.isOpened():
    
    success, frame = cap.read()

    if success:
   
        results = model(frame,classes=2)

        annotated_frame = results[0].plot()

       
        cv2.imshow("YOLOv8 Inference", annotated_frame)

    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
       
        break


cap.release()
cv2.destroyAllWindows()



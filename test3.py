import cv2
import torch
import numpy as np
from vidgear.gears import CamGear
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#stream = CamGear(source='https://www.youtube.com/watch?v=h1wly909BYw', stream_mode = True, logging=True).start() # YouTube Video URL as input
stream = CamGear(source='https://www.youtube.com/watch?v=LY5YfLX-AVc', stream_mode = True, logging=True).start() # YouTube Video URL as input

count=0
while True:
    frame = stream.read()
    count += 1
    if count % 3 != 0:
        continue

    frame=cv2.resize(frame,(1020,600))
    results=model(frame)
    frame=np.squeeze(results.render())
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
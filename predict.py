import cv2
import numpy as np
from kalmanfilter import KalmanFilter
from detect_object import DetectColor

dc = DetectColor((20,100,100),(30,255,255)) # yellow
cap = cv2.VideoCapture(0)
kalman = KalmanFilter()


while True:

    ret,frame = cap.read()
    orriented_frame = cv2.flip(frame,1)

    if not ret:
        #ret false doncek kesin buraya girdiyse
        assert ret, "Error: Could not open or find the image."
        break

    else:
        dc.setFrame(orriented_frame)
        #cv2.drawContours(frame,dc.getContours(),0,(255,0,0),3)
        dc.drawCircle()
        x,y,w,h = dc.getPosProperties()
        cv2.circle(orriented_frame,kalman.predict(x+w //2 ,y+h //2),10,(0,255,0),2)
        cv2.imshow("frame",orriented_frame)
        #waitKey(0) diyince sure siniri olmadan banko bekliyor.ondan donuyor
        #input icin kucuk bir zaman vermemiz lazim yani
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
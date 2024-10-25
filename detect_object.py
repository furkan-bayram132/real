import cv2
import numpy as np

class DetectColor:


    def __init__(self,low,high):
        ##girecegimizi parametreler opencv hsv cinsinden en alt ve en ust renk siniri
        self.low_color = low 
        self.high_color = high
        self.mask = None
        self.frame = None

    def image2Binary(self):
        assert self.frame is not None, "PLEASE USE setFrame METHOD TO ASSIGN FRAME"
        hsv_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_image, self.low_color, self.high_color)
    

    def getContours(self):
        contours,_  = cv2.findContours(self.image2Binary(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # key liste icindeki elemanlarin her birini girdigimiz methoda tabi tutup ona gore azdan coka siraliyor
        #reverse = true dedigimizde coktan aza siraliyor
        return sorted(contours, key = cv2.contourArea, reverse = True)
    
    def getPosProperties(self):
        return cv2.boundingRect(self.getContours()[0])
    

    def drawRectangle(self):
        if(len(self.getContours()) > 0):

            x,y,w,h = self.getPosProperties()
            cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    def drawCircle(self):
        if(len(self.getContours()) > 0):
            x, y, w, h = cv2.boundingRect(self.getContours()[0])
            cv2.circle(self.frame, (x + w // 2, y + h // 2), 10, (255, 0, 0), 2)


    def setFrame(self,frame):

        self.frame = frame
    
                            
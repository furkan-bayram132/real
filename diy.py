from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from kalmanfilter import KalmanFilter

# args kısmına en son gel sen zaten webcamden goruntu alicaksin belki videoyu da parametreli olarak ekleyebilirsin

ap = argparse.ArgumentParser()

#bu gerideki noktalari cizdirmek icin i think !!!!!!!!!!!!
ap.add_argument("-b","--buffer",type = int, default=64, help = "max buffer size")
ap.add_argument("-v","--video",type = str, help = "path to your optional video (if not specified webcam will be used)" )

#ayrilmis argumanlari dictionarye cevir
args = vars(ap.parse_args())

#------------

kf = KalmanFilter()

#renk araliklari

red_lower1 = (161, 85, 186)
red_upper1 =  (180, 154, 255)
red_lower2 = (161, 85, 186)
red_upper2 = (180, 154, 255)

tracked_points = deque(maxlen = args["buffer"])


#video ya da webcam

#eger -v parametresi girilmediyse asagisi False donecek not ile True olacak
#yani webcam ile goruntu alacaz demek olur

if not args.get("video",False):
    #src == 0 demek webcam demek
    # VideoStream classi videoyu farkli bir threadde baslatir daha efficient  
    # live actionlar icin VideoStream libraryisi kullanilir
    stream = VideoStream(src = 0).start()
else:
    stream = cv2.VideoCapture(args["video"])

#kameralarin acilmasi icin sure
time.sleep(2.0)


while True:

    #frame'yi aliyoruz
    #VideoCapture classi icin de VideoStream classi icin de .read() methodu kullanilir
    frame = stream.read()

    #pythonda bos olmayan stringler true, bos stringler falsy
    #yani eger video girilcekse yani VideoCapture classi ise frame[1] cunku ret, frame donuyor
    #eger webcam ise frame
    frame = frame[1] if args.get("video",False) else frame

    #webcam calismayi durdurmus ya da video bitmisse donguden cik
    if frame is None:
        break

    #frameyi resize et, blurla ve hsvye cevir
        #frameyi kuculterek (width = 600) islenecek veri sayisini azaltiyoruz
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=600)
        #blurlamak noiseyi ve detaili dusurur
        #The kernel size must be positive and odd (e.g., 3, 5, 7, 9, 11).
        #daha buyuk deger daha cok blur demek 
        #biz blurred imageyi islicez ama asil frameyi display edecegiz
    blurred = cv2.GaussianBlur(frame,(11,11),0)


    hsvframe = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsvframe,red_lower1,red_upper1)
    mask2 = cv2.inRange(hsvframe,red_lower2,red_upper2)
    mask = mask1 | mask2 


    mask = cv2.erode(mask,None,iterations = 2)
    mask = cv2.dilate(mask,None,iterations = 2)

    cv2.imshow("mask", mask)

    #maskenin copysini kullaniyoruz cunku findContours fonksiyonu icine girilen 
    #maskede degisiklik yapiyor biz orijinal maskemizin manipule olmasini istemiyoruz
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    
    #yukarda zaten contourlari almistik
    #ama opencv2 ve 3 arasinda findContours methodunun return ettigi
    #degerler farkli, grap_contours methodu her platformda ayni calismasi
    #icin ortak bir deger donduruyor
    contours = imutils.grab_contours(contours)

    center = None
    
    #eger contour detect edebilmissek
    if len(contours) > 0:
        max_area_contour = max(contours, key= cv2.contourArea)
        #contour demek sinirlari olusturan noktalar demek
        #bu noktalar bir numpy arrayi icinde tutuluyor
        #bu contourlari iceren en kucuk cemberin merkezinin
        #konumu ve yaricapini aliyoruz
        #BU X VE Y DE NE OLUYOR CENTTERI ASAGIDA HESAPLADIK
        ((x,y),radius) = cv2.minEnclosingCircle(max_area_contour)

        moments = cv2.moments(max_area_contour)
        
        #verilen contoura gore bazi momentler hesaplaniyor yukarda
        #asagidaki oranlar centerx ve centery'yi veriyor
        
        #bu seklin agirlik merkezini veriyor
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

        #en buyuk contouru kapsayan cemberin bile radius 10dan kucukse o muhtemelen gurultudur hesabi
        #onu almiyoruz
        if(radius > 10):
            
            cv2.circle(frame,(int(x),int(y)),int(radius),(255,255,0),3)
            
            #centeroid
            cv2.circle(frame, center, 2, (0, 0, 255), -1)

        
        #gectigimiz noktayi double-ended queueye ekliyoruz
        #appendleft sola yani ilk siraya ekliyor en guncel elemani
        #quenin buffer sizesi asildikca en eski (sagdaki) elemanlar siliniyor
        tracked_points.appendleft(center)

        for i in range(1,len(tracked_points)):
            #son iki nokta arasi cizgi cekecegimizden birinin yoklugunda 
            #cizgiyi cekemeyiz continiue dememiz gerekir
            if tracked_points[i] is None or tracked_points[i-1] is None:
                continue
            else:
                #cizgilerin kalinligi gitgide artsin
                #tracked_pointsin en basindaki ogeler en guncel ogeler
                #en sonundaki ogeler en eski ogeler
                #so i arttikca eskiye gidiyoruz ve kalinlik azaliyor
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv2.line(frame, tracked_points[i - 1], tracked_points[i], (0, 0, 255), thickness)


    cv2.imshow("frame", frame)
    cv2.imshow("mask",mask)


    #bir miktar delay islemek icin gerekli kaldirirsan calismiyor
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#webcam kullaniliyorsa
if(not args.get("video", False)):
    stream.stop()
else:
    stream.release()
cv2.destroyAllWindows()
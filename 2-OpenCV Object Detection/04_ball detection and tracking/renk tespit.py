import cv2
import numpy as np
from collections import deque

buffer_size = 16
pts = deque(maxlen=buffer_size)

#mavi renk aralığı
blue_lower = (84,98,0)
blue_upper = (179,255,255)

#capture
capture = cv2.VideoCapture(0)
capture.set(3,960)
capture.set(4,480)

while True:
    
    success,imgOrigin = capture.read()
    
    if success:
        #blur
        blurred = cv2.GaussianBlur(imgOrigin,(11,11),0)
        
        #hsv
        hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        
        #mask Blue
        mask = cv2.inRange(hsv,blue_lower,blue_upper)
        
        #mask etrafı gürültüleri sil
        mask = cv2.erode(mask,None,iterations = 2)
        mask = cv2.dilate(mask,None,iterations = 2)
        
        #Kontur
        (contour,_) = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        
        if len(contour)>0:
            #en buyuk kontur al
            c = max (contour,key=cv2.contourArea)
            
            #rectangle çevir
            rect = cv2.minAreaRect(c)
            ((x,y),(width,height),rotation) = rect
            
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)
            
            # kutucuk
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            # moment
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            # konturu çizdir: imgOrigin
            cv2.drawContours(imgOrigin, [box], 0, (0,255,255),2)
            
            # merkere bir tane nokta çizelim: pembe
            cv2.circle(imgOrigin, center, 5, (255,0,255),-1)
            
            # bilgileri ekrana yazdır
            cv2.putText(imgOrigin, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
            
            
        # deque
        pts.appendleft(center)
        
        for i in range(1, len(pts)):
            
            if pts[i-1] is None or pts[i] is None: continue
        
            cv2.line(imgOrigin, pts[i-1], pts[i],(0,255,0),3) # 
            
        cv2.imshow("Orijinal Tespit",imgOrigin)
        
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
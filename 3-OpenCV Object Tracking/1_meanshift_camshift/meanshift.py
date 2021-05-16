import cv2 
import numpy as np

#open cam
cap = cv2.VideoCapture(0)

#read single frame 
ret, frame = cap.read()

if ret == False:
    print("Erorr...")
    
#Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) = tuple(face_rects[0])

track_window = (face_x, face_y, w, h) #Input Meanshift Algorithm
    
#Region Of İnterest
roi = frame[face_y:face_y+h:, face_x:face_x+w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0,180]) #Histogram needed for tracking
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#Stop Situations for tracking
#count = maximum number of objects to be calculated
#eps = changes
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)

while True:
    ret, frame = cap.read()
    
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window 
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 5)
        
        cv2.imshow("Takip",  img2)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
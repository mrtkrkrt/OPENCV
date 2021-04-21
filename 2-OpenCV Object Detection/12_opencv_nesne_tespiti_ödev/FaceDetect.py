import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("odev2.jpg",0)
plt.figure(),plt.imshow(img,cmap="gray"),plt.axis("off")

# resim üzerinde bulunan kenarları tespit edelim ve görselleştirelim edge detection
edges = cv2.Canny(image=img, threshold1=200, threshold2 = 255)
plt.figure(),plt.imshow(edges,cmap="gray"),plt.axis("off")

# yüz tespiti için gerekli haar cascade'i içe aktaralım
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect = cascade.detectMultiScale(img,minNeighbors=5)
for (x,y,w,h) in face_rect:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),10)
cv2.imshow("cross",img)


# HOG ilklendirelim insan tespiti algoritmamızı çağıralım ve svm'i set edelim
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(rects, weights) = hog.detectMultiScale(img, padding=(8, 8), scale=1.05)

for (xA, yA, xB, yB) in rects:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 0, 255), 2)
	
cv2.imshow("insan Tespiti", img)
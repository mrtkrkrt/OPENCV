import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("scene.jpeg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img)
plt.axis("off")
plt.show()

#calculate and visualize histogram
hist = cv2.calcHist([img],[0],None,[256],[0,256])

#Visualize normal
plt.figure()
plt.hist(img.ravel(),256,[0,256])
plt.axis("off")
plt.show()

#visualize using enum
plt.figure()

colors = ("b","g","r")
for i,index in enumerate(colors):
    hist = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color = index)
    plt.xlim([0,256])
    
plt.show()
    
    
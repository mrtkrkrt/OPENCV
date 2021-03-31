import cv2
import matplotlib.pyplot as plt
import numpy as np

# resmi içe aktar
img = cv2.imread("london.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

#without Threshold
edges = cv2.Canny(image = img ,threshold1 =0,threshold2=255)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

med_val = np.median(img)
print(med_val)

low = int(max(0,(1 - 0.33)*med_val))
high = int(min(255,(1 + 0.33)*med_val))

print("Low:",low)
print("High",high)

edges = cv2.Canny(image = img ,threshold1 =low,threshold2=high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

#blurring
blur_img = cv2.blur(img,ksize=(7,7))

med_val = np.median(blur_img)
print(med_val)

low = int(max(0,(1 - 0.33)*med_val))
high = int(min(255,(1 + 0.33)*med_val))

print("Low:",low)
print("High",high)

edges = cv2.Canny(image = blur_img ,threshold1 =low,threshold2=high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")









import cv2
import matplotlib.pyplot as plt

#Şablon eşleme 
img = cv2.imread("cat.jpg",0)
print(img.shape)

template = cv2.imread("cat_face.jpg",0)
print(template.shape)

h, w = template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


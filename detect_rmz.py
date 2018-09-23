from imutils import paths
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(13,5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,21))

image = cv2.imread('passport.png')
image = imutils.resize(image,height=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (3,3), 0)
blackchat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

gradX = cv2.Sobel(blackchat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                       


plt.imshow(thresh, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
	

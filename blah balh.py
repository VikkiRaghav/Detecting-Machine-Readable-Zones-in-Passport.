from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
from skimage import io, color
import matplotlib.pyplot as plt
import math
from scipy.stats import norm


def convolve(image, kernel, n):
	(iH, iW) = image.shape[:2]
	kH = 1
	kW = n
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			k = (roi * kernel).sum()
			output[y - pad, x - pad] = k

	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	return output

def convolve2(image, kernel, n):
       (iH, iW) = image.shape[:2]
       kH = n
       kW = 1
       pad = (kW - 1) // 2
       image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
       output = np.zeros((iH, iW), dtype="float32")
       for y in np.arange(pad, iH + pad):
               for x in np.arange(pad, iW + pad):
                       roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                       k = (roi * kernel).sum()
                       output[y - pad, x - pad] = k
                       output = rescale_intensity(output, in_range=(0, 255))
                       output = (output * 255).astype("uint8")
                       return output
	
       
	
                        

    

#def Reduce(image, kernel):
#	(iH, iW) = image.shape[:2]
	
#	factor = 0.5
#	kH = n
#	kW = 1
#	pad = (kW - 1) // 2
#	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
#		cv2.BORDER_REPLICATE)
#	output = np.zeros((iH, iW), dtype="float32")

#	for y in np.arange(pad, iH + pad):
 #               for x in np.arange(pad, iW + pad): 
                       


def getelements(n,mat):
  
  for i in range(n):
      for j in range(n):
          number = int(input("Enter the elements of the matrix"))
          mat[i][j] = number
          

def GaussianKernel(n):
        mid =  n / 2
        sigma=math.sqrt(2*1/np.pi)
        result = [(1/(sigma*math.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]
        return result
    


     
img = io.imread('pokemon.png')    # Load the image
img = color.rgb2gray(img)

n = input("Enret Number of Rows : ")
Kernel = np.zeros(n)
Kernel = GaussianKernel(n)
print(Kernel)
convolveOutput = convolve(img,Kernel,n)
Kernel = np.flipud(Kernel)
finalConvolveOutput = convolve2(convolveOutput,Kernel,n)
#finalImage = Reduce(finalConvolveOutput,Kernel)
plt.imshow(finalConvolveOutput, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

           
	

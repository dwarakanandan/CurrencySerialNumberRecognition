from PIL import Image
import PIL.ImageOps 
import numpy as np
import os
import cv2

def process():
	if not os.path.exists("grayscale"):
		os.makedirs("grayscale")

	directory = "digits"
	for k in os.listdir(directory):
		print "Processing image",k
		img = Image.open(directory+"/"+k).convert('L')
		imgarr = np.array(img)
		ret,thresh = cv2.threshold(imgarr,100,255,cv2.THRESH_BINARY_INV)
		kernel = np.ones((3,3),np.uint8)
		erosion = cv2.erode(thresh,kernel,iterations = 2)
		dilation = cv2.dilate(erosion,kernel,iterations = 2)
		img = Image.fromarray(dilation)
		img = img.resize((50,50), Image.ANTIALIAS)
		img.save("grayscale/"+k)

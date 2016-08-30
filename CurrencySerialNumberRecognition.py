import cv2
import numpy as np
import os
import random
import string
import cPickle
from PIL import Image

def gethsv(h,s,v):
	h = h/2
	s = int(s*255/100.0)
	v = int(v*255/100.0)
	return [h,s,v]
	
def rect_box(out,mask):
	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(mask,kernel,iterations = 1)
	dilation = cv2.dilate(erosion,kernel,iterations = 30)#tweak box size
	contours, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	boxes_part1 = []
	boxes_part2 = []
	for contour in contours:
		[x,y,w,h] = cv2.boundingRect(contour)
		ratio = float(w)/h
		if ratio >1.7 and ratio<2.0:
			boxes_part1.append(out[y:y+h,x:x+w])
		if ratio>3.0:
			boxes_part2.append(out[y:y+h,x:x+w])
	return boxes_part1,boxes_part2

def getmask(img,satu_val):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_red = np.array(gethsv(0,satu_val,20))#0-30 change for htc to redmi
	upper_red = np.array(gethsv(20,100,85))
	mask1 = cv2.inRange(hsv, lower_red, upper_red)
	lower_red = np.array(gethsv(340,51,20))
	upper_red = np.array(gethsv(360,100,85))
	mask2 = cv2.inRange(hsv, lower_red, upper_red)
	mask = cv2.add(mask1,mask2)
	return mask

def rect_digit(out,mask):
	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(mask,kernel,iterations = 1)
	dilation = cv2.dilate(erosion,kernel,iterations = 15)
	contours, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	digits = []
	for contour in contours:
		[x,y,w,h] = cv2.boundingRect(contour)
		if float(w)/h<1.0 and w*h > 5000:
			digits.append([out[y:y+h,x:x+w],x])
	return digits

def grayscale(img):
	img = Image.fromarray(img).convert('L')
	imgarr = np.array(img)
	ret,thresh = cv2.threshold(imgarr,100,255,cv2.THRESH_BINARY_INV)
	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(thresh,kernel,iterations = 2)
	dilation = cv2.dilate(erosion,kernel,iterations = 2)
	img = Image.fromarray(dilation)
	img = img.resize((50,50), Image.ANTIALIAS)
	return np.array(img)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def feedforward(a):
	for w,b in zip(weights,biases):
		a = sigmoid(w.dot(a)+b)
	return a

def predict(img):
	img = img.reshape(2500,1)/255.0
	predict_val = feedforward(img)
	return np.argmax(predict_val)

def process_digits(img):
	img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
	mask = getmask(img,0)
	digits = rect_digit(img,mask)
	digits = sorted(digits,key=lambda x: x[1])
	prediction = []
	for i,digit in enumerate(digits):
		print "  processing digit",i
		img = grayscale(digit[0])
		prediction.append(predict(img))
	print ''.join(str(i) for i in prediction)


img = cv2.imread("preprocess/notes/set1/12.jpg")
output = cPickle.load(open("27_18_14.p","rb"))
weights = output["weights"]
biases = output["biases"]
mask = getmask(img,30)
boxes_part1,boxes_part2 = rect_box(img,mask)
for i,box in enumerate(boxes_part2):
	print "processing box",i
	cv2.imshow("box"+str(i),box)
	process_digits(box)
cv2.waitKey(0)

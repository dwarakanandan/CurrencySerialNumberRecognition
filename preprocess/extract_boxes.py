import cv2
import numpy as np
import os
import random
import string

def gethsv(h,s,v):
	h = h/2
	s = int(s*255/100.0)
	v = int(v*255/100.0)
	return [h,s,v]

def rect(out,mask,i):
	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(mask,kernel,iterations = 1)
	dilation = cv2.dilate(erosion,kernel,iterations = 30)#tweak box size
	contours, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	global cnt_1,cnt_2
	for contour in contours:
		[x,y,w,h] = cv2.boundingRect(contour)
		ratio = float(w)/h
		name= ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
		if ratio >1.7 and ratio<2.0:
			cv2.imwrite("boxes/part1/"+name+".jpg",out[y:y+h,x:x+w])
			cv2.rectangle(out,(x,y),(x+w,y+h),(255,0,255),2)
		if ratio>3.0:
			cv2.imwrite("boxes/part2/"+name+".jpg",out[y:y+h,x:x+w])
			cv2.rectangle(out,(x,y),(x+w,y+h),(255,0,255),2)
	cv2.imwrite("boxes/original/"+i+".jpg",out)

def process(directory):
	if not os.path.exists("boxes/part1"):
		os.makedirs("boxes/part1")
		os.makedirs("boxes/part2")
		os.makedirs("boxes/original")

	for i in os.listdir(directory):
		print "processing image",i,"..."
		img = cv2.imread(directory+"/"+i)
		#img = cv2.resize(img,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		lower_red = np.array(gethsv(0,30,20))#0-30 change for htc to redmi
		upper_red = np.array(gethsv(20,100,85))
		mask1 = cv2.inRange(hsv, lower_red, upper_red)
		lower_red = np.array(gethsv(340,51,20))
		upper_red = np.array(gethsv(360,100,85))
		mask2 = cv2.inRange(hsv, lower_red, upper_red)
		mask = cv2.add(mask1,mask2)
		res = cv2.bitwise_and(hsv,hsv, mask= mask)
		rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
		blur = cv2.GaussianBlur(mask,(3,3),0)
		rect(img,mask,i)
		#cv2.waitKey(0)

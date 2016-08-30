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
	dilation = cv2.dilate(erosion,kernel,iterations = 15)
	#cv2.imshow("ero",erosion)
	#cv2.imshow("dil",dilation)
	#cv2.imshow("img",mask)
	contours, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	for contour in contours:
		[x,y,w,h] = cv2.boundingRect(contour)
		if float(w)/h<1.0 and w*h > 5000:
			name= ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
			cv2.imwrite("digits/"+name+".jpg",out[y:y+h,x:x+w])
			cv2.rectangle(out,(x,y),(x+w,y+h),(255,0,255),2)


def process():
	if not os.path.exists("digits"):
		os.makedirs("digits")

	directory = "boxes/part2"
	for i in os.listdir(directory):
		print "processing img",i
		img = cv2.imread(directory+"/"+i)
		img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		lower_red = np.array(gethsv(0,0,20))#0-30 change for htc to redmi
		upper_red = np.array(gethsv(20,100,85))
		mask1 = cv2.inRange(hsv, lower_red, upper_red)
		lower_red = np.array(gethsv(340,51,20))
		upper_red = np.array(gethsv(360,100,85))
		mask2 = cv2.inRange(hsv, lower_red, upper_red)
		mask = cv2.add(mask1,mask2)
		rect(img,mask,i)
		cv2.waitKey(0)

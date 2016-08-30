import os
import numpy as np
import cv2
import random
import cPickle

image = []
lable = []

def load_class(cname):
	for i in os.listdir("classified/"+str(cname)):
		img = cv2.imread("classified/"+str(cname)+"/"+i)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = img.reshape(2500,1)/255.0
		image.append(img)
		lable.append(cname)

def encode(num):
	arr = np.zeros((10,1))
	arr[num] = 1
	return arr

def load(num_test):
	for i in range(0,10):
		load_class(i)
	temp = zip(image,lable)
	random.shuffle(temp)
	images,lables = zip(*temp)
	images = list(images)
	lables = list(lables)
	num_train = len(images)-num_test
	train_x = images[:num_train]
	train_y_temp = lables[:num_train]
	test_x = images[num_train:]
	test_y = lables[num_train:]
	train_y = []
	for i in range(0,len(train_y_temp)):
		train_y.append(encode(train_y_temp[i]))
	training_data = [[x,y] for x,y in zip(train_x,train_y)]
	test_data = [[x,y] for x,y in zip(test_x,test_y)]
	return training_data,test_data

training_data,test_data = load(100)
save = {"training_data":training_data,"test_data":test_data}
output = open("data_1.pkl","wb")
cPickle.dump(save,output)
output.close()

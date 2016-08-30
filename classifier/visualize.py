import numpy as np
import matplotlib.pyplot as plt
import random
import cPickle

data = cPickle.load(open("data.pkl","rb"))
train = data["training_data"]
test = data["test_data"]

visualize = train


m =  len(visualize)-1

a = np.array([random.randint(0,m) for i in range(0,100)]).reshape(10,10)
img = np.zeros([10,10,50,50])
label = np.ones([10,10])
for i in range(0,10):
	for j in range(0,10):
		img[i][j] = visualize[a[i][j]][0].reshape(50,50)
		if visualize == train:
			label[i][j] = np.argmax(visualize[a[i][j]][1])
		else:
			label[i][j] = visualize[a[i][j]][1]
print label
big_pic = np.zeros((500,500))
for i in range(0,10):
	for j in range(0,10):
		for k in range(0,50):
			for l in range(0,50):
				big_pic[i*50+k][j*50+l] = img[i][j][k][l]


plt.imshow(big_pic,cmap="Greys_r")#display the big image
plt.show()

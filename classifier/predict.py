import cPickle
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(z):
		return 1.0/(1.0+np.exp(-z))

def feedforward(a):
	for w,b in zip(weights,biases):
		a = sigmoid(w.dot(a)+b)
	return a

def predict_train(data):
	count=0
	for x,y in data:
		predict = np.argmax(feedforward(x))
		if predict==np.argmax(y):
			count+=1
	return count

def predict_test(data):
	count=0
	for x,y in data:
		predict = np.argmax(feedforward(x))
		if predict== y:
			count+=1
	return count

output = cPickle.load(open("27_18_14.p","rb"))
weights = output["weights"]
biases = output["biases"]
data = cPickle.load(open("data.pkl","rb"))
training_data = data["training_data"]
test_data = data["test_data"]
print "Training accuracy = %.3f %% (%d/%d)"%(predict_train(training_data)*100/float(len(training_data)),predict_train(training_data),len(training_data))
print "Testing accuracy  = %.3f %% (%d/%d)"%(predict_test(test_data)*100/float(len(test_data)),predict_test(test_data),len(test_data))


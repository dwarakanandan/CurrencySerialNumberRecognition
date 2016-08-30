import random
import numpy as np
from time import strftime
import cPickle as pickle

class NeuralNetwork():
	
	def __init__(self,sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(x,1) for x in sizes[1:]]
		self.weights = [np.random.randn(x,y) for y,x in zip(sizes[:-1],sizes[1:])]
	
	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))
	
	def sigmoid_prime(self,z):
		return self.sigmoid(z)*(1-self.sigmoid(z))
		
	def feedforward(self,a):
		for w,b in zip(self.weights,self.biases):
			a = self.sigmoid(w.dot(a)+b)
		return a
	
	def SGD(self,train_data,epochs,batch_size,eta,test_data):
		n = len(train_data)
		n_test = len(test_data)
		print "Initial: Train Score = %d/%d  Test Score = %d/%d "%(self.predict_train(train_data),n,self.predict_test(test_data),n_test)
		for j in range(0,epochs):
			random.shuffle(train_data)
			mini_batches = [train_data[i:i+batch_size] for i in range(0,n,batch_size)]
			for mini_batch in mini_batches:
				self.GD(mini_batch,eta)
			print "Iteration %d : Train Score = %d/%d  Test Score = %d/%d "%(j,self.predict_train(train_data),n,self.predict_test(test_data),n_test)
		self.backup()

	def GD(self,mini_batch,eta):
		sum_biases = [np.zeros(i.shape) for i in self.biases]
		sum_weights = [np.zeros(i.shape) for i in self.weights]
		m = len(mini_batch)
		for x,y in mini_batch:
			delta_biases,delta_weights = self.backprop(x,y)
			sum_biases = [sb+db for sb,db in zip(sum_biases,delta_biases)]
			sum_weights = [sw+dw for sw,dw in zip(sum_weights,delta_weights)]
		self.weights = [w-(eta/m)*sw for w,sw in zip(self.weights,sum_weights)]
		self.biases = [b-(eta/m)*sb for b,sb in zip(self.biases,sum_biases)]
	
	def predict_test(self,test_data):
		count=0
		for x,y in test_data:
			predict = np.argmax(self.feedforward(x))
			if predict==y:
				count+=1
		return count
	
	def predict_train(self,train_data):
		count=0
		for x,y in train_data:
			predict = np.argmax(self.feedforward(x))
			if predict==np.argmax(y):
				count+=1
		return count
	
	def backup(self):
		fname = strftime("%M_%H_%d")+".p"
		output = {"weights":self.weights,"biases":self.biases}
		pickle.dump(output,open(fname,"wb"))
		print "Weights backed up to",fname
		
	def restore(self,fname):
		output = pickle.load(open(fname,"rb"))
		self.weights = output["weights"]
		self.biases = output["biases"]
		print "Weights successfully restored from",fname
		
	def backprop(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		activation = x
		activations = [x]
		zs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = self.sigmoid(z)
			activations.append(activation)
		delta = self.cost_derivative(activations[-1], y) * \
			self.sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in xrange(2, self.num_layers):
			z = zs[-l]
			sp = self.sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)
        
	def cost_derivative(self, output_activations, y):
		return (output_activations-y)


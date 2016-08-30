import cPickle
import neuralnetwork as nn

data = cPickle.load(open("data.pkl","rb"))
training_data = data["training_data"]
test_data = data["test_data"]
net = nn.NeuralNetwork([2500, 100, 10])
net.restore("28_17_14.p")
net.SGD(training_data, 50, 10, 1.0,test_data)

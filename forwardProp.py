import numpy as np
from numpy import random


class Network:

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        # num_hidden is a list because: It represents number of neurons in a hidden layer
        # default case [3, 5] --> has 2 hidden layers, 
        # first layer has 3 neurons and, second layer has 5.
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        #output of layers is the concatenation of all layer inputs --> default = [3, 3, 5, 2]

        #initiate random weights
        self.weights = []
        for i in range(len(layers)-1):
            #we create the random matrix of weights to connect to each layer
            #layer[i] is connected to layer[i+1]
            #So we get a layer[i] x layer[i+1] matrix 
            w = random.rand(layers[i], layers[i+1])
            self.weights.append(w)
        for w in self.weights:
            
            #we calculate the new inputs
            print (f"these are the individual weight matrix {w}")
    def forward_propagate(self, inputs):
        
        activations = inputs
        #we are looping through all the weight matrices (layers)
        for w in self.weights:
            #we calculate the new inputs
            
            net_inputs = np.dot(activations, w)
            print (f"activations = {activations},\n weights = {w},\n result = {net_inputs}")
            #calculate the activations
            activations = self.sigmoid(net_inputs)
            
        return activations
    def sigmoid(self, x):
        return 1 /(1 + np.exp(-x))


if __name__ == '__main__':
   
   # create a MLP
    nn = Network()
   # create some inputs
    inputs = random.rand(nn.num_inputs)
   # forward prop
    outputs = nn.forward_propagate(inputs)
   # print result
    print(f"Network input is: {inputs}")
    print(f"Network output is: {outputs}")
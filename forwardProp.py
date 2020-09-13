import numpy as np


class MLP:

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
            #we create the random matrix
            w = np.random.rand(layers[i], layers[i+1])
            

if __name__ == '__main__':
   

import numpy as np
from numpy import random
from random import random as ran



# Implement train






class Network(object):

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
        weights = []
        for i in range(len(layers)-1):
            #we create the random matrix of weights to connect to each layer
            #layer[i] is connected to layer[i+1]
            #So we get a layer[i] x layer[i+1] matrix 
            w = random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        # Save the activations and derivatives
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i]) 
            activations.append(a)

        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)

        self.derivatives = derivatives
            

   
    def forward_propagate(self, inputs):
        
        activations = inputs
        self.activations[0] = activations
        #we are looping through all the weight matrices (layers)
        for i, w in enumerate(self.weights):
            #we calculate the new inputs
            net_inputs = np.dot(activations, w)
            #print (f"activations = {activations},\n weights = {w},\n result = {net_inputs}")
            #calculate the activations
            activations = self.sigmoid(net_inputs)
            self.activations[i+1] = activations
            
        return activations

     # implement backpropagation  
    def backpropagate(self, error, verbose=False):

        # dE/dW[i] = (y  - a[i+1]) * s'(h[i+1]) * a[i]
        # s'(h[i+1]) s(h[i+1]) * (1 - s(h[i+1]))
        # s(h[i+1]) = a[i+1]
        #dE/dW[i - 1] = (y - a[i+1] * s'[i+1]) * s'(h[i+1]) W[i] * s'(h[i]]) * a[i-1]
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self.sigmoid_derivative(activations) # --> ndarry([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i] # --> ndarry([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if(verbose):
                print(f"Derivatives for W{i}: {self.derivatives[i]}")
        return error

    # Implement gradient descent

    def gradient_descent(self, learning_rate):
        
        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print(f"Original W{i},  {weights}")
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate 
            #print(f"updated W{i},  {weights}")


# train network with dummy dataset
    def train(self, inputs, targets, epochs, learning_rate):
        
        for i in range(epochs):
            sum_error = 0
            for j, (input, target) in enumerate(zip(inputs, targets)):
                output = self.forward_propagate(input)
                #calculate_error
                error = target - output
                #backpropagation
                self.backpropagate(error)
                #apply gradient descent
                self.gradient_descent(learning_rate)
                #report error
                sum_error += self.mse(target, output)
            print(f"Error: {sum_error/len(inputs)} at epoch {i}")


    def mse(self, target, output):
        return np.average((target - output) ** 2)


    def sigmoid_derivative(self, x):
        return x * (1 - x)


    def sigmoid(self, x):
        return 1 /(1 + np.exp(-x))


if __name__ == '__main__':
   
    # create a MLP
    nn = Network(2, [5], 1)
   
    # create some inputs
    inputs = np.array([[ran() / 2 for _ in range(2)] for _ in range(1000)])
    target = np.array([[i[0] + i[1]] for i in inputs])
     # train model
    nn.train(inputs, target, 100, 0.3)
    
    # make predictions

    #create dummy data

    input = np.array([.7, .1])
    target = np.array([.8])

    output = nn.forward_propagate(input)
    print(f"Our network believes that, {input[0]} + {input[1]} equals {output}")
   

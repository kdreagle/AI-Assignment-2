# mlp.py
# -------------

# mlp implementation
import util
import math
from random import *
from sklearn.neural_network import MLPClassifier
PRINT = True

class MLPClassifier:
    """
    mlp classifier
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mlp"
        self.max_iterations = max_iterations
        self.weights = []

    def sigmoid(self, input):
        try:
            return 1/(1+math.exp(-input))
        except OverflowError:
            return 1

    def sigmoidPrime(self, input):
        return input * (1 - input)

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):

        inputLayer = len(trainingData[0])   # 784 nodes (1 node per pixel)
        hiddenLayer = 100                   # this value can be changed (textbook says 300 is optimal)
        outputLayer = len(self.legalLabels) # 10 nodes (1 node per digit)
        layers = [inputLayer,hiddenLayer,outputLayer] # must be at least 3
        # initialize connections between input layer and hidden layer
        self.weights.append({})
        for i in range(inputLayer):
            for j in range(hiddenLayer):
                self.weights[0][(i,j)] = random()
        self.weights.append({})
        for i in range(hiddenLayer):
            for j in range(outputLayer):
                self.weights[1][(i,j)] = random()
        #for iteration in range(self.max_iterations):
        #   print "Starting iteration ", iteration, "..."
        for x in range(len(trainingData)):
            "*** YOUR CODE HERE ***"
            #print trainingLabels[x]
            # initialize the inputs for the input nodes
            a = []
            a.append([trainingData[x][pixel] for pixel in trainingData[x]])
            # visit all layers after input layer
            for layer in range(1,len(layers)):
                a.append([])
                for j in range(layers[layer]):
                    input = 0
                    for i in range(layers[layer-1]):
                        input += self.weights[layer-1][(i,j)] * a[layer-1][i]
                    a[layer].append(self.sigmoid(input))
            # back-propogate
            delta = [[],[]]
            for j in range(0,layers[-1]):
                delta[-1].append(self.sigmoidPrime(a[-2][j])*(trainingLabels[x]-a[-1][j]))
            for layer in range(len(layers)-2,0,-1):
                for i in range(0,layers[layer]):
                    delta[layer-1].append(self.sigmoidPrime(a[layer][i])*sum([self.weights[layer][(i,j)]*delta[layer][j] for j in range(layers[layer+1])]))
            # update weights --- needs to be changed to allow more than one hidden layer
            alpha = 0.001
            for i in range(inputLayer):
                for j in range(hiddenLayer):
                    self.weights[0][(i, j)] += alpha * a[0][i] * delta[0][j]
            for i in range(hiddenLayer):
                for j in range(outputLayer):
                    self.weights[1][(i, j)] += alpha * a[1][i] * delta[1][j]

    def classify(self, data ):

        inputLayer = len(data[0])  # 784 nodes (1 node per pixel)
        hiddenLayer = 100  # this value can be changed (textbook says 300 is optimal)
        outputLayer = len(self.legalLabels)  # 10 nodes (1 node per digit)
        layers = [inputLayer, hiddenLayer, outputLayer]  # must be at least 3

        guesses = []
        for x in range(len(data)):
            a = []
            a.append([data[x][pixel] for pixel in data[x]])
            for layer in range(1,len(layers)):
                a.append([])
                for j in range(layers[layer]):
                    input = 0
                    for i in range(layers[layer-1]):
                        input += self.weights[layer-1][(i,j)] * a[layer-1][i]
                    a[layer].append(self.sigmoid(input))
            guesses.append(1) # this needs to be changed

        return guesses

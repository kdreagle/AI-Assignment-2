# mlp.py
# -------------

# mlp implementation
import util
import math
import numpy as np

PRINT = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


class MLPClassifier:
    """
    mlp classifier
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mlp"
        self.max_iterations = max_iterations

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        rate = 0.001
        hidden = 50

        X = []
        y = []

        for i in range(len(trainingData)):
            X.append([])
            y.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            y[i][trainingLabels[i]] = 1
            #y.append(trainingLabels[i])
            for point in trainingData[i]:
                X[i].append(trainingData[i][point])

        X = np.array(X)
        y = np.array(y)

        self.w1 = 2 * np.random.random((len(X[0]), hidden)) - 1
        self.w2 = 2 * np.random.random((hidden, 10)) - 1
        for iteration in range(200):

            level1 = sigmoid(np.dot(X, self.w1))
            level2 = sigmoid(np.dot(level1, self.w2))

            level2_error = y - level2
            level2_delta = np.multiply(level2_error, sigmoid_deriv(level2))

            level1_error = np.dot(level2_delta, self.w2.T)
            level1_delta = np.multiply(level1_error, sigmoid_deriv(level1))

            self.w2 += rate * np.dot(level1.T, level2_delta)
            self.w1 += rate * np.dot(X.T, level1_delta)


    def classify(self, data):
        guesses = []
        X = []

        for i in range(len(data)):
            X.append([])
            for point in data[i]:
                X[i].append(data[i][point])

        X = np.array(X)

        level1 = sigmoid(np.dot(X, self.w1))
        level2 = sigmoid(np.dot(level1, self.w2))

        for output in level2:
            guesses.append(np.argmax(output))
        return guesses

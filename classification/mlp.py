# mlp.py
# -------------

# mlp implementation
import util
import math
import numpy as np

PRINT = True

def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))


def sigmoid_deriv(self, x):
    return x * (1 - x)

class MLPClassifier:
  """
  mlp classifier
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mlp"
    self.max_iterations = max_iterations
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    rate = 0.5
        hidden = 4
        w1 = 2 * np.random.random((3, hidden)) - 1
        w2 = 2 * np.random.random((hidden,1)) - 1
        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                level1 = sigmoid(np.dot(trainingData[i], w1))
                level2 = sigmoid(np.dot(level1, w2))

                level2_error = level2 - trainingLabels[i]
                level2_delta = level2_error * sigmoid_deriv(level2)
                level1_delta = level2_delta.dot(w2.T) * sigmoid_deriv(level1)

                w1 -= (rate * level1.T.dot(level2_delta))
                w2 -= (rate * trainingData[i].T.dot(level1_delta))
    
  def classify(self, data ):
    guesses = []
    for datum in data:
      # fill predictions in the guesses list
      "*** YOUR CODE HERE ***"
      util.raiseNotDefined()
    return guesses

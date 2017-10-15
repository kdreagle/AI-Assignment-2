# svm.py
# -------------

# svm implementation
import util
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
PRINT = True

class SVMClassifier:
    """
    svm classifier
    """
    def __init__( self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "svm"
        self.svm = OneVsRestClassifier(LinearSVC())

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        iteration = 0
        print "Starting iteration ", iteration, "..."
        X = []
        y = []

        for i in range(len(trainingData)):
            X.append([])
            y.append(trainingLabels[i])
            for point in trainingData[i]:
                X[i].append(trainingData[i][point])

        self.svm.fit(X,y)

    def classify(self, data):
        
        X = []
        for i in range(len(data)):
            X.append([])
            for point in data[i]:
                X[i].append(data[i][point])

        guesses = self.svm.predict(X)

        return guesses

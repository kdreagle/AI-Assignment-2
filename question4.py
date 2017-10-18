import matplotlib.pyplot as plt
import numpy as np
from random import randint

def i2():
    return (-w[0]-i1*w[1])/w[2]

def err(point):
    if point in negClass:
        return -1 - classify(point)
    else:
        return 1 - classify(point)

def classify(point):
    if point in data[0]:
        if w[0] + point[0]*w[1]+point[1]*w[2] < 0:
            return 1
        else:
            return 0
    if w[0] + point[0]*w[1]+point[1]*w[2] > 0:
        return 1
    else:
        return 0

def updateWeights(point):
    w[0] += alpha*err(point)
    for i in range(1,len(w)):
        w[i] += alpha*err(point)*point[i-1]


# initialize the data
classes = [0,1]
negLabels = ['A','B','C','D']
posLabels = ['E','F','G','H']
labels = [negLabels,posLabels]
negClass = [(0.11,1),(0.35,0.96),(0.72,0.66),(0.93,0.45)]
posClass= [(0.08,0.72),(0.28,0.57),(0.44,0.15),(0.6,0.31)]
data = [negClass,posClass]
w = [0.2,1,-1]
alpha = 1

# plot the points
for i in range(len(negClass)):
    plt.plot(negClass[i][0],negClass[i][1],'ro')
    plt.annotate(negLabels[i],negClass[i])
for i in range(len(posClass)):
    plt.plot(posClass[i][0],posClass[i][1],'bo')
    plt.annotate(posLabels[i],posClass[i])


i1 = np.arange(0.,1.,0.01)

plt.plot(i1,i2())

misclassified_points = [1]

i = 0
while len(misclassified_points) != 0 and i<100:

    misclassified_points = []
    mc_labels = []

    # determine misclassified points
    for y in classes:
        for x in range(len(data[y])):
            if classify(data[y][x]) == 0:
                misclassified_points.append(data[y][x])
                mc_labels.append(labels[y][x])


    # update the weights if any points were misclassified and draw the new separator
    if len(misclassified_points) > 0:
        updateWeights(misclassified_points[randint(0,len(misclassified_points)-1)])
        plt.plot(i1,i2())
        print("Iteration",i+1)
        print("Missclassified points:",mc_labels)
        print("New weight vector:",w)


    i += 1

plt.axis([0,1.1,0,1.1])
plt.show()



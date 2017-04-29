import pandas as pd
import numpy as np
import matplotlib as plt
from pylab import *

# a
d = '/Users/mikejaron/Google Drive/QMSS/Machine_Learning/hw2/'
X_train = pd.read_csv(d+'hw2-data/X_train.csv', header=None)
X_test = pd.read_csv(d+'hw2-data/X_test.csv', header=None)
y_train = pd.read_csv(d+'hw2-data/y_train.csv', header=None)
y_test = pd.read_csv(d+'hw2-data/y_test.csv', header=None)
names = open(d+'hw2-data/spambase.names', 'rb')

def the(cl):
    y = y_train[y_train[0] == cl]
    ix_ = y.index.tolist()
    x = X_train.ix[ix_]
    x = x.reset_index(drop=True)
    class_prior = len(y)/float(len(y_train))

    thetas = []
    for i in range(X_train.shape[1]):
        if i <= 53:
            theta = (sum(x[i]) / float(len(x)))
        else:
            theta = (float(len(x)) / sum(np.log(x[i])))
        thetas.append(theta)
    
    return thetas, class_prior


def pred(thetas, class_prior):
    probs = []
    for j in range(len(X_test)):
        prob = 1
        for i in range(X_test.shape[1]):
            if i <= 53:
                prob *= (thetas[i]**X_test[i][j]) * (1-thetas[i])**(1-X_test[i][j])
            else:
                prob *= thetas[i] * X_test[i][j]**-(thetas[i] + 1)
        probs.append(prob * class_prior)
    
    return probs

thetas0, class_prior0 = the(0)
thetas1, class_prior1 = the(1)

probs0 = pred(thetas0, class_prior0)
probs1 = pred(thetas1, class_prior1)

pred = []
for i,j in zip(probs0, probs1):
    if i > j:
        pred.append(0)
    else:
        pred.append(1)

a = 0
b = 0
c = 0
d = 0
for i,j in zip(pred, y_test[0]):
    if i == 1 and j == 1:
        a += 1
    elif i < j:
        b += 1
    elif i > j:
        c += 1
    else:
        d += 1

class_table = pd.DataFrame()
class_table[0] = [d,b]
class_table[1] = [c,a]

print 'Accuracy = ', (54+ 32)/93.
print ''
print "  y'"
print class_table





# b, stem plot
markerline, stemlines, baseline = stem(range(54), thetas0[:54], '-.')
setp(markerline, 'markerfacecolor', 'b', label='class 0')
plt.title("Class 0")

markerline, stemlines, baseline = stem(range(54), thetas1[:54], '-.')
setp(markerline, 'markerfacecolor', 'r', label='class 1')
plt.title("Theta for Class 0 and 1")
plt.ylabel('theta')
plt.xlabel('features')
plt.legend(loc='upper left')
plt.savefig('thetas.png', dpi=500)





# c, knn
from operator import itemgetter


acc = []
for k in range(1,21):
	print k
	pred = []
	for i in range(len(X_test)):
		print i
		dist = []
		for j in range(len(X_train)):
		    dist.append((j,[sum(abs(x-y)) for x,y in zip(np.array(X_test.iloc[[i]]),np.array(X_train.iloc[[j]]))] ))
		dist_s = sorted(dist,key=itemgetter(1))
		class1 = 0
		class0 = 1
		for nb in dist_s[:k]:
		    c = np.array(y_train.loc[[nb[0]]])
		    if c == 1:
		        class1 += 1
		    else:
		        class0 += 1
		if class1 > class0:
		    pred.append(1)
		else:
		    pred.append(0)
	correct = 0
	for y,y1 in zip(y_test,pred):
		if y == y1:
		    correct += 1
	acc.append(float(correct) / len(y_test))
	print float(correct) / len(y_test), len(pred)


# d






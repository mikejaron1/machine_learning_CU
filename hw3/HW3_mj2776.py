import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
np.random.seed(7)

d = '/Users/mikejaron/Google Drive/QMSS/Machine_Learning/hw3/data/'

# Problem 1

type_ = 'gaussian_process'
X_train = pd.read_csv(d + type_ + '/X_train.csv', header=None)
X_test = pd.read_csv(d + type_ + '/X_test.csv', header=None)
y_train = pd.read_csv(d + type_ + '/y_train.csv', header=None)
y_test = pd.read_csv(d + type_ + '/y_test.csv', header=None)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# a)
def kernals(x1, x2, b):
    kernal = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            kernal[i][j] = np.exp((-1. / b) * (np.linalg.norm([x1[i]] - x2[j], 2) ** 2))
    return kernal

def predict(k_XX, k_Xx, var):
    a = np.linalg.inv(var * np.identity(len(k_XX)) + k_XX) 
    mu = np.dot(k_Xx, np.dot(a, y_train))
    return mu


# b)
    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

b = [5, 7, 9, 11, 13, 15]
var = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

df = pd.DataFrame()
df['var'] = var
df = df.set_index('var')
df['b'] = [''] * len(var)

for bi in b:
    rm = []
    k_XX = kernals(X_train, X_train, bi)
    k_Xx = kernals(X_test, X_train, bi)
    for vi in var:
        mu = predict(k_XX, k_Xx, vi)
        rm.append(rmse(mu, y_test))
    df[bi] = rm

# print df



# d)
b = 5
var = 2
x4_train = X_train[:,3]

k_XX = kernals(x4_train, x4_train, b)
mu = predict(k_XX, k_XX, var)

plt.scatter(x4_train, y_train, label='Training data')
plt.plot(sorted(x4_train, reverse=True), sorted(mu), label='Predicted')
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.legend()
plt.savefig('hw3_1d.png')
plt.close()



# Problem 2

type_ = 'boosting'
X_train = pd.read_csv(d + type_ + '/X_train.csv', header=None)
X_test = pd.read_csv(d + type_ + '/X_test.csv', header=None)
y_train = pd.read_csv(d + type_ + '/y_train.csv', header=None)
y_test = pd.read_csv(d + type_ + '/y_test.csv', header=None)

X_train[5] = [1] * len(X_train)
X_test[5] = [1] * len(X_test)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# a)

def ls(X, y):
    return (np.linalg.inv(X.T.dot(X))).dot(X.T.dot(y))

def pred(X, y, wt, weights):
    ypred = np.sign(X.dot(wt))
    idx = np.argwhere(ypred != y)
    idx = idx[:, 0]
    eps = np.sum(weights[idx])
    
    return eps, ypred

def errors(X, y, wt, alpha, sums):
    ypred = np.sign(X.dot(wt))
    sums += alpha * ypred
    pred_test = np.sign(sums)
    error = np.sum(pred_test != y)
    error_rate = float(error) / len(X)
    
    return error_rate, sums

np.random.seed(7)

alpha_t = []
eps_t = []
testing_error = []
training_error = []
bound_t = []

sums1 = 0
sums2 = 0
T = 1500

weights = np.array([1. / len(X_train)] * len(X_train))
df = pd.DataFrame()

for t in range(1, T + 1):
    B_idx = np.random.choice(len(X_train), len(X_train), replace=True, p=weights)
    
    df[t] = B_idx
    
    B_x = X_train[B_idx]
    B_y = y_train[B_idx]

    wt = ls(B_x, B_y)
    
    eps, ypred = pred(X_train, y_train, wt, weights)
    
    if eps > .5:
        wt = wt * -1
        eps, ypred = pred(X_train, y_train, wt, weights)
    eps_t.append(eps)
    
    alpha = .5 * np.log((1. - eps) / eps)
    alpha_t.append(alpha)

    weights = weights.reshape(1036, 1) * np.exp((-1 * alpha * y_train * ypred))
    weights = weights / np.sum(weights)
    weights = weights.flatten()

    bound = 0
    for z in range(0, t):
        bound += np.square(0.5 - eps_t[z])
    bound_t.append(np.exp(-2 * bound))
    
    error_rate, sums1 = errors(X_test, y_test, wt, alpha, sums1)
    testing_error.append(error_rate)
    
    error_rate, sums2 = errors(X_train, y_train, wt, alpha, sums2)
    training_error.append(error_rate)

   
plt.plot(range(T), testing_error, label='test error')
plt.plot(range(T), training_error, label='train error')
plt.legend()
plt.ylabel('Error')
plt.xlabel('T - Iterations')
plt.legend()
plt.savefig('hw3_2a.png')
plt.close()

# b)
plt.plot()
plt.plot(range(T), bound_t)
plt.axis((0, 1500, 0, .5))
plt.xlabel('T - Iterations')
plt.ylabel('Upper Bound')
plt.savefig('hw3_2b.png')
plt.close()

# c)
df_m = np.array(df)
hist_di = {}
for i in range(len(df_m)):
    for j in range(df_m.shape[1]):
        if df_m[i][j] in hist_di:
            hist_di[df_m[i][j]] += 1
        else:
            hist_di[df_m[i][j]] = 1
            
plt.bar(hist_di.keys(), hist_di.values())
plt.xlabel('Point')
plt.ylabel('# of times selected')
plt.savefig('hw3_2c.png')
plt.close()


# d)
plt.scatter(range(T), alpha_t)
plt.ylabel('Alpha')
plt.xlabel('T - iterations')
plt.savefig('hw3_2d1.png')
plt.close()

plt.scatter(range(T), eps_t)
plt.ylabel('Epsilon')
plt.xlabel('T - iterations')
plt.savefig('hw3_2d2.png')
plt.close()

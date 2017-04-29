'''
Michael Jaron, mj2776
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_train = pd.read_csv('./hw1-data/X_train.csv', header=None)
X_test = pd.read_csv('./hw1-data/X_test.csv', header=None)
y_train = pd.read_csv('./hw1-data/y_train.csv', header=None)
y_test = pd.read_csv('./hw1-data/y_test.csv', header=None)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


def RR(lamda, I, X_train, y_train):
    a = lamda*I + (X_train.transpose().dot(X_train))
    b = np.linalg.pinv(a).dot(X_train.transpose()).dot(y_train)
    
    return b

def RMSE(rr, X_test, y_test):
    a = np.sqrt(np.sum((y_test - X_test.dot(rr))**2)/len(y_test))
    
    return a

## part 2a

I = np.identity(7)
U,S,V = np.linalg.svd(X_train, full_matrices=False)

df = []
for lamda in range(5001):
    df.append(sum(S**2/(lamda+S**2)))
    
w_rr = np.empty((7,5001))
for lamda in range(5001):
    rr = RR(lamda, I, X_train, y_train)
    w_rr[:,lamda] = list(rr)

plt.figure(figsize=(8, 6))
for i in range(7):
    plt.plot(df, w_rr[i], label='x'+str(i+1))
    
plt.ylabel('lambda')
plt.xlabel('degrees of freedom')
plt.title('w_rr vs lambda')
plt.legend(loc='lower left')
plt.savefig('w_rr_lambda.png')



## Part 2b

rmse = []
for lamda in range(51):
    rr = RR(lamda, I, X_train, y_train)
    b = RMSE(rr, X_test, y_test)
    rmse.append(b) 

plt.figure(figsize=(8, 6))
plt.plot(np.array(range(51)), rmse)
    
plt.ylabel('RMSE')
plt.xlabel('lambda')
plt.title('RMSE vs lambda')
plt.savefig('rmse_lambda.png')


## Part 2d

X_test_2 = np.append(X_test, X_test[:,:6]**2, 1)
X_train_2 = np.append(X_train, X_train[:,:6]**2, 1)
X_test_3 = np.append(X_test_2, X_test[:,:6]**3, 1)
X_train_3 = np.append(X_train_2, X_train[:,:6]**3, 1)

rmse1 = []
rmse2 = []
rmse3 = []

print X_test.shape, X_test_2.shape, X_test_3.shape

for lamda in range(501):
    rr = RR(lamda, I, X_train, y_train)
    b = RMSE(rr, X_test, y_test)
    rmse1.append(b)
    
    rr = RR(lamda, np.identity(13), X_train_2, y_train)
    b = RMSE(rr, X_test_2, y_test)
    rmse2.append(b)
    
    rr = RR(lamda, np.identity(19), X_train_3, y_train)
    b = RMSE(rr, X_test_3, y_test)
    rmse3.append(b)

plt.figure(figsize=(8, 6))
plt.plot(np.array(range(501)), rmse1, label='p=1')
plt.plot(np.array(range(501)), rmse2, label='p=2')
plt.plot(np.array(range(501)), rmse3, label='p=3')
    
plt.ylabel('RMSE')
plt.xlabel('lambda')
plt.title('RMSE vs lambda')
plt.legend(loc='upper left')
plt.savefig('rmse_lambda_pth.png')


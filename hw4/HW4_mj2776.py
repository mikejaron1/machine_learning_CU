import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
# %matplotlib inline
dire = '/Users/mikejaron/Google Drive/QMSS/Machine_Learning/hw4/data/'

# 1
pi = [.2, .5, .3]

mu1 = np.array([1, 1])
cov1 = np.matrix('1, 0; 0, 1')

mu2 = np.array([3, 0])
cov2 = np.matrix('1, 0; 0, 1')

mu3 = np.array([0, 3])
cov3 = np.matrix('1, 0; 0, 1')

mus = [mu1, mu2, mu3]
covs = [cov1, cov2, cov3]

# generate data
x = np.array([])
y = np.array([])
for weight, mu, cov in zip(pi, mus, covs):
    x1, y1 = np.random.multivariate_normal(mu, cov, size=int(500 * weight)).T
    x = np.hstack((x, x1))
    y = np.hstack((y, y1))
print x.shape, y.shape
X = np.stack((x, y)).T
print X.shape
# plt.scatter(X, Y)
# plt.axis('equal')
# plt.show()

# a)
ks = [2, 3, 4, 5]
k_dict = defaultdict(list)
point_dict = defaultdict(list)
for k in ks:
    print 'k=', k
    idx = np.random.randint(500, size=(k))
    centroids = X[idx]
    for i in range(20):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        c = np.argmin(distances, axis=0)
        centroids = []
        L = 0
        for ki in range(k):
            L += np.sum(distances[ki][c == ki])
            centroids.append(X[c == ki].mean(axis=0))
        centroids = np.array(centroids)
        k_dict[k].append(L)
        
    # this is for 1b
    if k == 3 or k == 5:
        point_dict[k].append(c)
            

# a)
plt.plot(range(20), k_dict[2], label='K = 2')
plt.plot(range(20), k_dict[3], label='K = 3')
plt.plot(range(20), k_dict[4], label='K = 4')
plt.plot(range(20), k_dict[5], label='K = 5')
plt.ylabel('L')
plt.xlabel('Iteration')
plt.title('Convergence of K-means')
plt.legend()
# plt.show()
plt.savefig('hw4_1a.png')
plt.close()


# b)
plt.scatter(X[point_dict[3][0] == 0][:, 0], X[point_dict[3][0] == 0][:, 1], color='b')
plt.scatter(X[point_dict[3][0] == 1][:, 0], X[point_dict[3][0] == 1][:, 1], color='g')
plt.scatter(X[point_dict[3][0] == 2][:, 0], X[point_dict[3][0] == 2][:, 1], color='r')
plt.title('K = 3')
plt.legend()
# plt.show()
plt.savefig('hw4_1b1.png')
plt.close()



# b)
plt.scatter(X[point_dict[5][0]==0][:,0], X[point_dict[5][0]==0][:,1], color='g')
plt.scatter(X[point_dict[5][0]==1][:,0], X[point_dict[5][0]==1][:,1], color='b')
plt.scatter(X[point_dict[5][0]==2][:,0], X[point_dict[5][0]==2][:,1], color='c')
plt.scatter(X[point_dict[5][0]==3][:,0], X[point_dict[5][0]==3][:,1], color='r')
plt.scatter(X[point_dict[5][0]==4][:,0], X[point_dict[5][0]==4][:,1], color='y')
plt.title('K = 5')
plt.legend()
# plt.show()
plt.savefig('hw4_1b2.png')
plt.close()




# 2
# get movies
movies_text = open(dire + '/movies.txt').read()
ratings_test = pd.read_csv(dire + '/ratings_test.csv', header=None)
ratings = pd.read_csv(dire + '/ratings.csv', header=None)

# turn text to dataframe
movies = pd.DataFrame()
movies['title'] = movies_text.split('\n')

# add column names
ratings_test.columns = ['user_id', 'movie_id', 'rating']
ratings.columns = ['user_id', 'movie_id', 'rating']

# turn dataframes into numop matrix
# movies = np.array(movies)
# ratings_test = np.array(ratings_test)
# ratings = np.array(ratings)


print ratings.head()
print movies.head()
# print ratings_test.head()
print ratings.shape, ratings_test.shape, movies.shape
# print len(set(ratings['user_id'])), len(set(ratings['movie_id']))

var = .25
d = 10
lamda = 1

# initiate M
N1 = len(set(ratings['user_id']))
N2 = len(movies)
M = np.zeros([N1, N2])
# M.fill(np.nan)
for i in range(len(ratings)):
    M[ratings['user_id'][i] - 1][ratings['movie_id'][i] - 1] = ratings['rating'][i]
print M.shape




def u_v():
    u = np.random.normal(0, 1, N1)
    v = np.random.normal(0, 1, N2)
    for it in range(9):
        ui = np.random.normal(0, 1, N1)
        vj = np.random.normal(0, 1, N2)
        u = np.vstack((u, ui))
        v = np.vstack((v, vj))
    return u.T, v


# a)

L_dict1 = defaultdict(list)
errors_dict1 = defaultdict(list)
pred_dict = defaultdict(list)
for t in range(10):
    u, v = u_v()
    for ii in range(100):
        u = np.linalg.pinv((v.dot(v.T)) + lamda * var * np.eye(d)).dot((M.dot(v.T)).T)
        v = np.linalg.pinv((u.dot(u.T)) + lamda * var * np.eye(d)).dot((u.dot(M)))
        L = -1. * np.sum((1. / (2 * var)) * (M - u.T.dot(v)) ** 2) - np.sum((lamda/2.) * u**2) - np.sum((lamda/2.) * v**2)
        L_dict1[t].append(L)
        errors_dict1[t].append(np.sqrt(((M - u.T.dot(v)) ** 2).mean()))
    pred_dict[t] = u.T.dot(v)


for i in range(10):
    plt.plot(range(100), L_dict1[i], label=str(i))
plt.title('Log Joint Likelihood, 2-100')
plt.ylabel('Log Joint Likelihood')
plt.xlabel('Iteration')
plt.axis((2, 100, -197000, -190000))
plt.legend(loc='lower right')
plt.savefig('hw4_2a.png', bbox_inches='tight')
# plt.show()


# for i in range(10):
#     plt.plot(range(100), errors_dict1[i], label=str(i))
# plt.legend()
# plt.show()

# initiate M test
# doesnt work well
# Mt = np.zeros([N1, N2])
# for i in range(len(ratings_test)):
#     Mt[ratings_test['user_id'][i] - 1][ratings_test['movie_id'][i] - 1] = ratings_test['rating'][i]
# print Mt.shape

# # create an indicator matrix 
# Ind = Mt != 0.
# Ind = Ind.astype(np.float64, copy=False)
# print Ind.shape

# RMSE = []
# ljl = []
# for i in pred_dict:
#     pred = np.multiply(pred_dict[i], Ind)
#     RMSE.append(np.sqrt(((Mt - pred) ** 2).mean()))
#     ljl.append(L_dict1[i][-1])
    
# e_df = pd.DataFrame()
# e_df['log joint likelihood'] = ljl
# e_df['RMSE'] = RMSE

# print e_df

rmse = []
ljl = []
for it in pred_dict:
    pred = pred_dict[it]
    diffs = []
    for i, j in zip(ratings_test['user_id'], ratings_test['movie_id']):
        diffs.append((ratings_test['rating'][i - 1] - pred[i - 1][j - 1]) ** 2)

    rmse.append(np.sqrt(np.mean(diffs)))
    ljl.append(L_dict1[it][-1])

e_df = pd.DataFrame()
e_df['log joint likelihood'] = ljl
e_df['RMSE'] = rmse
e_df = e_df.sort_values(by='log joint likelihood', ascending=False)
e_df = e_df.reset_index(drop=True)
print e_df

# 2b)
# star wars = 50
# my fair lady = 485
# goodfellas = 182

pred = u.T.dot(v)


def top_ten(pred, main_movie):
    sw_dist = {}
    for j in range(pred.shape[1]):
        sw_dist[np.sqrt(((main_movie - pred[:, j])**2).sum())] = j
    top_ten_idx = sorted(sw_dist.keys())[:11]
    top_ten_movies = []
    for i in top_ten_idx:
        idx = sw_dist[i]
        top_ten_movies.append(movies['title'][idx])

    df = pd.DataFrame()
    df['movies'] = top_ten_movies
    df['dist'] = top_ten_idx
    
    return df

df_SW = top_ten(pred, pred[:, 49])
df_MFL = top_ten(pred, pred[:, 484])
df_GF = top_ten(pred, pred[:, 181])

print df_SW
print df_MFL
print df_GF

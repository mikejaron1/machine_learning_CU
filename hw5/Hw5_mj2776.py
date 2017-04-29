import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

dire = '/Users/mikejaron/Google Drive/QMSS/Machine_Learning/hw5/'

# 1)
# Team A index, Team A points, Team B index, Team B points
teams = open(dire + 'TeamNames.txt').readlines()
df = pd.read_csv(dire + 'CFB2016_scores.csv', header=None)
# print df.shape
# df.head()

# update M

M_hat = np.zeros((760, 760))
for i, score_i, j, score_j in zip(df[0], df[1], df[2], df[3]):
    if score_i > score_j:
        ind_i = 1.
        ind_j = 0.
    else:
        ind_i = 0.
        ind_j = 1. 
    M_hat[i-1][i-1] += ind_i + (float(score_i) / (score_i + score_j))
    M_hat[j-1][j-1] += ind_j + (float(score_j) / (score_i + score_j))
    M_hat[i-1][j-1] += ind_j + (float(score_j) / (score_i + score_j))
    M_hat[j-1][i-1] += ind_i + (float(score_i) / (score_i + score_j))


# normalize rows to sum to 1
def normalize(M, axis):
    row_sums = np.sum(M, axis=axis)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if axis == 1:
                z = i
            else:
                z = j
            M[i][j] = M[i][j] / float(row_sums[z])
                
    return M

M = normalize(M_hat, axis=1)


def top25_table(wt):
    top25 = np.argsort(wt)[::-1][:25]
    team = []
    value = []
    for i in top25:
        team.append(teams[i].replace('\n', ''))
        value.append(wt[i])
        
    df = pd.DataFrame()
    df['team'] = team
    df['value'] = value
    
    return df

# a)
ts = [10, 100, 1000, 10000]
data_dict = {}
for t in ts:
    print t
    wt = np.random.uniform(low=0.0, high=1.0, size=760)
    wt = np.array([float(i) / np.sum(wt) for i in wt])
    for i in range(t):
        wt = wt.dot(M)
    data_dict[t] = top25_table(wt)

# write all data to 1 massive dataframe
new_df = pd.DataFrame()
for i in ts:
    new_df['t_' + str(i)] = [i] * len(data_dict[i])
    new_df['Team_' + str(i)] = data_dict[i]['team']
    new_df['Value_' + str(i)] = data_dict[i]['value']

new_df.to_csv('top25_teams.csv')
# print data_dict[10000]


# b)
w, u = np.linalg.eig(M.T)
# print np.argmax(w)
u = u[:,0]
w_inf = u.T / np.sum(u)
wt = np.random.uniform(low=0.0, high=1.0, size=760)
wt = wt.T / np.sum(wt)
dist = []
for t in range(10000):
    wt = wt.dot(M)
    l1 = np.sum(abs(wt - w_inf))
    dist.append(l1)
    


plt.plot(range(10000), dist)
plt.xlim([-100,10010])
plt.title('Wt - W_inf')
plt.ylabel('L1 Distance')
plt.xlabel('Iteration')
plt.savefig('hw5_1b.png')
# plt.show()
plt.close()


# 2)
vocab = open(dire + 'nyt_vocab.dat').readlines()
counts = open(dire + 'nyt_data.txt').readlines()
counts = [i.replace('\n', '').split(',') for i in counts]

X = np.zeros((len(vocab), len(counts)))
for j, doc in enumerate(counts):
    for word in doc:
        word = word.split(':')
        X[int(word[0]) - 1][j] += int(word[1])


sm_nm = .00000000000000001
K = 25
W = np.random.uniform(low=1.0, high=2.0, size=(len(vocab), K))
H = np.random.uniform(low=1.0, high=2.0, size=(K, len(counts)))

obj_list = []
for i in range(100):
    p = X / (W.dot(H) + sm_nm)
    
    Wtn = normalize(W.T, axis=1)
    H = H * Wtn.dot(p)

    p = X / (W.dot(H) + sm_nm) 
    
    Htn = normalize(H.T, axis=0)
    W = W * p.dot(Htn)
    
    WH = (W.dot(H)) + sm_nm
    obj = np.sum(X * np.log(1/(WH)) + WH)
    obj_list.append(obj)
    
    print obj, '/ iter= ', i


plt.plot(range(100), obj_list)
# plt.xlim([-100,10010])
plt.title('Divergence Penalty')
plt.ylabel('Objective Func.')
plt.xlabel('Iteration')
plt.savefig('hw5_2a.png')
# plt.show()
plt.close()


# b)
W_norm = normalize(W, axis=0)
top_words = {}
for i in range(W_norm.shape[1]):
    top10 = np.argsort(W_norm[:,i])[::-1][:10]
    word = []
    value = []
    for j in top10:
        word.append(vocab[j].replace('\n', ''))
        value.append(W_norm[j][i])

    df = pd.DataFrame()
    df['word'] = word
    df['weight'] = value
    
    top_words[i] = df


# write to 1 massive dataframe
top_words.keys()
new_df = pd.DataFrame()
for i in top_words.keys():
    new_df['word_' + str(i)] = top_words[i]['word']
    new_df['weight_' + str(i)] = top_words[i]['weight']
new_df.to_csv('top_words.csv')
top_words[0]



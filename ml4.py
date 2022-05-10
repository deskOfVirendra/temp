from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
data =pd.DataFrame({'X':[0.1,0.15,0.08,0.16,0.2,0.25,0.24,0.3],
                 'y':[0.6,0.71,0.9,0.85,0.3,0.5,0.1,0.2]})
print("Input Data and Shape")
print(data.shape)
data

f1 = data['X'].values
f2 = data['y'].values

X = np.array(list(zip(f1, f2)))

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

k = 2
C_x = np.array([0.1, 0.3])
C_y = np.array([0.6, 0.2])

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("Initial Centroids")
print(C)

C_old = np.zeros(C.shape)
print(C.shape)
clusters = np.zeros(len(X))
error = dist(C, C_old, None)
print(error)

plt.scatter(f1, f2, c='#050505', s=70)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

while error != 0:
    for i in range(len(X)):
        distances = dist(X[i], C)
        
        cluster = np.argmin(distances)
        clusters[i] = cluster

    #print("Clusters:")
    #print(clusters)      
    C_old = deepcopy(C)

    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        #print("Points :" , points)
        C[i] = np.mean(points, axis=0)
    
    #print(C)
    error = dist(C, C_old, None)

print(clusters)
print(C)
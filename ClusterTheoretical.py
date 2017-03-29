#! user/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt
import ClusterPackage as clu
#Crafting the Artificial Dataset
D=2
N1=220
N2=280

mean = np.ones(D)
def varID(scale):
	return(scale*np.identity(D)) 
X1 = np.random.multivariate_normal(mean, varID(0.5), N1).T #DxN matrix
X2 = np.random.multivariate_normal(-mean, varID(0.75), N2).T
X = np.concatenate((X1,X2), axis=1)

print(X.shape)
#plt.hist(X1.T)
#plt.hist(X2.T)
#plt.show()
#print(X.shape)
#clu.Kmeans(X)
Color, Centroids = clu.Kmeans(X) #returns the color scheme of the clusters as a list
print(Centroids.T)

plt.subplot(121)
plt.scatter(X[0],X[1], c=Color)
plt.scatter(Centroids.T[0],Centroids.T[1], c='c')
#Since we know the correct clusters, let's test the effectiveness of this algorithm
#by separating the data
plt.subplot(122)
plt.scatter(X1[0],X1[1])
plt.scatter(X2[0],X2[1], c='g')
plt.show()
